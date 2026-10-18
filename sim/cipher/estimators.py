"""CIPHER pose estimator and baselines (all use receiver-clock observables only)."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, minimize

from .geometry import mic_world, wrap
from .lagmodel import LagFit


@dataclass
class Model:
    anchors: np.ndarray            # (I,3)
    h: float
    mic_body: np.ndarray           # (M,3) nominal (or calibrated) body positions
    c: float
    fs: float
    chirp_len: float
    sigma_toa: float = 5e-6        # TOA error floor for the TDOA term (sync jitter, multipath bias)
    cal: np.ndarray = None         # (I,M,K) complex per-mic response (None -> ideal)
    mic_gain: np.ndarray = None    # (I,M) complex per-(band, mic) gain from calibration (None -> 1)
    mic_delay: np.ndarray = None   # (I,M) per-(band, mic) group delay from calibration (s)
    room_xy: tuple = ((0.0, 10.0), (0.0, 10.0))
    ceiling: float = None          # ceiling height; with it, floor+ceiling specular paths are modelled
    beta_rms: float = 400.0        # RMS bandwidth (Hz) of the compressed pulse, for TOA noise std
    body_plate: float = None       # known reflector depth below the array plane (m, from the robot's
                                   # mechanical design): modelled as one more image path (free gain)

    def sources(self, i):
        """Direct source and (optionally) its floor / ceiling images for anchor i."""
        a = self.anchors[i]
        if self.ceiling is None:
            return a[None]
        return np.stack([a, [a[0], a[1], -a[2]], [a[0], a[1], 2 * self.ceiling - a[2]]])

    def toa_std(self, psnr):
        """Per-anchor TOA std: group-delay CRB for the measured SNR, plus the floor."""
        gd = 1.0 / (2 * np.pi * self.beta_rms * np.sqrt(np.maximum(psnr * len(self.mic_body), 1.0)))
        return np.sqrt(self.sigma_toa ** 2 + gd ** 2)


# ---------------------------------------------------------------- TDOA (group delay)
def _toa_resid(theta, t, model, w=None):
    x, y, t0 = theta
    d = np.linalg.norm(model.anchors - np.array([x, y, model.h]), axis=1)
    r = (t - t0 - d / model.c) / model.sigma_toa
    return r if w is None else r * w


def tdoa_solve(t, model, valid=None):
    """Position from arrival times t (receiver clock, s); receiver offset t0 is a nuisance.
    Returns p (2,), cov (2,2), t0."""
    sel = np.ones(len(t), bool) if valid is None else valid
    A, tt = model.anchors[sel], t[sel]
    sub = Model(A, model.h, model.mic_body, model.c, model.fs, model.chirp_len, model.sigma_toa)
    (x0, x1), (y0, y1) = model.room_xy
    best = None
    for gx in np.linspace(x0, x1, 6):
        for gy in np.linspace(y0, y1, 6):
            d = np.linalg.norm(A - np.array([gx, gy, model.h]), axis=1)
            th0 = np.array([gx, gy, np.mean(tt - d / model.c)])
            sol = least_squares(_toa_resid, th0, args=(tt, sub), method='lm')
            if best is None or sol.cost < best.cost:
                best = sol
    Jac = best.jac
    cov = np.linalg.pinv(Jac.T @ Jac)
    return best.x[:2], cov[:2, :2], best.x[2]


def tdoa_solve_2frame(obs_a, obs_b, model, pulses, frame_len):
    """Doppler-robust gate from two consecutive frames with opposite chirp directions per anchor
    (checker scheme). First-path times t_ik = t_rx + k T_f + |p + v (k - 1/2) T_f - a_i| / c
    + kappa_ik (u_i . v) / c  (kappa = d(peak)/d(beta), sign flips with the sweep direction).
    Unknowns (x, y, vx, vy, t_rx): 8 equations for 4 anchors. Returns p at each frame and v."""
    A = model.anchors
    rows = []
    for k, obs in enumerate((obs_a, obs_b)):
        for i in np.where(obs.valid)[0]:
            rows.append((k, i, obs.n_arr[i] / model.fs, pulses.kappa(i, obs.dirs[i])))
    if len(rows) < 6:
        return None

    def res(th):
        x, y, vx, vy, trx = th
        out = []
        for k, i, t, kap in rows:
            pk = np.array([x, y]) + np.array([vx, vy]) * (k - 0.5) * frame_len
            d = A[i] - np.array([pk[0], pk[1], model.h])
            u = d / np.linalg.norm(d)
            out.append((t - trx - k * frame_len - np.linalg.norm(d) / model.c
                        - kap * (u[0] * vx + u[1] * vy) / model.c) / model.sigma_toa)
        return np.array(out)

    (x0, x1), (y0, y1) = model.room_xy
    best = None
    t_first = np.mean([r[2] for r in rows if r[0] == 0])
    for gx in np.linspace(x0, x1, 6):
        for gy in np.linspace(y0, y1, 6):
            th0 = np.array([gx, gy, 0.0, 0.0, t_first - 0.02])
            sol = least_squares(res, th0, method='lm')
            if best is None or sol.cost < best.cost:
                best = sol
    x, y, vx, vy, _ = best.x
    v = np.array([vx, vy])
    p = np.array([x, y])
    return p - v * frame_len / 2, p + v * frame_len / 2, v


# ---------------------------------------------------------------- coherent (clock-free) term
def _fit_paths(A, Y, n_iter=6):
    """Multipath-aware coherence. A (L,N,M,K) steering of direct + image paths (L>=1), Y (M,K).
    Model Y_k = C_k (a0_k + sum_l rho_l a_l,k): C_k per-bin nuisance (clocks, speaker response),
    rho frequency-flat reflection gains. Alternating LS; returns the fitted energy (N,)."""
    L, N, M, K = A.shape
    a0 = A[0]
    if L == 1:
        return np.sum(np.abs(np.einsum('nmk,mk->nk', np.conj(a0), Y)) ** 2, -1) / M
    rho = np.zeros((N, L - 1), complex)
    for _ in range(n_iter):
        b = a0 + np.einsum('nl,lnmk->nmk', rho, A[1:])
        C = np.einsum('nmk,mk->nk', np.conj(b), Y) / np.sum(np.abs(b) ** 2, 1)       # (N,K)
        R = Y[None] - C[:, None] * a0                                                 # (N,M,K)
        Z = C[None, :, None] * A[1:]                                                  # (L-1,N,M,K)
        G = np.einsum('lnmk,pnmk->nlp', np.conj(Z), Z)                               # (N,L-1,L-1)
        h = np.einsum('lnmk,nmk->nl', np.conj(Z), R)
        rho = np.linalg.solve(G + 1e-9 * np.eye(L - 1), h[..., None])[..., 0]
    b = a0 + np.einsum('nl,lnmk->nmk', rho, A[1:])
    return np.sum(np.abs(np.einsum('nmk,mk->nk', np.conj(b), Y)) ** 2 / np.sum(np.abs(b) ** 2, 1), -1)


def _robust_mean(x):
    """Mean without the smallest value (one bad anchor tolerated)."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    return float(x[1:].mean()) if len(x) > 2 else float(x.mean()) if len(x) else np.nan


def coherent(model, obs, P, PSI, dt=None, vel=(0.0, 0.0), omega=0.0, normalized=True,
             multipath=True):
    """Per-anchor clock-free coherence J (N,I) in [0,1], or log-likelihood M*SNR_i*J (N,I).
    P (N,2), PSI (N,). dt (I,) epoch of each anchor's observation minus reference epoch.
    multipath: include the floor/ceiling specular paths if the model has a ceiling height."""
    P = np.atleast_2d(P)
    PSI = np.atleast_1d(PSI)
    I, M, K = obs.Y.shape
    dt = np.zeros(I) if dt is None else dt
    out = np.zeros((len(P), I))
    for i in range(I):
        if not obs.valid[i]:
            continue
        pi = P + np.asarray(vel)[None] * dt[i]
        q = mic_world(pi, PSI + omega * dt[i], model.h, model.mic_body)          # (N,M,3)
        src = model.sources(i) if multipath else model.anchors[i][None]
        tau = np.linalg.norm(q[None] - src[:, None, None], axis=-1) / model.c    # (L,N,M)
        A = np.exp(-2j * np.pi * tau[..., None] * obs.freqs[i])                   # (L,N,M,K)
        Yi = obs.Y[i] if model.cal is None else obs.Y[i] * np.conj(model.cal[i])
        if model.mic_gain is not None:
            Yi = Yi / model.mic_gain[i][:, None]
        J = _fit_paths(A, Yi) / np.sum(np.abs(Yi) ** 2)
        out[:, i] = J if normalized else M * obs.peak_snr[i] * J
    return out


def _peaks_circ(v, grid, min_sep):
    n = len(v)
    idx = [k for k in range(n) if v[k] >= v[k - 1] and v[k] > v[(k + 1) % n]]
    idx.sort(key=lambda k: -v[k])
    keep = []
    for k in idx:
        if all(abs(wrap(grid[k] - grid[j])) > min_sep for j in keep):
            keep.append(k)
    return keep


@dataclass
class Estimate:
    p: np.ndarray
    psi: float
    ratio: float = np.nan
    coh: float = np.nan
    p_tdoa: np.ndarray = None
    extra: dict = field(default_factory=dict)


def cipher(model, obs, dt=None, vel=(0.0, 0.0), omega=0.0, prior=None, n_cand=3,
           psi_step_deg=0.5, prior_halfwidth_deg=4.0, lag=None, estimate_vel=True,
           vel_sigma=0.02, use_tdoa=True, gate=None, estimate_omega=True, omega_sigma=0.3,
           eta=0.0, eta_sigma=100e-6, estimate_c=True, c_sigma=5e-3, n_paths=10):
    """CIPHER: TDOA gate -> heading ambiguity search (+ratio test) -> coherent ML refinement.
    prior = (p, psi) from tracking: restricts the heading search to +-prior_halfwidth_deg.
    lag = (pulses, room_size, multipath): refine with the lag-domain multipath model instead of
    the per-bin gated-spectrum model.
    vel / vel_sigma: velocity prior (from tracking; cold start assumes a slow start, 0 +- 2 cm/s).
    The velocity is a parameter because the up-chirp envelope (hence the sync constraint) depends
    on it through the range-Doppler coupling."""
    t = obs.n_arr / model.fs
    p0, cov_p, _ = tdoa_solve(t, model, obs.valid)
    if gate is not None:                   # externally supplied gate (e.g. two-frame Doppler-robust)
        p0 = np.asarray(gate, float)
    if prior is None:
        grid = np.deg2rad(np.arange(0, 360, psi_step_deg))
        # leave-one-out TDOA gates: a single anchor whose first path was missed (e.g. robot under
        # its beam) can throw the gate off by metres; the pose-domain coherence decides
        gates = [p0]
        if gate is None and obs.valid.sum() >= 4:
            for i in np.where(obs.valid)[0]:
                v = obs.valid.copy()
                v[i] = False
                gates.append(tdoa_solve(t, model, v)[0])
        best = None
        for g in gates:
            lam_g = coherent(model, obs, np.repeat(g[None], len(grid), 0), grid, dt, vel, omega,
                             multipath=False).mean(1)
            if best is None or lam_g.max() > best[1].max():
                best = (g, lam_g)
        p0, lam = best
    else:
        # tracking: the prediction (mm, ~0.05 deg) is far inside one lobe -> no lobe re-selection
        # (a +-few-degree search can jump to the neighbouring pose-domain sidelobe, sec. 5.2)
        grid = np.array([prior[1]])
        p0 = np.asarray(prior[0], float)
        lam = coherent(model, obs, p0[None], grid, dt, vel, omega, multipath=False).mean(1)
    pk = _peaks_circ(lam, grid, np.deg2rad(3.0)) if prior is None else [0]
    ratio = lam[pk[0]] / lam[pk[1]] if len(pk) > 1 else np.inf

    t_obs = t[obs.valid]
    A = model.anchors[obs.valid]
    w_t = 1.0 / model.toa_std(obs.peak_snr[obs.valid]) ** 2

    def negloglik(z, scale):
        x, y, psi = z[0] * 1e-3, z[1] * 1e-3, z[2] * 1e-3
        L = coherent(model, obs, np.array([[x, y]]), np.array([psi]), dt, vel, omega,
                     normalized=False).sum()
        d = np.linalg.norm(A - np.array([x, y, model.h]), axis=1)
        r = t_obs - d / model.c
        r = r - np.sum(w_t * r) / np.sum(w_t)          # unknown receiver offset t0
        return -(L - 0.5 * np.sum(w_t * r ** 2)) / scale

    cands = [k for k in pk[:n_cand] if lam[k] >= lam[pk[0]] / 1.3]   # only plausible lobes
    if lag is not None:                    # lag-domain variable-projection refinement
        pulses, room_size, mp = lag
        best = None
        for k in cands:
            lf = LagFit(model, obs, pulses, room_size, p0, multipath=mp, vel=vel, omega=omega,
                        estimate_vel=estimate_vel, vel_sigma=vel_sigma, use_tdoa=use_tdoa,
                        estimate_omega=estimate_omega, omega_sigma=omega_sigma,
                        estimate_eta=use_tdoa, eta=eta, eta_sigma=eta_sigma,
                        estimate_c=estimate_c, c_sigma=c_sigma, n_paths=n_paths)
            sol = least_squares(lf.residuals, lf.z0(p0, grid[k]), method='lm', x_scale='jac')
            lf.reweight(sol.x)
            sol = least_squares(lf.residuals, sol.x, method='lm', x_scale='jac')
            if best is None or sol.cost < best[0]:
                best = (sol.cost, sol.x, sol.jac, lf)
        z, lf = best[1], best[3]
        try:                                            # pose covariance (m, m, rad) from the LM Jacobian
            cfull = np.linalg.inv(best[2].T @ best[2])
            cz = cfull[:3, :3] * np.outer([1e-3] * 3, [1e-3] * 3)
            c_var = cfull[-1, -1] * 1e-6 if estimate_c else np.nan
        except np.linalg.LinAlgError:
            cz, c_var = np.full((3, 3), np.nan), np.nan
        lf.residuals(z)                                 # leaves the direct-path gains at z
        p, psi = z[:2] * 1e-3, wrap(z[2] * 1e-3)
        coh_i = coherent(model, obs, p[None], np.array([psi]), dt, vel, omega, multipath=False)[0]
        coh = coh_i.mean()
        v_m, w_m, e_m = lf._motion(z)
        t0 = np.full(len(model.anchors), np.nan)
        t0[lf.idx] = z[3:3 + len(lf.idx)] * 1e-6
        v_est = v_m if v_m is not None else np.asarray(vel, float)
        w_est = w_m if w_m is not None else float(omega)
        return Estimate(p, float(psi), ratio, float(coh), p0,
                        dict(cov_tdoa=cov_p, lam=lam, grid=grid, t_ref=lf.t_ref, vel=v_est,
                             omega=w_est, eta=e_m if e_m is not None else eta,
                             t0=t0, c_direct=lf.c_direct.copy(), c_corr=lf.c_corr.copy(), cov_pose=cz, lagfit=lf, z=z,
                             c_scale=lf.c_scale(z), c_var=c_var,
                             coh_robust=_robust_mean(coh_i[obs.valid])))

    best = None
    for k in pk[:n_cand]:
        z0 = np.array([p0[0] * 1e3, p0[1] * 1e3, grid[k] * 1e3])
        scale = max(abs(negloglik(z0, 1.0)), 1.0)
        simplex = z0 + np.array([[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 5]])
        sol = minimize(negloglik, z0, args=(scale,), method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-12, maxiter=4000))
        val = sol.fun * scale
        if best is None or val < best[0]:
            best = (val, sol.x)
    z = best[1]
    p, psi = z[:2] * 1e-3, wrap(z[2] * 1e-3)
    coh = coherent(model, obs, p[None], np.array([psi]), dt, vel, omega).mean()
    return Estimate(p, float(psi), ratio, float(coh), p0, dict(cov_tdoa=cov_p, lam=lam, grid=grid))


def gate_shifts(model, p, psi):
    """Inter-mic delays (samples, relative to the array mean) predicted at pose (p, psi)."""
    q = mic_world(np.asarray(p)[None], np.array([psi]), model.h, model.mic_body)[0]      # (M,3)
    tau = np.linalg.norm(q[None] - model.anchors[:, None], axis=-1) / model.c           # (I,M)
    return (tau - tau.mean(1, keepdims=True)) * model.fs


def cipher_frame(model, fe, y, start, passes=2, prior=None, **kw):
    """Full CIPHER on one frame: pass 1 with a common gate, then re-gate each mic on its
    predicted pulse (pose-steered gating) and refine from the previous solution."""
    obs = fe.process(y, start)
    est = cipher(model, obs, prior=prior, **kw)
    for _ in range(passes - 1):
        obs = fe.process(y, start, gate_shift=gate_shifts(model, est.p, est.psi))
        nxt = cipher(model, obs, prior=(est.p, est.psi), **kw)
        nxt.ratio, nxt.p_tdoa = est.ratio, est.p_tdoa      # ambiguity decided in pass 1
        est = nxt
    return est, obs


def track(model, fe, y, starts, lag, sigma_pos=3e-3, sigma_psi=3e-4, window=5, vsig_floor=0.05,
          wsig_floor=0.2, sync_when_moving=True, motion_prior='inframe', aid=None, v_max=5.0,
          jump_max=0.5, coh_min=0.88, eta0_sigma=100e-6):
    """Frame-by-frame CIPHER with constant-velocity / constant-yaw-rate prediction.
    Yields (Estimate, t_ref_local) frame by frame, t_ref_local = estimate epoch in receiver time
    (s); only (p, psi, t) of past frames are kept, so long runs do not hold every frame's fit. Velocity and yaw-rate priors: linear regression of the last `window` estimates
    (stds from the assumed per-frame stds); before that, the in-frame estimates with loose priors.
    Checker waveform: the first two frames give a Doppler-robust gate and velocity (cold start
    while moving).
    Lost lock (implausible speed > v_max, a jump > jump_max from the prediction, or coherence <
    coh_min in two consecutive frames -- a neighbouring pose-domain lobe has coherence <= ~0.86,
    the true pose >= ~0.9 when all anchors are clear) -> the next frame is a cold start (heading
    search + TDOA gate) instead of trusting a diverged prior.
    aid(est, t_win) -> dict(p, psi, fixed): an outer stage (e.g. carrier.CarrierTracker) whose
    fixed solution replaces the CIPHER pose in the tracking history, so the next frame's prior
    comes from it (carrier-aided tracking); its output is stored in est.extra['aided']."""
    out = []                                    # (p, psi, t_loc) of past frames
    low_coh = 0
    vel, vsig, omega, wsig = np.zeros(2), 0.02, 0.0, 0.3
    etas = []                                   # receiver clock drift estimates (constant)
    gates = [None] * len(starts)
    if fe.sig.scheme == 'checker' and len(starts) >= 2:
        g = tdoa_solve_2frame(fe.process(y, starts[0]), fe.process(y, starts[1]), model, lag[0],
                              fe.sig.frame_len)
        if g is not None:
            gates[0], gates[1], vel, vsig = g[0], g[1], g[2], 0.1
    prev_obs = None
    for k, st in enumerate(starts):
        obs = fe.process(y, st)
        if not out and gates[k] is None and prev_obs is not None and fe.sig.scheme == 'checker':
            # mid-run cold start: Doppler-robust two-frame gate from the previous frame as well
            g = tdoa_solve_2frame(prev_obs, obs, model, lag[0], fe.sig.frame_len)
            if g is not None:
                gates[k], vel, vsig = g[1], g[2], 0.1
        prev_obs = obs
        prior = None
        if out:
            p1, psi1, t1 = out[-1]
            dtp = st / model.fs + t_ref1 - t1
            prior = (p1 + vel * dtp, psi1 + omega * dtp)
        eta = float(np.mean(etas)) if etas else 0.0
        # receiver drift: wide at first (the checker waveform makes it observable in one frame;
        # crystals can be far off), tightening as the constant is averaged over frames
        eta_sig = max(eta0_sigma / np.sqrt(len(etas) + 1), 2e-6)
        est = cipher(model, obs, prior=prior, vel=vel, omega=omega, lag=lag, vel_sigma=vsig,
                     omega_sigma=wsig, use_tdoa=sync_when_moving or prior is None,
                     gate=gates[k] if prior is None else None, eta=eta, eta_sigma=eta_sig)
        # robust coherence (worst anchor dropped): a blocked anchor does not trigger, a wrong
        # pose-domain lobe lowers every anchor
        cr = est.extra.get('coh_robust', est.coh)
        low_coh = low_coh + 1 if (np.isfinite(cr) and cr < coh_min) else 0
        lost = (not np.all(np.isfinite(est.p)) or np.linalg.norm(est.extra['vel']) > v_max
                or (prior is not None and np.linalg.norm(est.p - prior[0]) > jump_max)
                or low_coh >= 2)
        if lost:
            low_coh = 0
        est.extra['lost'] = bool(lost)
        etas.append(est.extra['eta'])
        t_loc = st / model.fs + est.extra['t_ref']
        t_ref1 = est.extra['t_ref']
        r = aid(est, st / model.fs) if aid is not None else None
        est.extra['aided'] = r
        healthy = r is not None and r['fixed'] and not r.get('coast', False)
        if healthy:                                 # carrier-aided: the fix seeds the next prior
            out.append((np.asarray(r['p'], float).copy(), float(r['psi']), t_loc))
        elif not lost:
            out.append((est.p.copy(), est.psi, t_loc))
        out = out[-window:]
        yield est, t_loc
        if lost and not healthy:
            out = []                                # cold start next frame, loose motion priors
            vel, vsig, omega, wsig = np.zeros(2), 0.5, 0.0, 0.5
            continue
        if len(out) == 1:
            # single-frame motion estimates: use them, but keep the priors loose
            vel, vsig = est.extra['vel'], max(vsig, 0.1)
            omega, wsig = est.extra['omega'], 0.5
        else:
            tt = np.array([o[2] for o in out])
            P = np.array([o[0] for o in out])
            H = np.unwrap(np.array([o[1] for o in out]))
            tc = tt - tt.mean()
            vel = (tc @ (P - P.mean(0))) / (tc @ tc)
            omega = float(tc @ (H - H.mean()) / (tc @ tc))
            vsig = max(sigma_pos / np.sqrt(tc @ tc), vsig_floor)
            wsig = max(sigma_psi / np.sqrt(tc @ tc), wsig_floor)
            if motion_prior == 'inframe':
                # centre on the latest in-frame (Doppler / aberration) estimates: they follow
                # accelerations that the window regression lags behind
                vel, omega = est.extra['vel'], est.extra['omega']


# ---------------------------------------------------------------- baselines
def drape_like(model, obs):
    """Per-mic group-delay TDOA positions + Procrustes (DRAPE-style rigid-body alignment)."""
    I, M = obs.n_arr_mic.shape
    pts = np.zeros((M, 2))
    for m in range(M):
        pts[m], _, _ = tdoa_solve(obs.n_arr_mic[:, m] / model.fs, model, obs.valid)
    B = model.mic_body[:, :2]
    Pc, Bc = pts - pts.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Bc.T @ Pc)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = Vt.T @ U.T
    psi = np.arctan2(Rm[1, 0], Rm[0, 0])
    p = pts.mean(0) - Rm @ B.mean(0)
    return Estimate(p, float(psi))


def bearings_per_anchor(model, obs, p_ref, step_deg=0.1):
    """Independent per-anchor wideband SRP bearing (body frame), elevation from p_ref."""
    az = np.deg2rad(np.arange(0, 360, step_deg))
    I, M, K = obs.Y.shape
    th = np.zeros(I)
    for i in range(I):
        d = model.anchors[i] - np.array([p_ref[0], p_ref[1], model.h])
        el = np.arctan2(d[2], np.hypot(d[0], d[1]))
        u = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full_like(az, np.sin(el))], 1)
        tau = -(u @ model.mic_body.T) / model.c                                # (A,M) far field
        E = np.exp(2j * np.pi * tau[..., None] * obs.freqs[i][None, None])
        s = np.einsum('amk,mk->ak', E, obs.Y[i])
        th[i] = az[np.argmax(np.sum(np.abs(s) ** 2, -1))]
    return th


def aoa_closed_form(model, th):
    """SCALA eq. (22)-(30): homogeneous LS for (x, y, psi) from body-frame bearings (2D)."""
    a, b = model.anchors[:, 0], model.anchors[:, 1]
    c, s = np.cos(th), np.sin(th)
    Mx = np.stack([c, s, -(a * c + b * s), -(a * s - b * c)], 1)
    u = np.linalg.svd(Mx)[2][-1]
    u = u / np.hypot(u[2], u[3])
    x = u[0] * u[2] + u[1] * u[3]
    y = u[1] * u[2] - u[0] * u[3]
    return Estimate(np.array([x, y]), float(np.arctan2(u[2], u[3])))


def aoa_tdoa_nls(model, obs, th, init):
    """Hybrid AOA + TDOA nonlinear LS (the 'AOA+TDOA (NLS)' baseline of SCALA), bearings fixed."""
    t = obs.n_arr / model.fs
    sig_th = np.deg2rad(1.0)

    def res(z):
        x, y, psi, t0 = z
        d3 = model.anchors - np.array([x, y, model.h])
        beta = np.arctan2(d3[:, 1], d3[:, 0])
        r1 = wrap(beta - psi - th) / sig_th
        r2 = (t - t0 - np.linalg.norm(d3, axis=1) / model.c) / model.sigma_toa
        return np.concatenate([r1, r2])

    d = np.linalg.norm(model.anchors - np.array([init.p[0], init.p[1], model.h]), axis=1)
    z0 = np.array([init.p[0], init.p[1], init.psi, np.mean(t - d / model.c)])
    z = least_squares(res, z0, method='lm').x
    return Estimate(z[:2], float(wrap(z[2])))
