"""Lag-domain parametric model (CIPHER refinement stage).

For anchor i and mic m, the compressed (matched-filtered, analytic) signal in a window around the
first path is modelled as

    r_m(t) = sum_l c_l * p(t - t0 - dtau_lm(pose)),

  p        known compressed pulse (reference chirp x band taper x nominal mic response),
  t0       one real delay per anchor   -> absorbs receiver clock offset/SFO, anchor sync error,
  c_l      one complex gain per path   -> absorbs amplitudes, speaker gain, reflection coefficients,
  dtau_lm  geometric delay of path l (direct + known-room images) at mic m, relative to the
           direct-path array mean.  Only these carry pose information.

Everything is done at baseband (carrier f_c removed) so p's envelope can be interpolated on a fine
grid while the carrier phase exp(-j 2 pi f_c dtau) is exact.  Gains c are concentrated out by
(ridge-regularised) least squares; t0 by a 1-D search.
"""
import numpy as np

from .channel import image_sources, resonance_resp
from .config import tilt_matrix
from .geometry import mic_world


class Pulse:
    """Baseband compressed pulse p_bb(t) of one anchor on a fine grid (t=0 <-> chirp start)."""

    def __init__(self, P_half, f_axis, fc, fs, L, up=16, t_span=(-0.02, 0.03)):
        # P_half: complex spectrum on the L-point FFT grid (positive band only, zeros elsewhere)
        Lu = L * up
        Pu = np.zeros(Lu, complex)
        Pu[:L // 2] = P_half[:L // 2]
        p = np.fft.ifft(Pu) * up                       # analytic pulse, sample spacing 1/(fs*up)
        t = np.arange(Lu) / (fs * up)
        t[t >= L / fs / 2] -= L / fs                  # centre the circular time axis
        o = np.argsort(t)
        t, pb = t[o], (p * np.exp(-2j * np.pi * fc * t))[o]
        # keep only the span the lag model can query (lag window + path delays + Doppler shifts):
        # the full circular axis is ~100x larger and long runs cache many Doppler-scaled pulses
        keep = (t >= t_span[0]) & (t <= t_span[1])
        self.t, self.pb = t[keep], pb[keep]
        self.dt = 1 / (fs * up)
        self.fc = fc
        self.f_lo = None
        # phase of the template at its envelope peak: c * exp(j peak_phase) is the data's carrier
        # phase at the peak, which tracks the chirp-centre delay free of range-Doppler coupling
        self.peak_phase = float(np.angle(self.pb[np.argmax(np.abs(self.pb))]))
        self.t_centre = None            # chirp-centre time in the template (set by PulseBank)

    def centre_correction(self):
        """Carrier read at the peak, -arg(c e^{j peak_phase}) / (2 pi fc), plus this = chirp-centre
        arrival exactly for the template's own Doppler (removes the beta^2 sweep-direction term;
        insensitive to the template velocity error to first order)."""
        return self.t_centre + self.peak_phase / (2 * np.pi * self.fc)

    def __call__(self, t):
        x = (np.asarray(t) - self.t[0]) / self.dt
        k = np.clip(np.floor(x).astype(int), 0, len(self.t) - 2)
        a = x - k
        out = self.pb[k] * (1 - a) + self.pb[k + 1] * a
        return np.where((x >= 0) & (x <= len(self.t) - 1), out, 0.0)


class PulseBank(list):
    """Compressed-pulse models per anchor (list = static up-chirp pulses) plus exact pulses for a
    Doppler time scale beta = v_r / c and either sweep direction.

    A receiver approaching at v_r sees s((1+beta) t): the chirp is shifted AND its rate scaled;
    the compressed pulse moves by about -beta T (f_start/B + 1) for an up-chirp (opposite sign for
    a down-chirp) and is slightly distorted (-24 dB at 0.8 m/s). scaled(i, beta, d) returns the
    exact compressed pulse; kappa(i, d) = d(peak position)/d(beta), computed numerically."""

    def __init__(self, fe, scn):
        super().__init__()
        self.fe, self.sig = fe, scn.signal
        H = np.ones(fe.L, complex)
        pos = fe.f > 0
        H[pos] = resonance_resp(scn.mic.f_res, scn.mic.q_res, fe.fs, fe.f[pos])
        self.H = H
        for i in range(fe.I):
            self.append(self._make(i, np.conj(fe.Sref[i]), 1))
            self[i].t_centre = scn.signal.chirp_len / 2
        self._cache, self._kappa = {}, {}
        self.max_cache = 256

    def _make(self, i, S, direction):
        f0, f1 = self.sig.bands[i]
        ref = self.fe.Sref[i] if direction > 0 else self.fe.Sref_dn[i]
        pu = Pulse(S * ref * self.fe.wband[i] * self.H, self.fe.f, 0.5 * (f0 + f1), self.fe.fs, self.fe.L)
        pu.f_lo = f0
        return pu

    def scaled(self, i, beta, direction=1):
        beta = float(np.clip(beta, -0.02, 0.02))               # |v| <= ~7 m/s: guards a diverged track
        if abs(beta) < 1e-6 and direction > 0:
            return self[i]
        key = (i, round(beta, 7), direction)
        if key in self._cache:
            self._cache[key] = self._cache.pop(key)            # LRU: most recent last
        else:
            if len(self._cache) >= self.max_cache:
                self._cache.pop(next(iter(self._cache)))
            from .signals import chirp_at
            f0, f1 = self.sig.bands[i]
            T = self.sig.chirp_len
            n = np.arange(int(np.ceil(T / (1 + beta) * self.fe.fs)) + 2)
            x = chirp_at((1 + beta) * n / self.fe.fs, f0, f1 - f0, T, self.sig.taper, direction)
            self._cache[key] = self._make(i, np.fft.fft(x, self.fe.L), direction)
            self._cache[key].t_centre = T / (2 * (1 + beta))
        return self._cache[key]

    @staticmethod
    def peak(pu):
        a = np.abs(pu.pb)
        k = int(np.argmax(a))
        d = a[k - 1] - 2 * a[k] + a[k + 1]
        return pu.t[k] + (0.5 * (a[k - 1] - a[k + 1]) / d * pu.dt if d < 0 else 0.0)

    def kappa(self, i, direction=1, db=1e-3):
        """d(pulse peak)/d(beta) (s per unit beta) -- the range-Doppler coupling coefficient."""
        key = (i, direction)
        if key not in self._kappa:
            self._kappa[key] = (self.peak(self.scaled(i, db, direction)) -
                                self.peak(self.scaled(i, -db, direction))) / (2 * db)
        return self._kappa[key]


def build_pulses(fe, scn):
    """Pulse models matching FrontEnd.compress (same reference, taper, nominal mic response)."""
    return PulseBank(fe, scn)


def image_set(anchor, room_size, ref_pos, h, c, max_extra, max_order=2, n_max=10):
    """Direct source + images whose extra path length (at ref_pos) is below max_extra (m).
    Ordered by extra path length; direct first."""
    img, _, _ = image_sources(anchor, room_size, max_order)
    q = np.array([ref_pos[0], ref_pos[1], h])
    d = np.linalg.norm(img - q, axis=1)
    d0 = np.linalg.norm(anchor - q)
    extra = d - d0
    keep = np.where((extra > 1e-6) & (extra < max_extra))[0]
    keep = keep[np.argsort(extra[keep])][:n_max - 1]
    return np.vstack([anchor[None], img[keep]])


class LagFit:
    """Variable-projection residuals for (x, y, psi, t0_1..t0_I[, t_rx]) at a reference epoch.

    Scaled parameter vector z = [x mm, y mm, psi mrad, t0_i us ..., (t_rx us)].

    Motion (vel (2,), omega) comes from tracking. With it, (i) each anchor's geometry is evaluated
    at that anchor's own chirp-centre epoch (arrivals differ by up to ~40 ms), and (ii) the
    range-Doppler coupling of the up-chirp is modelled per mic and path: the carrier phase follows
    the chirp-centre delay exactly, while the envelope is advanced by (v_r T / c)(f_c/B - 1/2)."""

    def __init__(self, model, obs, pulses, room_size, ref_p, ridge=1e-3, max_extra=1.6,
                 multipath=True, use_tdoa=True, vel=(0.0, 0.0), omega=0.0, chirp_len=None,
                 estimate_vel=False, vel_sigma=0.3, estimate_omega=False, omega_sigma=1.0,
                 estimate_eta=False, eta=0.0, eta_sigma=100e-6, estimate_c=False, c_sigma=5e-3,
                 n_paths=10):
        self.model, self.obs, self.pulses = model, obs, pulses
        self.c_direct = np.full(len(model.anchors), np.nan + 0j)
        self.c_corr = np.zeros(len(model.anchors))
        self.ridge, self.use_tdoa = ridge, use_tdoa
        self.idx = np.where(obs.valid)[0]
        self.src = [image_set(model.anchors[i], room_size, ref_p, model.h, model.c, max_extra,
                              n_max=n_paths)
                    if multipath else model.anchors[i][None] for i in range(len(model.anchors))]
        if getattr(model, 'body_plate', None) is not None:     # reflector fixed to the robot
            zp = model.h - model.body_plate
            self.src = [np.vstack([s_, [a[0], a[1], 2 * zp - a[2]]]) for s_, a in zip(self.src, model.anchors)]
        self.scale = obs.lag_scale.copy()
        self.vel, self.omega = np.asarray(vel, float), float(omega)
        # velocity as a parameter: observable within one frame through the range-Doppler
        # coupling of the envelope (t0_i) together with the sync constraint
        self.estimate_vel = estimate_vel and use_tdoa
        self.vel_prior, self.vel_sigma = self.vel.copy(), vel_sigma
        # yaw rate as a parameter: anchors are observed at epochs up to +-20 ms apart, so the
        # heading drift between them makes omega observable within one frame
        self.estimate_omega = estimate_omega
        self.omega_prior, self.omega_sigma = float(omega), omega_sigma
        # receiver SFO eta acts on every received chirp like a common Doppler time scale. With
        # up-chirps only it is absorbed by t_rx; with mixed sweep directions it is not -- and it
        # becomes observable (the checker waveform measures the receiver clock drift).
        self.estimate_eta = estimate_eta
        self.eta_prior, self.eta_sigma = float(eta), eta_sigma
        # relative speed-of-sound error s (c = c_model (1 + s)): the sync constraint compares
        # ranges of up to ~13 m at us precision, so 0.3 % (0.5 degC) would bias the position
        self.estimate_c = estimate_c and use_tdoa
        self.c_sigma = c_sigma
        self.T = model.chirp_len if chirp_len is None else chirp_len
        # Doppler time scaling at the prior velocity, per anchor (direct path, array centre)
        q0 = np.array([ref_p[0], ref_p[1], model.h])
        self.u = (model.anchors - q0) / np.linalg.norm(model.anchors - q0, axis=1, keepdims=True)
        self.v_away0 = -(self.u[:, :2] @ self.vel)                # (I,) prior receding speed
        I = len(model.anchors)
        self.dirs = np.ones(I) if getattr(obs, 'dirs', None) is None else obs.dirs
        self.pu = [pulses.scaled(i, -self.v_away0[i] / model.c + self.eta_prior, self.dirs[i])
                   for i in range(I)]
        # coupling coefficient d(peak)/d(beta) per anchor for its sweep direction
        self.kap = np.array([pulses.kappa(i, self.dirs[i]) for i in range(I)])
        # per path (direct + images): prior receding speed and its own Doppler-scaled pulse
        self.v_away0_path, self.pu_path = [], []
        for i in range(len(model.anchors)):
            dirs = q0 - self.src[i]
            va = (dirs[:, :2] @ self.vel) / np.linalg.norm(dirs, axis=1)
            va[0] = self.v_away0[i]
            self.v_away0_path.append(va)
            self.pu_path.append([pulses.scaled(i, round(-x / model.c + self.eta_prior, 5), self.dirs[i])
                                 for x in va])
        # later paths (images) reach the array later: evaluate them where the array is then
        self.lag_path = [(np.linalg.norm(q0 - src, axis=1) - np.linalg.norm(q0 - src[0])) / model.c
                         for src in self.src]
        tc = self.t0_init() + self.T / 2                   # chirp-centre arrival per anchor
        self.t_ref = float(np.mean(tc[self.idx]))           # reference epoch (frame time base)
        self.dt = tc - self.t_ref

    def _motion(self, z):
        """(vel, omega, eta) from the parameter vector (None where fixed at the prior)."""
        k = len(z) - (1 if self.estimate_c else 0)
        omega = eta = None
        if self.estimate_eta:
            eta, k = z[k - 1] * 1e-6, k - 1
        if self.estimate_omega:
            omega, k = z[k - 1] * 1e-3, k - 1
        vel = z[k - 2:k] * 1e-3 if self.estimate_vel else None
        return vel, omega, eta

    def _geom(self, i, p, psi, vel=None, omega=None, eta=None, s=0.0):
        """Mic positions and per-mic/per-path envelope advance for anchor i at its epoch."""
        m = self.model
        vel = self.vel if vel is None else vel
        omega = self.omega if omega is None else omega
        eta = self.eta_prior if eta is None else eta
        pe, se = p + vel * self.dt[i], psi + omega * self.dt[i]
        q = mic_world(pe[None], np.array([se]), m.h, m.mic_body)[0]                   # (M,3)
        shift = np.concatenate([vel, [0.0]])[None, None] * self.lag_path[i][:, None, None]
        diff = q[None] + shift - self.src[i][:, None]                                  # (L,M,3)
        dist = np.linalg.norm(diff, axis=-1)
        env, v_bar, beta = 0.0, 0.0, np.zeros((len(dist), 1))
        if omega != 0.0 or np.any(vel != 0.0) or self.v_away0[i] != 0.0 or eta != self.eta_prior:
            rb = q[:, :2] - np.array([pe[0], pe[1]])
            vm = vel[None] + omega * np.stack([-rb[:, 1], rb[:, 0]], 1)          # (M,2)
            v_away = np.einsum('lmk,mk->lm', diff[..., :2], vm) / dist                 # (L,M)
            # envelope advance relative to the scaled pulse (which already holds the prior Doppler
            # of each path): kappa * delta(v_away) / c, kappa = d(peak)/d(beta) of this waveform
            # (about -T (f_c/B + 1/2) for an up-chirp, opposite for a down-chirp)
            env = self.kap[i] * ((v_away - self.v_away0_path[i][:, None]) / m.c - (eta - self.eta_prior))
            v_bar = v_away[0].mean()
            beta = -v_away.mean(1, keepdims=True) / m.c                 # approach speed / c per path
        return dist / (m.c * (1 + s)), env, v_bar, beta

    def _res_i(self, i, p, psi, t0, vel=None, omega=None, eta=None, s=0.0):
        """Whitened projection residual of anchor i (complex, M*W) and its mean direct delay."""
        tau, env, v_bar, beta = self._geom(i, p, psi, vel, omega, eta, s)           # (L,M)
        # a moving array samples the wavefront at different instants: measured inter-mic delay
        # differences are the same-instant geometric ones divided by (1 + beta)
        dtau = tau[:, :1] - tau[0].mean() + (tau - tau[:, :1]) / (1 + beta)
        pu = self.pu[i]
        spread = tau.mean(1, keepdims=True) / tau                               # near-field 1/d
        if self.model.mic_delay is not None:
            env = env - self.model.mic_delay[i][None]                          # mic group delay
        arg = self.obs.lag_t[i][None, None] - t0 - (dtau - env)[..., None]           # (L,M,W)
        S = np.stack([self.pu_path[i][l](arg[l]) for l in range(len(arg))]) \
            * (spread * np.exp(-2j * np.pi * pu.fc * dtau))[..., None]
        S = S.reshape(len(dtau), -1).T                                          # (M*W, L)
        r = self.obs.lag_r[i]
        if self.model.mic_gain is not None:
            r = r / self.model.mic_gain[i][:, None]
        y = r.reshape(-1)
        G = S.conj().T @ S
        reg = np.full(len(G), self.ridge * np.real(np.trace(G)) / len(G))
        reg[0] = 1e-9 * reg[1] if len(reg) > 1 else 0.0                       # keeps G regular when a
        try:                                                                    # diverged state pushes a
            cvec = np.linalg.solve(G + np.diag(reg), S.conj().T @ y)            # path out of the window
        except np.linalg.LinAlgError:
            cvec = np.linalg.lstsq(S, y, rcond=None)[0]
        self.c_direct[i] = cvec[0] * np.exp(1j * self.pu_path[i][0].peak_phase)   # carrier at the peak
        self.c_corr[i] = self.pu_path[i][0].centre_correction()
        # chirp-start arrival (what t0 measures) = centre-epoch delay - v_away T / (2c)
        return (y - S @ cvec) * np.sqrt(self.scale[i]), tau[0].mean() - v_bar * self.T / (2 * self.model.c)

    def reweight(self, z):
        """Re-estimate each anchor's residual variance at z (noise + unmodelled mismatch); use it
        when it exceeds the noise floor. One IRLS step of the likelihood with unknown per-anchor
        variances -- stops a high-SNR anchor with residual mismatch from dominating."""
        p, psi = z[:2] * 1e-3, z[2] * 1e-3
        vel, omega, eta = self._motion(z)
        for k, i in enumerate(self.idx):
            res, _ = self._res_i(i, p, psi, z[3 + k] * 1e-6, vel, omega, eta, self.c_scale(z))
            var_hat = np.vdot(res, res).real / self.scale[i] / res.size      # per lag sample
            self.scale[i] = min(self.obs.lag_scale[i], self.obs.lag_enbw[i] / var_hat)

    def t0_init(self):
        """t0 means chirp-start arrival; the detector returns the pulse peak. They differ by the
        peak offset of the (Doppler-scaled) compressed pulse (about -1 ms at 0.8 m/s)."""
        return np.array([self.obs.lag_t0[i] - self.pu[i].t[np.argmax(np.abs(self.pu[i].pb))]
                         for i in range(len(self.pu))])

    def z0(self, p, psi):
        t0 = self.t0_init()[self.idx] * 1e6
        z = [p[0] * 1e3, p[1] * 1e3, psi * 1e3, *t0]
        if self.use_tdoa:
            q = np.array([p[0], p[1], self.model.h])
            d = np.linalg.norm(self.model.anchors[self.idx] - q, axis=1)
            z.append(np.mean(self.t0_init()[self.idx] - d / self.model.c) * 1e6)
        if self.estimate_vel:
            z += list(self.vel_prior * 1e3)
        if self.estimate_omega:
            z.append(self.omega_prior * 1e3)
        if self.estimate_eta:
            z.append(self.eta_prior * 1e6)
        if self.estimate_c:
            z.append(0.0)
        return np.array(z)

    def c_scale(self, z):
        """Relative speed-of-sound error s from the parameter vector (0 if not estimated)."""
        return z[-1] * 1e-3 if self.estimate_c else 0.0

    def residuals(self, z):
        m = self.model
        p, psi = z[:2] * 1e-3, z[2] * 1e-3
        vel, omega, eta = self._motion(z)
        out, tbar = [], []
        for k, i in enumerate(self.idx):
            res, tb = self._res_i(i, p, psi, z[3 + k] * 1e-6, vel, omega, eta, self.c_scale(z))
            out += [res.real, res.imag]
            tbar.append(tb)
        if self.use_tdoa:
            # anchors are synchronised: the fitted (multipath-corrected) direct-path delays obey
            # t0_i = t_rx + tau_i(epoch_i) + eps_i, eps_i ~ N(0, sigma_sync^2).  Uses t0 from the
            # fit, not the first-path detector, so floor/ceiling reflections cannot bias it.
            t_rx = z[3 + len(self.idx)] * 1e-6
            t0 = z[3:3 + len(self.idx)] * 1e-6
            e_ = self.eta_prior if eta is None else eta
            out.append((t0 - t_rx - (1 + e_) * np.array(tbar)) / m.sigma_toa)
        if self.estimate_vel:
            out.append((vel - self.vel_prior) / self.vel_sigma)
        if self.estimate_omega:
            out.append(np.array([(omega - self.omega_prior) / self.omega_sigma]))
        if self.estimate_eta:
            out.append(np.array([(eta - self.eta_prior) / self.eta_sigma]))
        if self.estimate_c:
            out.append(np.array([self.c_scale(z) / self.c_sigma]))
        return np.concatenate(out)


def _cal_columns(lf, i, p, psi, t0):
    """Model columns (L, M, W) of anchor i for a known pose (used by calibration)."""
    tau, env, _, beta = lf._geom(i, p, psi)
    dtau = tau[:, :1] - tau[0].mean() + (tau - tau[:, :1]) / (1 + beta)
    if lf.model.mic_delay is not None:
        env = env - lf.model.mic_delay[i][None]
    pu = lf.pu[i]
    spread = tau.mean(1, keepdims=True) / tau
    return pu(lf.obs.lag_t[i][None, None] - t0 - (dtau - env)[..., None])         * (spread * np.exp(-2j * np.pi * pu.fc * dtau))[..., None]


def _fit_gamma(cols, data, gamma, n_iter=4):
    """Alternating LS: per-mic complex gains (shared over frames) and per-frame path gains."""
    M = data[0].shape[0]
    for _ in range(n_iter):
        num, den = np.zeros(M, complex), np.zeros(M)
        for S, r in zip(cols, data):
            A = np.transpose(S * gamma[None, :, None], (1, 2, 0)).reshape(-1, S.shape[0])
            c = np.linalg.lstsq(A, r.reshape(-1), rcond=None)[0]
            u = np.einsum('l,lmw->mw', c, S)
            num += np.sum(np.conj(u) * r, 1)
            den += np.sum(np.abs(u) ** 2, 1)
        gamma = num / den
        gamma = gamma / np.mean(gamma)
    res = []
    for S, r in zip(cols, data):
        A = np.transpose(S * gamma[None, :, None], (1, 2, 0)).reshape(-1, S.shape[0])
        c = np.linalg.lstsq(A, r.reshape(-1), rcond=None)[0]
        res.append(r.reshape(-1) - A @ c)
    return gamma, np.concatenate(res)


def plate_depth_scan(model, frames, pulses, room_size, depths=(None,) + tuple(np.arange(0.02, 0.405, 0.02)),
                     min_gain_db=1.0):
    """Depth of a reflector fixed to the robot below the array (AGV deck) from calibration frames
    at known poses: for each candidate depth (None = no reflector) the lag model with that extra
    image path is fitted per frame/anchor (t0 by 1-D search, path gains free) and the whitened
    residual energy summed; a reflector is accepted only if it lowers the cost by min_gain_db
    (one extra free path always fits a little). Returns (best depth or None, {depth: cost})."""
    import copy as _copy
    from scipy.optimize import minimize_scalar
    costs = {}
    for d in depths:
        mdl = _copy.copy(model)
        mdl.body_plate = d
        tot = 0.0
        for obs, p, psi in frames:
            lf = LagFit(mdl, obs, pulses, room_size, p, use_tdoa=False)
            for i in lf.idx:
                f = lambda t0: np.vdot(*(2 * [lf._res_i(i, p, psi, t0)[0]])).real
                grid = obs.lag_t0[i] + np.arange(-5e-4, 5e-4 + 1e-9, 2e-5)
                g0 = grid[np.argmin([f(t) for t in grid])]
                tot += minimize_scalar(f, bounds=(g0 - 2e-5, g0 + 2e-5), method='bounded',
                                       options=dict(xatol=1e-9)).fun
        costs[d] = tot
    best = min(costs, key=costs.get)
    if best is not None and 10 * np.log10(costs[None] / costs[best]) < min_gain_db:
        best = None
    return best, costs


def calibrate(model, frames, pulses, room_size, multipath=True, geometry=True, max_nfev=30, tilt=True):
    """Array calibration from frames with known poses (e.g. motion-capture truth).

    Estimates the mounting tilt of the array plane (roll, pitch; tilt=True), per microphone an
    in-plane port-position offset (geometry=True) and, per (band, mic), a complex gain and a group
    delay. Path gains / t0 per frame are nuisance. The tilt matters: 0.25 deg puts the opposite
    mics +-0.2 mm apart vertically, ~14 mm of CIPHER position error (and a spirit level is no better).
    Gauges: mean group delay per band is 0 (absorbed by t0); the array frame is the mocap frame.
    Returns dict(mic_body, mic_gain (I,M), mic_delay (I,M), tilt_deg); apply by setting the Model fields."""
    import copy as _copy
    from scipy.optimize import least_squares, minimize_scalar
    I, M = len(model.anchors), len(model.mic_body)
    body0 = model.mic_body.copy()
    mdl = _copy.copy(model)
    mdl.mic_gain, mdl.mic_delay = None, np.zeros((I, M))

    # stage 1: t0 per frame/anchor with the nominal model
    t0s = {}
    lfs = []
    for f_idx, (obs, p, psi) in enumerate(frames):
        lf = LagFit(mdl, obs, pulses, room_size, p, multipath=multipath, use_tdoa=False)
        lfs.append(lf)
        for i in range(I):
            if not obs.valid[i]:
                continue
            f = lambda t0: np.vdot(*(2 * [lf._res_i(i, p, psi, t0)[0]])).real
            grid = obs.lag_t0[i] + np.arange(-5e-4, 5e-4 + 1e-9, 1e-5)
            g0 = grid[np.argmin([f(t) for t in grid])]
            t0s[f_idx, i] = minimize_scalar(f, bounds=(g0 - 1e-5, g0 + 1e-5), method='bounded',
                                            options=dict(xatol=1e-9)).x
    gammas = np.ones((I, M), complex)

    nt = 2 if tilt else 0

    def unpack(th):
        k = nt
        body = body0 @ tilt_matrix(*(th[:2] * 0.1)).T if tilt else body0.copy()   # 0.1 deg units
        if geometry:
            d = th[k:k + 2 * M].reshape(M, 2) * 1e-3                  # mm
            # gauge: no common translation / rotation (array centre and heading zero belong to
            # the mocap-to-array extrinsics, calibrated separately)
            d = d - d.mean(0)
            ang = np.arctan2(body0[:, 1], body0[:, 0])
            tang = np.stack([-np.sin(ang), np.cos(ang)], 1)
            d = d - np.mean(np.sum(d * tang, 1)) * tang
            body[:, :2] += d
            k += 2 * M
        dl = th[k:].reshape(I, M) * 1e-6                             # us
        dl = dl - dl.mean(1, keepdims=True)
        return body, dl

    def resid(th):
        body, dl = unpack(th)
        mdl.mic_body, mdl.mic_delay = body, dl
        out = []
        for i in range(I):
            cols, data = [], []
            for f_idx, (obs, p, psi) in enumerate(frames):
                if (f_idx, i) not in t0s:
                    continue
                cols.append(_cal_columns(lfs[f_idx], i, p, psi, t0s[f_idx, i]))
                data.append(obs.lag_r[i])
            gammas[i], r = _fit_gamma(cols, data, gammas[i])
            w = np.sqrt(np.mean([frames[f][0].lag_scale[i] for f in range(len(frames))]))
            out += [r.real * w, r.imag * w]
        return np.concatenate(out)

    th0 = np.zeros(nt + (2 * M if geometry else 0) + I * M)
    sol = least_squares(resid, th0, method='trf', x_scale=1.0, max_nfev=max_nfev, diff_step=1e-3)
    body, dl = unpack(sol.x)
    resid(sol.x)
    return dict(mic_body=body, mic_gain=gammas.copy(), mic_delay=dl,
                tilt_deg=tuple(sol.x[:2] * 0.1) if tilt else (0.0, 0.0))
