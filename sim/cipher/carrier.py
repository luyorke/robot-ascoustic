"""CIPHER-X stage 2: cross-anchor carrier phase (docs/review.md).

The anchors are synchronised to microseconds and the acoustic carrier period is 50 us, so each
anchor's direct-path carrier phase measures its arrival time to a fraction of a microsecond --
modulo one carrier period. With the CIPHER pose as the float solution the integer cycles are
fixed, and the arrivals obey the sync model

    t_i = tau_phi_i + b_i + n_i / f_i = t_rx + (1 + eta)(T/2 + tau_c,i(p; c)) + eps_i,   eps_i ~ N(0, s_s^2)

tau_phi_i = -arg(c_i0 e^{j phi_pk}) / (2 pi f_i)  carrier phase at the pulse peak = chirp-centre
                                                  arrival (no range-Doppler coupling, theory sec. 3)
b_i                                    per-anchor phase constant (speaker / mic phase), calibrated
tau_c,i(p; c)                          propagation delay at the anchor's chirp-centre epoch
c                                      speed of sound, estimated (scale s: c = c0 (1 + s))
"""
import itertools

import numpy as np


def carrier_times(est, fc):
    """Chirp-centre arrival (s, modulo 1/fc) of each anchor's direct path: the carrier phase of
    the data at the compressed-pulse peak (CIPHER lag model), free of range-Doppler coupling."""
    return -np.angle(est.extra['c_direct']) / (2 * np.pi * fc) + est.extra['c_corr']


def predicted_arrivals(est, p):
    """Propagation part (1 + eta) tau_c of the chirp-centre arrival at array position p (other
    motion terms from the CIPHER fit; model speed of sound). Arrival = trx_centre + this."""
    lf, z = est.extra['lagfit'], est.extra['z']
    vel, omega, eta = lf._motion(z)
    e_ = lf.eta_prior if eta is None else eta
    out = np.full(len(lf.model.anchors), np.nan)
    for i in lf.idx:
        tau, _, _, _ = lf._geom(i, np.asarray(p, float), z[2] * 1e-3, vel, omega, eta)
        out[i] = (1 - e_) * tau[0].mean()          # receiver time runs at (1 - eta_CIPHER)
    return out


def trx_centre(est):
    """Envelope-level receive offset of chirp centres: CIPHER's t_rx (chirp starts) + T/2 in receiver time."""
    lf, z = est.extra['lagfit'], est.extra['z']
    _, _, eta = lf._motion(z)
    e_ = lf.eta_prior if eta is None else eta
    return z[3 + len(lf.idx)] * 1e-6 + (1 - e_) * lf.T / 2


def calibrate_bias(ests, poses, fc, n_iter=4, s_span=5e-4, s_step=2e-6):
    """Per-anchor carrier constants b_i (s) and the speed-of-sound scale s during calibration,
    from CIPHER estimates at known array positions (motion capture).

    Frame k, anchor i: w_ik = trx_c + g_ik (1 - s) - tau_phi_i = b_i - Delta_k (mod 1/f_i), with
    Delta_k CIPHER's envelope-level t_rx error (can exceed half a period). s is common to all
    frames (one session) and separable from b_i because g_ik changes with the pose; it must not
    come from CIPHER's single-frame estimates (0.02 % x 25 ms = 5 us). For each s on a grid:
    alternate Delta_k (circular grid search per frame) and b_i (circular mean per anchor); keep
    the most coherent s. Returns (b (I,), s), b_i principal values modulo 1/f_i."""
    fc = np.asarray(fc, float)
    G = np.array([predicted_arrivals(est, p) for est, p in zip(ests, poses)])         # (F, I)
    T0 = np.array([trx_centre(est) - carrier_times(est, fc) for est in ests])          # (F, I)
    ok = np.isfinite(G) & np.isfinite(T0)
    grid = np.arange(-80e-6, 80e-6 + 1e-9, 0.25e-6)

    def fit(s_):
        W = T0 + G * (1 - s_)
        b, Dk = np.zeros(len(fc)), np.zeros(len(W))
        for _ in range(n_iter):
            for k in range(len(W)):
                m = ok[k]
                cost = -np.sum(np.cos(2 * np.pi * fc[m] * (W[k, m][None] + grid[:, None] - b[m])), 1)
                Dk[k] = grid[np.argmin(cost)]
            ph = np.where(ok, np.exp(2j * np.pi * fc * (W + Dk[:, None])), 0)
            b = np.angle(ph.sum(0)) / (2 * np.pi * fc)
        coh = np.sum(np.abs(ph.sum(0))) / ok.sum()
        return coh, b

    s_mid = float(np.median([est.extra.get('c_scale', 0.0) for est in ests]))   # CIPHER: ~0.01 %
    s_grid = s_mid + np.arange(-s_span, s_span + 1e-12, 10 * s_step)
    cs = [fit(x)[0] for x in s_grid]
    s0 = s_grid[int(np.argmax(cs))]
    s_fine = np.arange(s0 - 10 * s_step, s0 + 10 * s_step + 1e-12, s_step)
    res = [fit(x) for x in s_fine]
    j = int(np.argmax([r[0] for r in res]))
    return res[j][1], float(s_fine[j])


def _frame_terms(est, fc, bias, t_win, h=1e-4):
    """Per-frame quantities of the linearised carrier model at the CIPHER position p_A, in
    absolute receiver time (t_win = window start, s):
    t_i = t_win + tau_phi_i + b_i + n_i / f_i = trx + (g0_i + G_i (p - p_A)) (1 - s) + eps_i."""
    pA = est.p.copy()
    g0 = predicted_arrivals(est, pA)
    G = np.stack([(predicted_arrivals(est, pA + d) - g0) / h
                  for d in (np.array([h, 0.0]), np.array([0.0, h]))], 1)
    C = est.extra['cov_pose']
    if not np.all(np.isfinite(C)) or np.any(np.linalg.eigvalsh(C[:2, :2]) <= 0):
        C = np.diag([5e-3 ** 2, 5e-3 ** 2, np.deg2rad(0.05) ** 2])
    return dict(idx=est.extra['lagfit'].idx, pA=pA, psiA=est.psi, g0=g0, G=G, C=C,
                trxA=t_win + trx_centre(est), tphi=t_win + carrier_times(est, fc) + bias,
                # CIPHER's eta is the receiver drift as an equivalent Doppler factor (a fast clock
                # stretches the signal like a receding source): the clock rate offset is -eta
                eta=-float(est.extra.get('eta', 0.0)), vel=np.asarray(est.extra.get('vel', np.zeros(2)), float),
                omega_est=float(est.extra.get('omega', 0.0)),
                p_best=pA.copy())


class CarrierTracker:
    """Multi-frame carrier stage (GNSS-style float -> fix -> hold).

    * Speed of sound: Kalman random walk on s, fed by CIPHER's per-frame estimate (the multipath
      model geometry makes it observable to ~0.02 % per frame). The integer search starts only
      once s is known to s_fix_std -- the decisive quantity: otherwise wrong integers trade
      against the speed of sound.
    * Between frames the carrier-time differences carry the integers (time-differenced carrier
      phase): dn_i = round(f_i (dtrx + dg_i - dtau_i)), dtrx by a circular search over anchors.
    * Acquisition: batch least squares over the last `window` frames, linearised at the CIPHER
      positions (priors: CIPHER covariance per frame, the s state). All integer candidates of the
      first frame share one projector, so the search is cheap; accepted by a difference test.
    * Hold: integers carried per anchor frame to frame; each frame solves (dp, trx, s) with the
      receive offset predicted through the known frame period and drift; an anchor whose residual
      exceeds 4 sigma (cycle slip, blocked direct path) is dropped and later re-fixed by rounding
      at the sub-mm solution; with fewer than 3 known anchors the hold is lost and re-acquired."""

    def __init__(self, fc, bias, frame_period, sigma_s=4e-6, s_std0=3e-3, s_walk=2e-6, window=6,
                 diff_min=12.0, s_fix_std=1.5e-4, n_search=1, n_common=3,
                 trx_sigma=1e-3, eta_sigma=20e-6, s_floor=1e-4, sigma_s_max=4e-6, max_coast=0, max_reject=2, bias_window=10,
                 bias_min=6e-3):
        self.fc, self.bias = np.asarray(fc, float), np.asarray(bias, float)
        self.s, self.s_var, self.s_walk = 0.0, s_std0 ** 2, s_walk ** 2
        self.sig_s, self.window, self.diff_min, self.s_fix_std = sigma_s, window, diff_min, s_fix_std
        self.n_search, self.n_common = n_search, n_common
        self.trx_sigma, self.Tf, self.eta_sigma = trx_sigma, frame_period, eta_sigma
        # CIPHER's sound speed has a systematic error of ~0.01 %: averaging frames cannot go below
        # it; below the floor only the fixed carrier solution can pin s down
        self.s_floor, self.sigma_s_max, self.max_coast = s_floor, sigma_s_max, max_coast
        self.max_reject, self.bias_window, self.bias_min = max_reject, bias_window, bias_min
        self.res2 = []                      # carrier residuals^2 (s^2) x dof factor, for sigma_s
        self.frames = []                    # acquisition window: _frame_terms dicts + 'off' (integers vs start)
        self.hold = None                    # per-anchor integers + receive offset / drift while fixed

    def _kalman(self, z, var):
        k = self.s_var / (self.s_var + var)
        self.s, self.s_var = self.s + k * (z - self.s), (1 - k) * self.s_var

    def _dtrx(self, a, b):
        """Receive-offset increment between frames: the anchors emit every T_f (system time),
        seen through the receiver clock drift eta (estimated by CIPHER) -- known to < 1 us."""
        m = np.round((b['trxA'] - a['trxA']) / self.Tf)
        return m * self.Tf * (1 + 0.5 * (a['eta'] + b['eta']))

    def _delta_n(self, a, b):
        """Integer change from frame a to frame b by time differencing (nan where not shared)."""
        f = self.fc
        # range change from the previous best position (sub-mm once fixed) moved by the Doppler
        # velocity, rather than from two independent CIPHER positions (~3-5 mm each)
        dtf = b['trxA'] - a['trxA']
        p_pred = a['p_best'] + 0.5 * (a['vel'] + b['vel']) * dtf
        ga = a['g0'] + a['G'] @ (a['p_best'] - a['pA'])
        gb = b['g0'] + b['G'] @ (p_pred - b['pA'])
        dg, dt = gb - ga, b['tphi'] - a['tphi']
        m = np.isfinite(dg) & np.isfinite(dt)
        grid = self._dtrx(a, b) + np.arange(-6e-6, 6e-6 + 1e-9, 0.1e-6)
        x = grid[:, None] + (dg - dt)[None, m]
        dtrx = grid[np.argmax(np.sum(np.cos(2 * np.pi * f[m] * x), 1))]
        dn = np.full(len(f), np.nan)
        dn[m] = np.round(f[m] * (dtrx + dg[m] - dt[m]))
        return dn

    def _design(self, frames):
        """Whitened linear system in (dp_k per frame, trx_0, eta, s). The receive offsets of the
        window's frames are tied by the anchors' fixed frame period seen through the receiver
        clock drift: trx_k = trx_0 + tau_k (1 + eta), tau_k = m_k T_f -- eta is a window parameter
        (CIPHER's per-frame eta is too noisy to chain frames at the us level).
        Returns A, b0 (data without the first-frame integers), D (maps them in), n_meas."""
        K, I = len(frames), len(self.fc)
        npar = 2 * K + 3
        it0, ieta, i_s = 2 * K, 2 * K + 1, 2 * K + 2
        rows, b0, D = [], [], []
        sig = self.sig_s

        def add(r, b, d=None):
            rows.append(r)
            b0.append(b)
            D.append(np.zeros(I) if d is None else d)

        for k, fr in enumerate(frames):
            tau = np.round((fr['trxA'] - frames[0]['trxA']) / self.Tf) * self.Tf
            for i in fr['idx']:
                r = np.zeros(npar)
                r[2 * k:2 * k + 2] = fr['G'][i] / sig
                r[it0], r[ieta], r[i_s] = 1 / sig, tau / sig, -fr['g0'][i] / sig
                d = np.zeros(I)
                d[i] = 1 / self.fc[i] / sig
                add(r, (fr['tphi'][i] + fr['off'][i] / self.fc[i] - tau - fr['g0'][i]) / sig, d)
        n_meas = len(rows)
        for k, fr in enumerate(frames):                        # CIPHER position priors
            Lw = np.linalg.cholesky(np.linalg.inv(fr['C'][:2, :2])).T
            for j in range(2):
                r = np.zeros(npar)
                r[2 * k:2 * k + 2] = Lw[j]
                add(r, 0.0)
        r = np.zeros(npar)                                     # weak envelope-level trx prior
        r[it0] = 1 / self.trx_sigma
        add(r, frames[0]['trxA'] / self.trx_sigma)
        r = np.zeros(npar)                                     # receiver drift prior (CIPHER)
        r[ieta] = 1 / self.eta_sigma
        add(r, np.mean([fr['eta'] for fr in frames]) / self.eta_sigma)
        r = np.zeros(npar)
        r[i_s] = 1 / np.sqrt(self.s_var)
        add(r, self.s / np.sqrt(self.s_var))
        return np.array(rows), np.array(b0), np.array(D), n_meas

    def _solve(self, n_first):
        A, b0, D, n_meas = self._design(self.frames)
        b = b0 + D @ n_first
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        res = b - A @ x
        self._last_res = res[:n_meas] * self.sig_s, max(n_meas - 2 * len(self.frames) - 3, 1)
        return x, float(res[:n_meas] @ res[:n_meas])

    def _update_sigma(self):
        """sigma_s from the carrier residuals of fixed solutions (sync residual + carrier noise)."""
        r, dof = self._last_res
        self.res2.append((float(r @ r), dof))
        self.res2 = self.res2[-30:]
        if len(self.res2) >= 3:
            ss = sum(a for a, _ in self.res2) / sum(d for _, d in self.res2)
            self.sig_s = float(np.clip(np.sqrt(ss), 0.3e-6, 50e-6))

    def _fixed_output(self, x):
        K, fr = len(self.frames), self.frames[-1]
        dp = x[2 * (K - 1):2 * K]
        C = fr['C']
        psi = fr['psiA'] + C[2, :2] @ np.linalg.solve(C[:2, :2], dp)
        self._kalman(x[2 * K + 2], self.s_var * 0.25)          # carrier-level sound speed
        for k, f_ in enumerate(self.frames):                    # best positions for propagation
            f_['p_best'] = f_['pA'] + x[2 * k:2 * k + 2]
        return dict(p=fr['pA'] + dp, psi=float(psi), fixed=True, s=self.s)

    def _drop_hold(self):
        """Lose the fix: the sound-speed state may carry errors of the lost solution, so it falls
        back to CIPHER-level uncertainty and re-converges before the next acquisition."""
        self.hold = None
        self.s_var = max(self.s_var, self.s_floor ** 2)

    def _start_hold(self, x, n_first):
        """Switch from the acquisition window to per-anchor hold at the window's last frame."""
        K, fr = len(self.frames), self.frames[-1]
        tau = np.round((fr['trxA'] - self.frames[0]['trxA']) / self.Tf) * self.Tf
        eta = float(x[2 * K + 1])
        n = np.full(len(self.fc), np.nan)
        n[fr['idx']] = (n_first + fr['off'])[fr['idx']]
        self.hold = dict(fr=fr, n=n, trx=float(x[2 * K] + tau * (1 + eta)), eta=eta)
        self.frames = []

    def _hold_step(self, fr):
        """One frame in hold: carry integers per anchor, solve (dp, trx, s), screen anchors."""
        h, f, I = self.hold, self.fc, len(self.fc)
        prev = h['fr']
        m = max(np.round((fr['trxA'] - prev['trxA']) / self.Tf), 1.0)
        dtf = m * self.Tf
        trx_pred = h['trx'] + dtf * (1 + h['eta'])
        p_pred = prev['p_best'] + 0.5 * (prev['vel'] + fr['vel']) * dtf
        g_prev = (prev['g0'] + prev['G'] @ (prev['p_best'] - prev['pA'])) * (1 - self.s)
        g_new = (fr['g0'] + fr['G'] @ (p_pred - fr['pA'])) * (1 - self.s)
        n = np.full(I, np.nan)
        both = [i for i in fr['idx'] if np.isfinite(h['n'][i])]
        for i in both:                                         # time-differenced carrier phase
            n[i] = h['n'][i] + np.round(f[i] * (dtf * (1 + h['eta']) + g_new[i] - g_prev[i]
                                                 - (fr['tphi'][i] - prev['tphi'][i])))
        Cp = fr['C'][:2, :2]
        sig_pred = 2e-3 + 0.03 * dtf                            # m: prediction uncertainty
        # innovation test: a CIPHER pose that contradicts the carrier prediction (blocked direct
        # paths can make CIPHER jump to a wrong lobe) is not used as a prior in this frame
        dpa = fr['pA'] - p_pred
        use_cipher = dpa @ np.linalg.solve(Cp + sig_pred ** 2 * np.eye(2), dpa) < 25.0
        # CIPHER is the independent check: repeated disagreement means the carrier solution (not
        # CIPHER) is wrong -> give the hold up rather than confirm it with its own prediction
        h['reject'] = 0 if use_cipher else h.get('reject', 0) + 1
        if h['reject'] >= self.max_reject:
            self._drop_hold()
            return None
        # persistent-bias test: a wrong integer set can be self-consistent and sit 10-20 mm off;
        # CIPHER's error averages out over frames, the offset does not
        h.setdefault('innov', []).append(fr['pA'] - p_pred)
        h['innov'] = h['innov'][-self.bias_window:]
        if len(h['innov']) >= self.bias_window // 2:
            mean_innov = np.linalg.norm(np.mean(h['innov'], 0))
            sig_c = np.sqrt(np.trace(Cp) / 2)
            if mean_innov > max(self.bias_min, 4 * sig_c / np.sqrt(len(h['innov']))):
                self._drop_hold()
                return None
        for _ in range(I):                                      # solve, drop the worst anchor if bad
            use = [i for i in fr['idx'] if np.isfinite(n[i])]
            # fewer than 3 anchors: coast on the carrier of the remaining ones + motion prediction
            # for a limited time (blockages are short), then give up
            if len(use) < 2 or (len(use) < 3 and h.get('coast', 0) >= self.max_coast):   # (default: no coasting)
                self._drop_hold()
                return None
            rows, b = [], []
            for i in use:
                rows.append(np.r_[fr['G'][i], 1.0, -fr['g0'][i]] / self.sig_s)
                b.append((fr['tphi'][i] + n[i] / f[i] - fr['g0'][i]) / self.sig_s)
            if use_cipher:
                Lw = np.linalg.cholesky(np.linalg.inv(Cp)).T        # CIPHER position
                for j in range(2):
                    rows.append(np.r_[Lw[j], 0.0, 0.0])
                    b.append(0.0)
            for j in range(2):                                     # prediction from the last fix
                e = np.zeros(4)
                e[j] = 1 / sig_pred
                rows.append(e)
                b.append((p_pred - fr['pA'])[j] / sig_pred)
            sig_t = 1e-6 * m
            rows.append(np.r_[0.0, 0.0, 1 / sig_t, 0.0])
            b.append(trx_pred / sig_t)
            rows.append(np.r_[0.0, 0.0, 0.0, 1 / np.sqrt(self.s_var)])
            b.append(self.s / np.sqrt(self.s_var))
            A, b = np.array(rows), np.array(b)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            r = (b - A @ x)[:len(use)]
            k = int(np.argmax(np.abs(r)))
            if abs(r[k]) > 4.0:                                    # cycle slip / blocked direct path
                n[use[k]] = np.nan
                continue
            break
        dp, trx, s_new = x[:2], x[2], x[3]
        h['coast'] = h.get('coast', 0) + 1 if (len(use) < 3 or not use_cipher) else 0
        # re-acquire integers of anchors that are visible but unknown (rounding at a sub-mm fix)
        g_fix = (fr['g0'] + fr['G'] @ dp) * (1 - s_new)
        for i in fr['idx']:
            if not np.isfinite(n[i]):
                u = f[i] * (trx + g_fix[i] - fr['tphi'][i])
                if abs(u - np.round(u)) < 0.15:
                    n[i] = np.round(u)
        self._last_res = r * self.sig_s, max(len(use) - 1, 1)
        self._update_sigma()
        h['eta'] += 0.2 * (trx - trx_pred) / dtf
        h['trx'], h['n'] = float(trx), n
        fr['p_best'] = fr['pA'] + dp
        h['fr'] = fr
        cov = np.linalg.inv(A.T @ A)
        self._kalman(s_new, max(cov[3, 3], 1e-12))
        psi = fr['psiA'] + fr['C'][2, :2] @ np.linalg.solve(Cp, dp)
        if not use_cipher:                                      # heading: predict, CIPHER's is suspect
            psi = h.get('psi', fr['psiA']) + fr['omega_est'] * dtf if 'omega_est' in fr else fr['psiA']
        h['psi'] = float(psi)
        return dict(p=fr['p_best'].copy(), psi=float(psi), fixed=True, s=self.s, n_anchors=len(use),
                    coast=not use_cipher or len(use) < 3)

    def update(self, est, t_win):
        """est: CIPHER estimate of the frame whose window starts at t_win (receiver time, s)."""
        self.s_var += self.s_walk
        cs, cv = est.extra.get('c_scale', 0.0), est.extra.get('c_var', np.nan)
        if np.isfinite(cv) and cv > 0 and self.hold is None:
            self._kalman(cs, cv)
            self.s_var = max(self.s_var, self.s_floor ** 2)
        I = len(self.fc)
        fr = _frame_terms(est, self.fc, self.bias, t_win)
        float_out = dict(p=est.p, psi=est.psi, fixed=False, s=self.s)
        if self.hold is not None and (self.sig_s > 1.5 * self.sigma_s_max or est.extra.get('lost', False)):
            self._drop_hold()                                  # sync too poor / CIPHER lost lock
        if self.hold is not None:
            out = self._hold_step(fr)
            if out is not None:
                return out
            self.frames = []                                   # lost: re-acquire from this frame
        fr['off'] = np.zeros(I)
        if self.frames:
            dn = self._delta_n(self.frames[-1], fr)
            if np.any(np.isnan(dn[fr['idx']])):               # an anchor (re)appeared: restart
                self.frames = []
            else:
                fr['off'] = self.frames[-1]['off'] + np.nan_to_num(dn)
        self.frames.append(fr)
        if len(self.frames) > self.window:                     # slide: re-reference the offsets
            self.frames.pop(0)
            shift = self.frames[0]['off'].copy()
            for f_ in self.frames:
                f_['off'] = f_['off'] - shift
        if np.sqrt(self.s_var) > self.s_fix_std or len(self.frames) < 2:
            return float_out
        # acquisition: candidates for the window's first-frame integers
        f0, f = self.frames[0], self.fc
        A, b0, D, _ = self._design(self.frames)
        R = np.eye(len(A)) - A @ np.linalg.pinv(A)
        ix = f0['idx']
        fbar = np.mean(f[ix])
        seen, cands = set(), []
        steps = range(-self.n_search, self.n_search + 1)
        for kc in range(-self.n_common, self.n_common + 1):
            base = np.zeros(I)
            base[ix] = np.round(f[ix] * (f0['trxA'] + kc / fbar + f0['g0'][ix] - f0['tphi'][ix]))
            for dn in itertools.product(steps, repeat=len(ix)):
                n = base.copy()
                n[ix] += dn
                key = tuple(n)
                if key in seen:
                    continue
                seen.add(key)
                r = R @ (b0 + D @ n)
                cands.append((float(r @ r), n))
        cands.sort(key=lambda t: t[0])
        x, chi2 = self._solve(cands[0][1])
        dof = self._last_res[1]
        # sigma_s from the best candidate (if it is wrong, chi2 is larger -> conservative)
        self.sig_s = float(np.clip(self.sig_s * (0.7 + 0.3 * np.sqrt(chi2 / dof)), 0.3e-6, 50e-6))
        # sigma_s above the ceiling (CIPHER regime): keep estimating sigma_s, do not fix
        if (len(cands) < 2 or cands[1][0] - cands[0][0] < self.diff_min or chi2 / dof > 4.0
                or self.sig_s > self.sigma_s_max):
            return float_out
        self._update_sigma()
        out = self._fixed_output(x)
        self._start_hold(x, cands[0][1])
        return out
