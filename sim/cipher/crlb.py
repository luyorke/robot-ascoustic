"""Cramer-Rao bounds for (x, y, psi).

Coherent (clock-free) bound: deterministic CRB with one unknown complex gain per (anchor, frequency
bin) -- this nuisance absorbs receiver clock offset/drift, anchor sync error and anchor hardware
response (Theorem 1), so the bound is exactly what a clock-immune estimator can reach.
"""
import numpy as np

from .channel import P_REF, piston, resonance_resp
from .geometry import mic_world
from .signals import reference


def direct_amplitudes(scn, p, psi):
    """Digital full-scale amplitude of each anchor's direct path at the array centre."""
    arr, anc, room, mic = scn.array, scn.anchors, scn.room, scn.mic
    centre = np.array([p[0], p[1], arr.height])
    axes = anc.axes(room, arr.height)
    pa_1m = P_REF * 10 ** (anc.spl_30cm / 20) * 0.3
    fs_per_pa = 10 ** (mic.sens_dbfs / 20)
    amps = []
    for i in range(len(anc.pos)):
        f0, f1 = scn.signal.bands[i]
        ka = 2 * np.pi * 0.5 * (f0 + f1) / scn.c * anc.piston_radius
        v = centre - anc.pos[i]
        d = np.linalg.norm(v)
        th = np.arccos(np.clip(v @ axes[i] / d, -1, 1))
        a_air = np.exp(-room.air_db_per_m / 20 * np.log(10) * d)
        amps.append(pa_1m * fs_per_pa * piston(th, ka) * a_air / d)
    return np.array(amps)


def noise_psd(scn):
    """Two-sided noise variance per sample (white approximation over the band) in FS^2."""
    mic, fs = scn.mic, scn.signal.fs
    fs_per_pa = 10 ** (mic.sens_dbfs / 20)
    self_var = (P_REF * 10 ** ((94 - mic.snr_dba) / 20) * fs_per_pa) ** 2 * (fs / 2) / 20_000
    amb_var = (P_REF * 10 ** (mic.ambient_spl / 20) * fs_per_pa) ** 2 * (fs / 2) / 7_000
    return self_var + amb_var      # per-sample variance if spread over the whole band


def crlb_pose(scn, p, psi, sigma_toa=None, amp_scale=1.0, include=('coh', 'toa')):
    """Return covariance (3,3) of (x, y, psi) for the requested information sources."""
    sig, arr = scn.signal, scn.array
    fs, c, h = sig.fs, scn.c, arr.height
    body = arr.mic_body()
    I, M = len(scn.anchors.pos), arr.n_mics
    amps = direct_amplitudes(scn, p, psi) * amp_scale
    sigma2 = noise_psd(scn)
    N = int(round(sig.chirp_len * fs))
    F = np.zeros((3, 3))
    eps = np.array([1e-6, 1e-6, 1e-6])

    def tau_all(z):
        q = mic_world(z[:2], z[2], h, body)
        return np.linalg.norm(q[None] - scn.anchors.pos[:, None], axis=-1) / c      # (I,M)

    z0 = np.array([p[0], p[1], psi])
    dtau = np.stack([(tau_all(z0 + e) - tau_all(z0 - e)) / (2 * e[k])
                     for k, e in enumerate(np.diag(eps))], -1)                     # (I,M,3)
    if 'coh' in include:
        for i in range(I):
            f0, f1 = sig.bands[i]
            S = np.fft.rfft(reference(sig, i))
            f = np.fft.rfftfreq(N, 1 / fs)
            sel = (f >= f0) & (f <= f1)
            Hm = np.abs(resonance_resp(scn.mic.f_res, scn.mic.q_res, fs, f[sel]))
            # per-bin SNR: |A S H|^2 / (N sigma2) for the DFT of length N
            snr_bin = (amps[i] * np.abs(S[sel]) * Hm) ** 2 / (N * sigma2)
            D = dtau[i]                                  # (M,3) d tau_m / d theta
            Dc = D - D.mean(0, keepdims=True)            # projection removing the common term
            for fk, sk in zip(f[sel], snr_bin):
                F += 2 * sk * (2 * np.pi * fk) ** 2 * (Dc.T @ Dc)
    if 'toa' in include:
        st = scn.anchors.sync_jitter if sigma_toa is None else sigma_toa
        # array-centre TOA with unknown common offset t0
        dc = dtau.mean(1)                                # (I,3)
        G = np.concatenate([dc, np.ones((I, 1))], 1)     # (I,4) with t0
        Ft = G.T @ G / st ** 2
        Fs = Ft[:3, :3] - np.outer(Ft[:3, 3], Ft[3, :3]) / Ft[3, 3]
        F += Fs
    return np.linalg.pinv(F)


def crlb_lag(scn, p, psi, sigma_sync=None, multipath=True, max_extra=1.6, win=(2e-3, 4e-3)):
    """Multipath-aware bound matching the CIPHER lag-domain estimator.

    Parameters: (x, y, psi) + per anchor t0 and one complex gain per modelled path (direct +
    room images with extra path < max_extra) + receiver offset t_rx tied to t0 by the sync prior
    (std sigma_sync; None -> no prior, i.e. the pure clock-immune bound).  True path gains are
    taken from the same physics as the synthesizer (reflection, directivity, spreading, air).
    Noise: lag-domain variance of the front end with its effective number of independent samples."""
    from .channel import image_sources
    from .frontend import FrontEnd
    from .lagmodel import build_pulses, image_set

    fe = FrontEnd(scn)
    pulses = build_pulses(fe, scn)
    arr, anc, room = scn.array, scn.anchors, scn.room
    fs, c, h = scn.signal.fs, scn.c, arr.height
    body = arr.mic_body()
    I, M = len(anc.pos), arr.n_mics
    axes = anc.axes(room, h)
    pa_1m = P_REF * 10 ** (anc.spl_30cm / 20) * 0.3
    fs_per_pa = 10 ** (scn.mic.sens_dbfs / 20)
    alpha_air = room.air_db_per_m / 20 * np.log(10)
    # lag-domain noise variance per sample for white input noise of variance noise_psd
    rng = np.random.default_rng(1)
    r_n = fe.compress(rng.standard_normal((16, fe.win)) * np.sqrt(noise_psd(scn)))
    t = (np.arange(-int(win[0] * fs), int(win[1] * fs))) / fs

    blocks, prior_rows = [], []
    n_nuis = []
    for i in range(I):
        src = image_set(anc.pos[i], room.size, p, h, c, max_extra) if multipath else anc.pos[i][None]
        # true complex gains of these paths
        img, gain, mirror = image_sources(anc.pos[i], room.size, max(room.max_order, 1), room.betas())
        f0, f1 = scn.signal.bands[i]
        ka = 2 * np.pi * 0.5 * (f0 + f1) / c * anc.piston_radius
        centre = np.array([p[0], p[1], h])
        amps = []
        for s_ in src:
            k = np.argmin(np.linalg.norm(img - s_, axis=1))
            if np.linalg.norm(img[k] - s_) > 1e-6:
                amps.append(0.0)
                continue
            d = np.linalg.norm(centre - img[k])
            dep = mirror[k] * (centre - img[k]) / d
            th = np.arccos(np.clip(dep @ axes[i], -1, 1))
            amps.append(pa_1m * fs_per_pa * gain[k] * piston(th, ka) * np.exp(-alpha_air * d) / d)
        amps = np.array(amps, complex)
        pu = pulses[i]
        noise_var = np.mean(np.abs(r_n[i][:, :fe.win]) ** 2)
        scale = fe.enbw[i] / noise_var

        def mean_vec(z, src=src, amps=amps, pu=pu):
            x, y_, ps, t0 = z[:4]
            cpl = amps + z[4::2][:len(amps)] + 1j * z[5::2][:len(amps)]
            q = mic_world(np.array([x, y_])[None], np.array([ps]), h, body)[0]
            tau = np.linalg.norm(q[None] - src[:, None], axis=-1) / c
            dt = tau - tau[0].mean()
            spread = tau.mean(1, keepdims=True) / tau
            S = pu(t[None, None] - t0 - dt[..., None]) * (spread * np.exp(-2j * np.pi * pu.fc * dt))[..., None]
            return np.einsum('l,lmw->mw', cpl, S).reshape(-1), tau[0].mean()

        z0 = np.concatenate([[p[0], p[1], psi, 0.0], np.zeros(2 * len(amps))])
        steps = np.concatenate([[1e-5, 1e-5, 1e-5, 1e-8], np.full(2 * len(amps), 1e-3 * max(np.abs(amps).max(), 1e-12))])
        Jc = []
        for k in range(len(z0)):
            e = np.zeros_like(z0); e[k] = steps[k]
            Jc.append((mean_vec(z0 + e)[0] - mean_vec(z0 - e)[0]) / (2 * steps[k]))
        Jc = np.stack(Jc, 1)                                           # (N, 4 + 2L)
        F = 2 * np.real(Jc.conj().T @ Jc) * scale
        blocks.append(F)
        n_nuis.append(F.shape[0] - 3)
        # gradient of the mean direct delay w.r.t. (x, y, psi) for the sync prior
        g = []
        for k in range(3):
            e = np.zeros_like(z0); e[k] = 1e-5
            g.append((mean_vec(z0 + e)[1] - mean_vec(z0 - e)[1]) / 2e-5)
        prior_rows.append(np.array(g))
    # assemble: shared (x,y,psi), per-anchor nuisance blocks, and t_rx
    n = 3 + sum(n_nuis) + 1
    F = np.zeros((n, n))
    off = 3
    t0_idx = []
    for i, Fi in enumerate(blocks):
        k = n_nuis[i]
        F[:3, :3] += Fi[:3, :3]
        F[:3, off:off + k] += Fi[:3, 3:]
        F[off:off + k, :3] += Fi[3:, :3]
        F[off:off + k, off:off + k] += Fi[3:, 3:]
        t0_idx.append(off)                     # t0 is the first nuisance of each block
        off += k
    if sigma_sync is not None:
        for i in range(I):
            a = np.zeros(n)
            a[t0_idx[i]] = 1.0
            a[-1] = -1.0
            a[:3] = -prior_rows[i]
            F += np.outer(a, a) / sigma_sync ** 2
    else:
        F[-1, -1] = 1.0                        # t_rx unused
    C = np.linalg.pinv(F)
    return C[:3, :3]
