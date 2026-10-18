"""Physically-based multichannel synthesis.

Exact for the dominant effects the theory cares about:
  * per-sample propagation delay of a moving rigid array (exact Doppler, rotation),
  * receiver clock offset and SFO (samples taken at t_n = (n/fs - delta)/(1+eta) system time),
  * residual inter-anchor sync error (per anchor, per frame),
  * shoebox image-source reflections, spherical spreading, air absorption, piston directivity,
  * per-mic resonance / phase / position errors, self-noise and ambient noise.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from scipy.special import j1

from .config import tilt_matrix
from .geometry import mic_world
from .signals import chirp_at

P_REF = 20e-6


def image_sources(src, room_size, max_order, beta=None):
    """Shoebox images (Allen & Berkley). beta: 6 reflection coefficients
    (x=0, x=Lx, y=0, y=Ly, z=0 floor, z=Lz ceiling). Returns (pos (P,3), gain (P,), mirror (P,3))."""
    L = np.asarray(room_size, float)
    beta = np.ones(6) if beta is None else np.asarray(beta, float)
    out_p, out_g, out_s = [], [], []
    rng = range(-max_order, max_order + 1)
    per_axis = []
    for ax in range(3):
        opts = []
        for n in rng:
            for q in (0, 1):
                n_lo, n_hi = abs(n - q), abs(n)          # hits on the wall at 0 / at L
                if n_lo + n_hi > max_order:
                    continue
                pos = 2 * n * L[ax] + (src[ax] if q == 0 else -src[ax])
                g = beta[2 * ax] ** n_lo * beta[2 * ax + 1] ** n_hi
                opts.append((pos, n_lo + n_hi, g, 1.0 if q == 0 else -1.0))
        per_axis.append(opts)
    for x in per_axis[0]:
        for y in per_axis[1]:
            for z in per_axis[2]:
                if x[1] + y[1] + z[1] <= max_order:
                    out_p.append((x[0], y[0], z[0]))
                    out_g.append(x[2] * y[2] * z[2])
                    out_s.append((x[3], y[3], z[3]))
    return np.array(out_p), np.array(out_g), np.array(out_s)


def piston(theta, ka):
    x = ka * np.sin(theta)
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(np.abs(x) < 1e-6, 1.0, 2 * j1(x) / x)
    return d


def resonance_ba(f0, q, fs):
    """2nd-order resonant low-pass (DC gain 1), bilinear with prewarp at f0."""
    w0 = 2 * fs * np.tan(np.pi * f0 / fs)
    return sps.bilinear([w0 ** 2], [1.0, w0 / q, w0 ** 2], fs)


def resonance_resp(f0, q, fs, f):
    b, a = resonance_ba(f0, q, fs)
    return sps.freqz(b, a, worN=np.asarray(f, float), fs=fs)[1]


@dataclass
class Truth:
    eps: np.ndarray          # (n_frames, I) anchor emission errors (s)
    mic_body: np.ndarray     # (M,3) true body positions
    f_res: np.ndarray        # (M,) true mic resonance
    phase_err: np.ndarray    # (M,) extra flat phase error (rad)
    traj: object
    delta: float
    eta: float

    def sys_time(self, t_loc):
        return (np.asarray(t_loc) - self.delta) / (1 + self.eta)

    def pose(self, t_loc):
        return self.traj(self.sys_time(t_loc))


def synthesize(scn, traj, n_frames, rng=None):
    rng = np.random.default_rng(scn.seed) if rng is None else rng
    sig, arr, room, anc, clk, mic = scn.signal, scn.array, scn.room, scn.anchors, scn.clock, scn.mic
    fs, c = sig.fs, scn.c
    I, M = len(anc.pos), arr.n_mics
    delta, eta = clk.offset, clk.drift_ppm * 1e-6

    hw = rng if scn.hw_seed is None else np.random.default_rng(scn.hw_seed)
    mic_body = arr.mic_body() @ tilt_matrix(*arr.tilt_deg).T     # mounting tilt (body frame)
    if mic.pos_err > 0:
        d = hw.normal(0, mic.pos_err, (M, 2))
        # the array frame is defined by the actual ports: no common translation / rotation
        d -= d.mean(0)
        ang = arr.mic_angles()
        tang = np.stack([-np.sin(ang), np.cos(ang)], 1)
        d -= np.mean(np.sum(d * tang, 1)) * tang
        mic_body[:, :2] += d
    f_res = mic.f_res * (1 + mic.f_res_spread * hw.standard_normal(M))
    phase_err = np.deg2rad(mic.phase_err_deg) * hw.standard_normal(M)
    eps = rng.normal(0, anc.sync_jitter, (n_frames, I))

    n_tot = int(np.ceil(((n_frames + 1) * sig.frame_len * (1 + eta) + delta) * fs)) + 1
    y = np.zeros((M, n_tot))
    pa_1m = P_REF * 10 ** (anc.spl_30cm / 20) * 0.3        # on-axis pressure at 1 m (Pa)
    fs_per_pa = 10 ** (mic.sens_dbfs / 20)
    axes = anc.axes(room, arr.height)
    T = sig.chirp_len
    alpha_air = room.air_db_per_m / 20 * np.log(10)    # Np/m

    for i in range(I):
        f0, f1 = sig.bands[i]
        ka = 2 * np.pi * 0.5 * (f0 + f1) / c * anc.piston_radius
        beta = room.betas()
        if np.any(beta > 0) and room.max_order > 0:
            img, gain_refl, mirror = image_sources(anc.pos[i], room.size, room.max_order, beta)
        else:
            img, gain_refl, mirror = anc.pos[i][None], np.array([1.0]), np.ones((1, 3))
        direct = int(np.argmin(np.linalg.norm(img - anc.pos[i], axis=1)))
        # unmodelled effects: point scatterers (anchor -> S -> mic) and a reflector moving with the
        # robot below the array plane (static horizontal plane -> one more image source)
        r1 = np.zeros(len(img))
        if room.scatterers:
            S = np.array([sc[:3] for sc in room.scatterers], float)
            img = np.vstack([img, S])
            gain_refl = np.concatenate([gain_refl, [sc[3] for sc in room.scatterers]])
            mirror = np.vstack([mirror, np.ones((len(S), 3))])
            r1 = np.concatenate([r1, np.linalg.norm(S - anc.pos[i], axis=1)])
        plate = -1
        if room.body_plate is not None:
            zp = arr.height - room.body_plate[0]
            plate = len(img)
            img = np.vstack([img, [anc.pos[i][0], anc.pos[i][1], 2 * zp - anc.pos[i][2]]])
            gain_refl = np.concatenate([gain_refl, [room.body_plate[1]]])
            mirror = np.vstack([mirror, [1.0, 1.0, -1.0]])
            r1 = np.concatenate([r1, [0.0]])
        c_i = c * (1 + (scn.c_anchor_rel[i] if scn.c_anchor_rel is not None else 0.0))
        yi = np.zeros((M, n_tot))
        for k in range(n_frames):
            t_e = k * sig.frame_len + eps[k, i]
            att_d, att_a = anc.attenuation(i, t_e)
            pc, _ = traj(t_e + T / 2)
            centre = np.array([pc[0], pc[1], arr.height])
            for pth in range(len(img)):
                if pth == plate and len(room.body_plate) >= 4:
                    # finite deck (body frame half-sizes): reflect only if the specular point is on it
                    im = img[pth]
                    S = im + (centre - im) * (zp - im[2]) / (centre[2] - im[2])
                    _, ps = traj(t_e + T / 2)
                    dxy = S[:2] - centre[:2]
                    bx = np.cos(ps) * dxy[0] + np.sin(ps) * dxy[1]
                    by = -np.sin(ps) * dxy[0] + np.cos(ps) * dxy[1]
                    if abs(bx) > room.body_plate[2] or abs(by) > room.body_plate[3]:
                        continue
                d_c = np.linalg.norm(centre - img[pth]) + r1[pth]
                # departure direction in the real source frame (undo mirrors; scatterers: towards S)
                dep = (img[pth] - anc.pos[i]) / r1[pth] if r1[pth] > 0 else                     mirror[pth] * (centre - img[pth]) / np.linalg.norm(centre - img[pth])
                theta = np.arccos(np.clip(dep @ axes[i], -1, 1))
                amp0 = pa_1m * fs_per_pa * gain_refl[pth] * piston(theta, ka)                     * 10 ** (-(att_a + (att_d if pth == direct else 0.0)) / 20)
                if abs(amp0) / d_c < 1e-7:
                    continue
                t_lo = t_e + (d_c - 2.0) / c_i
                t_hi = t_e + T + (d_c + 2.0) / c_i
                n_lo = max(int(np.floor(((1 + eta) * t_lo + delta) * fs)), 0)
                n_hi = min(int(np.ceil(((1 + eta) * t_hi + delta) * fs)), n_tot)
                if n_hi <= n_lo:
                    continue
                n = np.arange(n_lo, n_hi)
                t_n = (n / fs - delta) / (1 + eta)
                p, psi = traj(t_n)
                q = mic_world(p, psi, arr.height, mic_body)              # (S,M,3)
                dist = np.linalg.norm(q - img[pth], axis=-1)              # (S,M)
                if r1[pth] > 0:                                           # bistatic scatter (1 m ref)
                    a = amp0 * np.exp(-alpha_air * (dist + r1[pth])) / (dist * r1[pth])
                    dist = dist + r1[pth]
                else:
                    a = amp0 * np.exp(-alpha_air * dist) / dist
                yi[:, n_lo:n_hi] += (a * chirp_at(t_n[:, None] - t_e - dist / c_i, f0, f1 - f0, T, sig.taper,
                                                  sig.direction(i, k))).T
        y += yi

    # receiver-side mic responses
    for m in range(M):
        b, a = resonance_ba(f_res[m], mic.q_res, fs)
        y[m] = sps.lfilter(b, a, y[m])
        if phase_err[m] != 0:
            ya = sps.hilbert(y[m])
            y[m] = np.real(ya * np.exp(1j * phase_err[m]))
        if mic.lr_skew and m % 2 == 1:
            f = np.fft.rfftfreq(y.shape[1], 1 / fs)
            y[m] = np.fft.irfft(np.fft.rfft(y[m]) * np.exp(-2j * np.pi * f * mic.lr_skew), y.shape[1])

    # noise: mic self-noise (white, 32 dBA over 20 kHz) + in-band ambient (17-23 kHz)
    self_rms = P_REF * 10 ** ((94 - mic.snr_dba) / 20) * fs_per_pa * np.sqrt((fs / 2) / 20_000)
    y += rng.normal(0, self_rms, y.shape)
    if mic.ambient_spl is not None:
        amb = rng.standard_normal(y.shape)
        sos = sps.butter(6, [16_500, 23_500], 'bandpass', fs=fs, output='sos')
        amb = sps.sosfiltfilt(sos, amb, axis=1)
        amb *= P_REF * 10 ** (mic.ambient_spl / 20) * fs_per_pa / np.std(amb)
        y += amb

    truth = Truth(eps=eps, mic_body=mic_body, f_res=f_res, phase_err=phase_err, traj=traj,
                  delta=delta, eta=eta)
    return y, truth
