"""Receiver front end: per-anchor pulse compression, first-path detection, delay-gated spectra.

Everything here runs on the receiver's own sample clock only.
"""
from dataclasses import dataclass

import numpy as np

from .signals import reference


@dataclass
class FrameObs:
    n_arr: np.ndarray        # (I,) first-path arrival (fractional sample index, chirp start), array sum
    n_arr_mic: np.ndarray    # (I,M) per-mic first-path arrival (for baselines)
    Y: np.ndarray            # (I,M,K) gated complex spectra
    freqs: np.ndarray        # (I,K)
    noise_var: np.ndarray    # (I,) per-bin noise variance of Y
    k_ind: np.ndarray        # (I,) number of independent bins in Y (for likelihood scaling)
    peak_snr: np.ndarray     # (I,) matched-filter peak SNR (power, array sum)
    valid: np.ndarray        # (I,) bool
    lag_r: np.ndarray = None     # (I,M,W) baseband compressed signal around the first path
    lag_t: np.ndarray = None     # (I,W) lag times (s, frame-window time base)
    lag_t0: np.ndarray = None    # (I,) first-path lag time (s), warm start for t0
    lag_scale: np.ndarray = None  # (I,) 1/(noise var per lag sample) * (independent/total samples)
    lag_enbw: np.ndarray = None   # (I,) independent/total lag samples (noise-equivalent bw / fs)
    dirs: np.ndarray = None       # (I,) chirp direction detected per anchor (+1 up / -1 down)


def _parabolic(v, k):
    if 0 < k < len(v) - 1:
        a, b, c = v[k - 1], v[k], v[k + 1]
        den = a - 2 * b + c
        if den < 0:
            return k + 0.5 * (a - c) / den
    return float(k)


def nominal_group_delay(scn):
    """Group delay (s) of the nominal mic response over each anchor band, weighted by the
    compressed-pulse spectrum. Subtracted from TOAs: it differs between bands near the mic
    resonance and would otherwise bias the TDOA."""
    from .channel import resonance_resp
    sig = scn.signal
    out = []
    for f0, f1 in sig.bands:
        f = np.linspace(f0, f1, 400)
        H = resonance_resp(scn.mic.f_res, scn.mic.q_res, sig.fs, f)
        gd = -np.gradient(np.unwrap(np.angle(H)), 2 * np.pi * f)
        w = np.hanning(len(f) + 2)[1:-1] * np.abs(H) ** 2
        out.append(np.sum(gd * w) / np.sum(w))
    return np.array(out)


class FrontEnd:
    def __init__(self, scn, gate_mul=4.0, first_path_ratio=0.2, lookback=8e-3, bin_stride=4,
                 lag_pre=2e-3, lag_post=4e-3, lag_decim=1):
        sig = scn.signal
        self.sig, self.fs = sig, sig.fs
        self.I = len(sig.bands)
        self.win = int(round(sig.frame_len * sig.fs))
        self.nchirp = int(round(sig.chirp_len * sig.fs))
        self.L = int(2 ** np.ceil(np.log2(self.win + self.nchirp)))
        f = np.fft.fftfreq(self.L, 1 / self.fs)
        self.f = f
        self.Sref, self.Sref_dn, self.wband, self.bins = [], [], [], []
        for i in range(self.I):
            f0, f1 = sig.bands[i]
            S = np.fft.fft(reference(sig, i), self.L)
            self.Sref_dn.append(np.conj(np.fft.fft(reference(sig, i, -1), self.L)))
            sel = np.where((f >= f0) & (f <= f1))[0]
            w = np.zeros(self.L)
            w[sel] = np.hanning(len(sel) + 2)[1:-1]       # spectral taper -> low range sidelobes
            self.Sref.append(np.conj(S))
            self.wband.append(w)
            self.bins.append(sel[::bin_stride])
        bw = np.array([b[1] - b[0] for b in sig.bands])
        self.gate = (gate_mul / bw * self.fs).astype(int)     # gate width (samples) per anchor
        self.fp_ratio, self.lookback = first_path_ratio, int(lookback * self.fs)
        self.lag_pre, self.lag_post = int(lag_pre * self.fs), int(lag_post * self.fs)
        # lag-domain decimation: the baseband pulse is only ~1.5 kHz wide, 48 kHz lags are ~30x
        # oversampled -- keeping every q-th lag loses no information (weights scaled by q)
        self.lag_decim = int(lag_decim)
        # fraction of independent lag samples = noise-equivalent bandwidth / fs
        self.enbw = np.array([np.sum(np.abs(self.Sref[i] * self.wband[i]) ** 2) ** 2 /
                              np.sum(np.abs(self.Sref[i] * self.wband[i]) ** 4) / self.L
                              for i in range(self.I)])
        self.fc = np.array([0.5 * (b[0] + b[1]) for b in sig.bands])
        self.gd = nominal_group_delay(scn) * self.fs          # samples, per anchor
        # numeric calibration: per-bin variance of Y relative to lag-domain noise power * sum(g^2)
        rng = np.random.default_rng(12345)
        noise = rng.standard_normal((64, self.win))
        self.rho = np.zeros(self.I)
        for i, r in enumerate(self.compress(noise)):
            p_lag = np.mean(np.abs(r[:, :self.win]) ** 2)
            c0, half = self.win / 2, self.gate[i] / 2
            k = np.arange(int(c0 - half), int(c0 + half) + 1)
            g = np.zeros(self.L)
            g[k] = 0.5 * (1 + np.cos(np.pi * (k - c0) / half))
            Yn = np.fft.fft(r * g[None], axis=1)[:, self.bins[i]]
            self.rho[i] = np.mean(np.abs(Yn) ** 2) / (p_lag * np.sum(g ** 2))

    def compress(self, seg, dirs=None):
        """seg (M,N) -> list over anchors of complex lag-domain signals (M,L).
        dirs: per-anchor chirp direction (+1 up / -1 down) of the reference; default up."""
        X = np.fft.fft(seg, self.L, axis=1)
        dirs = np.ones(self.I) if dirs is None else dirs
        return [np.fft.ifft(X * ((self.Sref[i] if dirs[i] > 0 else self.Sref_dn[i]) * self.wband[i])[None],
                            axis=1) for i in range(self.I)]

    def first_path(self, P):
        kmax = int(np.argmax(P))
        lo = max(kmax - self.lookback, 1)
        thr = self.fp_ratio * P[kmax]
        for k in range(lo, kmax + 1):
            if P[k] >= thr and P[k] >= P[k - 1] and P[k] >= P[k + 1]:
                return _parabolic(P, k)
        return _parabolic(P, kmax)

    def process(self, y, start, gate_shift=None):
        """Process one frame window starting at global sample `start`.
        gate_shift (I,M): per-mic gate offsets (samples) relative to the array first path, normally
        the inter-mic delays predicted from a pose estimate ("pose-steered gating"). With them every
        mic's pulse is truncated identically, so Y keeps the exact pure-delay model."""
        seg = y[:, start:start + self.win]
        if self.sig.scheme == 'up':
            dirs = np.ones(self.I)
            r_all = self.compress(seg)
        else:
            # direction is detected, not assumed: compress with both references, keep the stronger
            r_up = self.compress(seg)
            r_dn = self.compress(seg, -np.ones(self.I))
            dirs = np.array([1.0 if np.max(np.sum(np.abs(u[:, :self.win]) ** 2, 0)) >=
                             np.max(np.sum(np.abs(d[:, :self.win]) ** 2, 0)) else -1.0
                             for u, d in zip(r_up, r_dn)])
            r_all = [u if dd > 0 else d for u, d, dd in zip(r_up, r_dn, dirs)]
        I, M = self.I, seg.shape[0]
        K = min(len(b) for b in self.bins)
        Y = np.zeros((I, M, K), complex)
        freqs = np.zeros((I, K))
        n_arr = np.zeros(I)
        n_arr_mic = np.zeros((I, M))
        noise_var = np.zeros(I)
        k_ind = np.zeros(I)
        psnr = np.zeros(I)
        valid = np.ones(I, bool)
        q = self.lag_decim
        W = (self.lag_pre + self.lag_post) // q
        lag_r = np.zeros((I, M, W), complex)
        lag_t = np.zeros((I, W))
        lag_t0 = np.zeros(I)
        lag_scale = np.zeros(I)
        lag_max = self.win          # valid lags (chirp start inside window)
        for i, r in enumerate(r_all):
            env = np.abs(r[:, :lag_max])
            P = np.sum(env ** 2, 0)
            n0 = self.first_path(P)
            n_arr[i] = start + n0 - self.gd[i]
            for m in range(M):
                n_arr_mic[i, m] = start + self.first_path(env[m] ** 2) - self.gd[i]
            # noise floor from lags well before the first arrival (or after, if too early)
            # noise floor: lags where the reference fully overlaps the window, excluding the
            # first path and its early reverberation
            full = np.arange(0, self.win - self.nchirp)
            excl = (full > n0 - 3 * self.gate[i]) & (full < n0 + 0.030 * self.fs)
            nz = P[full[~excl]]
            if len(nz) < 100:
                nz = P[full]
            noise_p = np.median(nz) / (M - 1 / 3)          # P ~ Gamma(M, p): median ~ p (M - 1/3)
            psnr[i] = P[int(round(n0))] / (M * noise_p) if noise_p > 0 else np.inf
            # gate centred on the first path (optionally steered per mic)
            half = self.gate[i] / 2
            shifts = np.zeros(M) if gate_shift is None else gate_shift[i]
            gm = np.zeros((M, self.L))
            for m in range(M):
                c0 = n0 + shifts[m]
                k = np.arange(int(np.floor(c0 - half)), int(np.ceil(c0 + half)) + 1)
                k = k[(k >= 0) & (k < self.L)]
                gm[m, k] = 0.5 * (1 + np.cos(np.pi * np.clip((k - c0) / half, -1, 1)))
            g = gm[0]
            G = np.fft.fft(r * gm, axis=1)
            b = self.bins[i][:K]
            Y[i] = G[:, b]
            freqs[i] = self.f[b]
            # noise variance per bin of Y: lag noise power * sum(g^2)
            noise_var[i] = noise_p * np.sum(g ** 2) * self.rho[i]
            k_ind[i] = max(1.0, (self.sig.bands[i][1] - self.sig.bands[i][0]) * np.sum(g) / self.fs)
            k_ind[i] = min(k_ind[i], K)
            valid[i] = psnr[i] > 10.0
            ks = int(np.floor(n0)) - self.lag_pre
            kk = np.clip(np.arange(ks, ks + W * q, q), 0, self.L - 1)
            lag_t[i] = kk / self.fs
            lag_r[i] = r[:, kk] * np.exp(-2j * np.pi * self.fc[i] * lag_t[i])[None]
            lag_t0[i] = n0 / self.fs
            lag_scale[i] = q * self.enbw[i] / noise_p
        return FrameObs(n_arr, n_arr_mic, Y, freqs, noise_var, k_ind, psnr, valid,
                        lag_r, lag_t, lag_t0, lag_scale, self.enbw.copy(), dirs)
