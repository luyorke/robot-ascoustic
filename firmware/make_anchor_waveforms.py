"""Anchor playback waveforms for the checker scheme used in the simulation (sim/cipher/config.py).

Anchor i, frame k: one chirp of T = 40 ms in band i at the start of every T_f = 100 ms frame,
sweep direction (-1)^(i+k) (up for even i+k). Each file is a 2-frame loop (200 ms) to be played
repeatedly, starting at the synchronised instant -- all anchors start their loop at the same
(synchronised) time; the receiver detects each anchor's sweep direction itself, so no frame index
needs to be transmitted. Tukey taper 10 % (as simulated). 16-bit PCM, peak 0.9 FS.

Usage (repo root):  python firmware/make_anchor_waveforms.py [fs ...]   (default: 48000 96000)
Output: firmware/waveforms/anchor{i}_{fs}Hz.wav
"""
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sim.cipher.config import SignalCfg          # noqa: E402
from sim.cipher.signals import chirp_at           # noqa: E402


def loop(sig, i, fs):
    """Two-frame loop (frames k = 0, 1) for anchor i at sample rate fs."""
    f0, f1 = sig.bands[i]
    n_frame = int(round(sig.frame_len * fs))
    x = np.zeros(2 * n_frame)
    t = np.arange(int(np.ceil(sig.chirp_len * fs))) / fs
    for k in range(2):
        x[k * n_frame:k * n_frame + len(t)] = chirp_at(t, f0, f1 - f0, sig.chirp_len, sig.taper, sig.direction(i, k))
    return x


def main(rates):
    sig = SignalCfg()
    out_dir = os.path.join(os.path.dirname(__file__), 'waveforms')
    os.makedirs(out_dir, exist_ok=True)
    for fs in rates:
        for i in range(len(sig.bands)):
            x = loop(sig, i, fs)
            pcm = np.round(0.9 * 32767 * x / np.max(np.abs(x))).astype(np.int16)
            path = os.path.join(out_dir, f'anchor{i}_{fs}Hz.wav')
            wavfile.write(path, fs, pcm)
            f0, f1 = sig.bands[i]
            print(f'{path}: band {f0/1e3:.1f}-{f1/1e3:.1f} kHz, frame 0 {"up" if sig.direction(i, 0) > 0 else "down"}, '
                  f'frame 1 {"up" if sig.direction(i, 1) > 0 else "down"}, {len(pcm)} samples')


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [48000, 96000])
