"""Offline processing of a recording without any knowledge of the receiver clock.

The receiver is not synchronised to the anchors: its clock offset and drift are unknown, so the
analysis windows (one per anchor frame) must be found and kept aligned from the signal itself.
Only the front end (first-path detection) is needed for this, so it runs as a cheap pre-pass:

* acquisition: windows stepped through the first frame period; each anchor's earliest first-path
  arrival; the first window starts `lead` before the earliest of them;
* tracking: one frame period per step, and a first-order loop keeps the earliest arrival of the
  frame at `lead` into the window (absorbs offset and receiver drift; the earliest arrival is the
  minimum of continuous functions, so it changes smoothly while the robot moves).

All chirps then lie inside the window: arrivals differ by at most (longest - shortest path)/c
(~40 ms in a 10 m room) and a chirp lasts 40 ms, within the 100 ms window.
"""
import numpy as np


def frame_starts(fe, y, n_frames=None, lead=5e-3, gain=0.5, max_step=2e-3):
    """Window start samples for consecutive anchor frames of recording y (M, n)."""
    fs = fe.fs
    nf = int(round(fe.sig.frame_len * fs))
    win = fe.win
    n_max = (y.shape[1] - win) // nf if n_frames is None else n_frames
    # acquisition: earliest arrival per anchor over one frame period of window positions
    best = np.full(fe.I, np.inf)
    for s0 in range(0, nf, nf // 4):
        if s0 + win > y.shape[1]:
            break
        obs = fe.process(y, s0)
        for i in np.where(obs.valid)[0]:
            best[i] = min(best[i], obs.n_arr[i])
    ok = np.isfinite(best)
    if not ok.any():
        raise RuntimeError('no anchor detected in the first frame period')
    # arrivals of the same emission frame: fold modulo the frame period around the earliest
    ref = np.min(best[ok])
    fold = ref + np.mod(best[ok] - ref, nf)
    st = int(round(np.min(fold) - lead * fs))
    if st < 0:
        st += nf
    starts = []
    target, lim = lead * fs, max_step * fs
    while len(starts) < n_max and st + win <= y.shape[1]:
        starts.append(st)
        obs = fe.process(y, st)
        e = obs.n_arr[obs.valid] - st if obs.valid.any() else np.array([target])
        corr = float(np.clip(gain * (np.min(e) - target), -lim, lim))
        st = int(round(st + nf + corr))
    return starts
