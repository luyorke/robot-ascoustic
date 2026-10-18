"""Exp04: microphone hardware errors, calibration with known poses, and numbering-handedness check.
Run from repo root:  python -m sim.experiments.exp04_mic_errors"""
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher_frame
from sim.cipher.frontend import FrontEnd
from sim.cipher.geometry import Static, wrap
from sim.cipher.lagmodel import build_pulses, calibrate
from sim.experiments.common import model_of, random_pose, save

REALISTIC = dict(f_res_spread=0.02, phase_err_deg=5.0, pos_err=0.3e-3, lr_skew=163e-9)


def scenario(mic_err, hw_seed, seed, handed=-45.0):
    scn = Scenario()
    for k, v in mic_err.items():
        setattr(scn.mic, k, v)
    scn.hw_seed, scn.seed = hw_seed, seed
    scn.array.step_deg = handed
    return scn


def frames_of(scn, p, psi, n_frames):
    y, tr = synthesize(scn, Static(p, psi), n_frames)
    fs, T = scn.signal.fs, scn.signal.frame_len
    return y, [int(round((k * T * (1 + tr.eta) + tr.delta) * fs)) + 50 for k in range(n_frames)]


def test_trial(args):
    mic_err, hw_seed, seed, p, psi, n_frames, gamma = args
    scn = scenario(mic_err, hw_seed, seed)
    y, starts = frames_of(scn, p, psi, n_frames)
    fe, mdl = FrontEnd(scn), model_of(scn)
    if gamma is not None:
        mdl.mic_body, mdl.mic_gain, mdl.mic_delay = gamma['mic_body'], gamma['mic_gain'], gamma['mic_delay']
    lag = (build_pulses(fe, scn), scn.room.size, True)
    out = []
    for st in starts:
        est, _ = cipher_frame(mdl, fe, y, st, passes=1, lag=lag)
        out.append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
    return out


def cal_obs(args):
    """Observations of one calibration frame at a known pose (ground truth from motion capture)."""
    mic_err, hw_seed, seed, p, psi = args
    scn = scenario(mic_err, hw_seed, seed)
    y, starts = frames_of(scn, p, psi, 1)
    return FrontEnd(scn).process(y, starts[0]), p, psi


def handed_trial(args):
    """Truth: counter-clockwise numbering (+45 deg). Compare model hypotheses -45 / +45."""
    hw_seed, seed, p, psi = args
    scn = scenario(REALISTIC, hw_seed, seed, handed=+45.0)
    y, starts = frames_of(scn, p, psi, 1)
    fe = FrontEnd(scn)
    lag = (build_pulses(fe, scn), scn.room.size, True)
    res = {}
    for hyp in (-45.0, +45.0):
        s2 = copy.deepcopy(scn)
        s2.array.step_deg = hyp
        est, _ = cipher_frame(model_of(s2), fe, y, starts[0], passes=1, lag=lag)
        res[hyp] = (est.coh, np.linalg.norm(est.p - p), abs(wrap(est.psi - psi)))
    return res


def summarize(E):
    E = np.array(E, float)
    return dict(pos_rmse=float(1e3 * np.sqrt(np.mean(E[:, 0] ** 2))), pos_med=float(1e3 * np.median(E[:, 0])),
                psi_rmse=float(np.rad2deg(np.sqrt(np.mean(E[:, 1] ** 2)))),
                psi_med=float(np.rad2deg(np.median(E[:, 1]))), gross=float(np.mean(np.rad2deg(E[:, 1]) > 1)))


def main(n_poses=16, n_frames=2, parts='ABC'):
    out = {'sensitivity': {}, 'calibration': {}, 'handedness': {}}
    rng = np.random.default_rng(7)
    poses = [random_pose(rng, Scenario()) for _ in range(n_poses)]
    pool = ProcessPoolExecutor(max_workers=8)

    # --- A. sensitivity without calibration (different board per pose)
    cases = [('ideal', {})] + \
        [(f'f_res spread {x*100:g}%', dict(f_res_spread=x)) for x in (0.01, 0.02, 0.04)] + \
        [(f'phase err {x:g} deg', dict(phase_err_deg=x)) for x in (2.0, 5.0, 10.0, 20.0)] + \
        [(f'port pos err {x*1e3:g} mm', dict(pos_err=x)) for x in (0.1e-3, 0.3e-3, 1e-3)] + \
        [('PDM L/R skew 163 ns', dict(lr_skew=163e-9)), ('realistic combined', REALISTIC)]
    for name, err in (cases if 'A' in parts else []):
        t = time.time()
        R = list(pool.map(test_trial, [(err, 1000 + j, j, p, psi, n_frames, None)
                                       for j, (p, psi) in enumerate(poses)]))
        s = summarize([e for r in R for e in r])
        out['sensitivity'][name] = s
        print(f"A {name:24s} pos RMSE {s['pos_rmse']:7.2f} mm (med {s['pos_med']:.2f})  psi RMSE "
              f"{s['psi_rmse']:.4f} deg (med {s['psi_med']:.4f})  gross {100*s['gross']:.0f}%  ({time.time()-t:.0f} s)")

    # --- B. calibration with known poses (same board for calibration and test)
    for n_cal in ((0, 4, 12) if 'B' in parts else ()):
        allE = []
        for board in range(3):
            hw = 2000 + board
            gamma = None
            if n_cal:
                cal_poses = [random_pose(np.random.default_rng(900 + board * 50 + k), Scenario()) for k in range(n_cal)]
                frames = list(pool.map(cal_obs, [(REALISTIC, hw, 5000 + k, p, psi) for k, (p, psi) in enumerate(cal_poses)]))
                scn = scenario(REALISTIC, hw, 0)
                fe = FrontEnd(scn)
                gamma = calibrate(model_of(scn), frames, build_pulses(fe, scn), scn.room.size)
            R = list(pool.map(test_trial, [(REALISTIC, hw, 100 + j, p, psi, n_frames, gamma)
                                           for j, (p, psi) in enumerate(poses)]))
            allE += [e for r in R for e in r]
        s = summarize(allE)
        out['calibration'][f'n_cal={n_cal}'] = s
        print(f"B realistic errors, {n_cal:2d} calibration frames: pos RMSE {s['pos_rmse']:.2f} mm "
              f"(med {s['pos_med']:.2f}) psi RMSE {s['psi_rmse']:.4f} deg (med {s['psi_med']:.4f}) gross {100*s['gross']:.0f}%")

    # --- C. handedness of mic numbering (PCB top/bottom view unknown)
    R = [] if 'C' not in parts else list(pool.map(handed_trial, [(3000 + j, j, p, psi) for j, (p, psi) in enumerate(poses)]))
    correct = np.mean([r[45.0][0] > r[-45.0][0] for r in R]) if R else np.nan
    out['handedness'] = dict(correct_rate=float(correct),
                             coh_true=float(np.mean([r[45.0][0] for r in R])) if R else np.nan,
                             coh_wrong=float(np.mean([r[-45.0][0] for r in R])) if R else np.nan)
    if R: print(f"C handedness by coherence: correct {100*correct:.0f}%  (mean coherence true {out['handedness']['coh_true']:.3f}"
          f" vs mirrored {out['handedness']['coh_wrong']:.3f})")
    pool.shutdown()
    save('exp04_mic_errors' + ('' if parts == 'ABC' else '_' + parts), out)


if __name__ == '__main__':
    import sys
    main(parts=sys.argv[1] if len(sys.argv) > 1 else 'ABC')
