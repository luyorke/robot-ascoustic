"""Exp13: array tilt and self-calibration without motion capture.
Board: realistic mic errors (exp04) + mounting tilt (0.5, -0.3) deg + AGV deck 12 cm below the
array (metal, finite); model speed of sound off by 0.3 %; frame timing acquired autonomously.
Calibration data: 4 spots x 8 headings (in-place rotation). Methods:
  nominal | mocap, no tilt (previous calibration) | mocap, with tilt | self-calibration (no mocap).
Tests: static CIPHER on 8 random poses x 2 frames; moving CIPHER-X (0.5 m/s, 8 trajectories x 20
frames) with per-anchor carrier constants from mocap poses vs from self-estimated poses.
Run from repo root:  python -m sim.experiments.exp13_selfcal"""
import copy
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher import carrier
from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher, cipher_frame, track
from sim.cipher.geometry import Lissajous, Static, wrap
from sim.cipher.lagmodel import calibrate, plate_depth_scan
from sim.cipher.pipeline import frame_starts
from sim.cipher.selfcal import apply_cal, self_calibrate
from sim.experiments.common import random_pose, save
from sim.experiments.exp03_motion import Shifted
from sim.experiments.exp04_mic_errors import REALISTIC
from sim.experiments.exp07_carrier import setup

HW = 2000
TILT = (0.5, -0.3)
DECK = (0.12, 0.8, 0.35, 0.25)
SPOTS = [(2.5, 2.5), (7.5, 2.8), (7.2, 7.5), (2.8, 7.2)]
HEADINGS = np.deg2rad(np.arange(0, 360, 45))


def scenario(seed):
    scn = Scenario()
    for k, v in REALISTIC.items():
        setattr(scn.mic, k, v)
    scn.hw_seed, scn.seed, scn.signal.scheme = HW, seed, 'checker'
    scn.array.tilt_deg = TILT
    scn.room.body_plate = DECK
    return scn


def cal_frame(args):
    seed, p, psi = args
    scn = scenario(seed)
    y, _ = synthesize(scn, Static(p, psi), 2)
    fe, _, _ = setup(scn)
    return fe.process(y, frame_starts(fe, y, n_frames=1)[0]), np.asarray(p, float), psi


def run_cal(args):
    method, frames = args
    scn = scenario(0)
    fe, mdl, lag = setup(scn)
    t = time.time()
    if method == 'nominal':
        return method, None, None, None, 0.0, []
    if method == 'self':
        cal, depth, poses, hist = self_calibrate(mdl, [f[0] for f in frames], lag[0], lag[1])
        return method, cal, depth, poses, time.time() - t, hist
    depth, _ = plate_depth_scan(mdl, frames, lag[0], lag[1])
    base = copy.copy(mdl)
    base.body_plate = depth
    cal = calibrate(base, frames, lag[0], lag[1], tilt=(method == 'mocap, with tilt'))
    return method, cal, depth, [(f[1], f[2]) for f in frames], time.time() - t, []


def model_for(scn, cal, depth):
    fe, mdl, lag = setup(scn)
    return fe, (apply_cal(mdl, cal, depth) if cal is not None else mdl), lag


def static_test(args):
    seed, p, psi, cal, depth = args
    scn = scenario(seed)
    y, _ = synthesize(scn, Static(p, psi), 3)
    fe, mdl, lag = model_for(scn, cal, depth)
    out = []
    for st in frame_starts(fe, y, n_frames=2):
        est, _ = cipher_frame(mdl, fe, y, st, passes=1, lag=lag)
        out.append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
    return out


def bias_frames(args):
    """CIPHER estimates on the calibration frames with the calibrated model (for carrier constants)."""
    (obs, _, _), (p, psi), cal, depth = args
    scn = scenario(0)
    fe, mdl, lag = model_for(scn, cal, depth)
    est = cipher(mdl, obs, lag=lag)
    return est, np.asarray(p, float)


def motion_test(args):
    seed, cal, depth, bias = args
    scn = scenario(seed)
    w = 0.5 / 6.7
    rng = np.random.default_rng(seed)
    traj = Shifted(Lissajous((5.0, 5.0), (3.0, 3.0), 2 * np.pi / w, 20.0), rng.uniform(0, 2 * np.pi / w), 0.0)
    y, tr = synthesize(scn, traj, 21)
    fe, mdl, lag = model_for(scn, cal, depth)
    fs, T = scn.signal.fs, scn.signal.frame_len
    fc = np.array([0.5 * (a + b) for a, b in scn.signal.bands])
    starts = frame_starts(fe, y, n_frames=20)
    ct = carrier.CarrierTracker(fc, bias, T, sigma_s=4e-6, window=10, diff_min=8.0)
    rows = []
    for est, t_loc in track(mdl, fe, y, starts, lag, aid=ct.update):
        p, psi = tr.pose(t_loc)
        r = est.extra['aided']
        rows.append((np.linalg.norm(est.p - p), np.linalg.norm(r['p'] - p), float(r['fixed'])))
    return rows


def main(workers=4):
    rm = lambda x: float(np.sqrt(np.mean(np.square(x))))
    fc = np.array([0.5 * (a + b) for a, b in Scenario().signal.bands])
    out = {'tilt_true_deg': TILT, 'deck_true_m': DECK[0]}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        t = time.time()
        jobs = [(5000 + 10 * s + h, SPOTS[s], HEADINGS[h]) for s in range(len(SPOTS)) for h in range(len(HEADINGS))]
        frames = list(ex.map(cal_frame, jobs))
        print(f'{len(frames)} calibration frames ({time.time() - t:.0f} s)', flush=True)
        cals = {m: r for m, *r in ex.map(run_cal, [(m, frames) for m in
                                                   ('nominal', 'mocap, no tilt', 'mocap, with tilt', 'self')])}
        true_body = synthesize(scenario(0), Static((5.0, 5.0), 0.0), 1)[1].mic_body
        rng = np.random.default_rng(7)
        poses = [random_pose(rng, Scenario()) for _ in range(8)]
        for m, (cal, depth, cposes, dt, hist) in cals.items():
            R = [e for r in ex.map(static_test, [(100 + j, p, psi, cal, depth) for j, (p, psi) in enumerate(poses)])
                 for e in r]
            E = np.array(R)
            v = dict(pos_rmse=1e3 * rm(E[:, 0]), psi_rmse=float(np.rad2deg(rm(E[:, 1]))), cal_time_s=dt,
                     deck_depth=depth, tilt_deg=None if cal is None else [float(x) for x in cal['tilt_deg']],
                     port_err_mm=None if cal is None else
                     float(1e3 * np.sqrt(np.mean((cal['mic_body'] - true_body)[:, :3] ** 2))), history=hist)
            if m == 'self' and cposes is not None:
                pe = [np.linalg.norm(np.asarray(cp[0]) - f[1]) for cp, f in zip(cposes, frames)]
                v['self_pose_err_mm'] = 1e3 * rm(pe)
            out[m] = v
            save('exp13_selfcal', out)
            print(f"{m:18s}: static CIPHER {v['pos_rmse']:6.2f} mm / {v['psi_rmse']:.4f} deg | tilt est {v['tilt_deg']} "
                  f"(true {TILT}) | deck {depth} (true {DECK[0]}) | mic 3-D error {v['port_err_mm']} mm "
                  f"| cal {dt:.0f} s" + (f" | self-estimated calibration poses {v['self_pose_err_mm']:.2f} mm"
                                         if 'self_pose_err_mm' in v else ''), flush=True)
        for m in ('mocap, with tilt', 'self'):
            cal, depth, cposes, _, _ = cals[m]
            B = list(ex.map(bias_frames, [(f, cp, cal, depth) for f, cp in zip(frames, cposes)]))
            bias, _ = carrier.calibrate_bias([b[0] for b in B], [b[1] for b in B], fc)
            R = np.array([r for rr in ex.map(motion_test, [(s, cal, depth, bias) for s in range(8)]) for r in rr])
            fx = R[:, 2] > 0
            v = dict(cipher=1e3 * rm(R[:, 0]), x=1e3 * rm(R[:, 1]), fixed=float(fx.mean()),
                     fixed_x=1e3 * rm(R[fx, 1]) if fx.any() else float('nan'),
                     fixed_cipher=1e3 * rm(R[fx, 0]) if fx.any() else float('nan'),
                     worse_fix=float(np.mean(R[fx, 1] > R[fx, 0] + 3e-3)) if fx.any() else 0.0)
            out[m]['motion'] = v
            save('exp13_selfcal', out)
            print(f"{m:18s}: moving CIPHER {v['cipher']:.2f} mm | CIPHER-X fixed {100 * v['fixed']:.0f}% "
                  f"(on fixed {v['fixed_x']:.2f} vs {v['fixed_cipher']:.2f} mm), bad fixes {100 * v['worse_fix']:.0f}%",
                  flush=True)
    save('exp13_selfcal', out)


if __name__ == '__main__':
    main()
