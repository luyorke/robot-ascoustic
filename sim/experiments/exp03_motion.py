"""Exp03: moving robot (reverberant room). CIPHER tracking with vs without motion modelling.
Run from repo root:  python -m sim.experiments.exp03_motion"""
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.estimators import cipher, tdoa_solve, track
from sim.cipher.frontend import FrontEnd
from sim.cipher.geometry import Lissajous, wrap
from sim.cipher.lagmodel import build_pulses
from sim.experiments.common import model_of, save


class Shifted:
    """Start the trajectory at phase t0 from rest: time warp s(t) = t - tr (1 - exp(-t/tr)),
    so the speed ramps up smoothly from zero (cold start happens at low speed)."""

    def __init__(self, traj, t0, t_ramp=0.4):
        # t_ramp = 0: already moving at full speed when tracking starts (cold start while moving)
        self.traj, self.t0, self.tr = traj, t0, t_ramp

    def __call__(self, t):
        t = np.maximum(np.asarray(t, float), 0.0)
        if self.tr <= 0:
            return self.traj(t + self.t0)
        return self.traj(t - self.tr * (1 - np.exp(-t / self.tr)) + self.t0)


def one(args):
    vmax, seed, n_frames = args[:3]
    scheme, ramp = (args[3], args[4]) if len(args) > 3 else ('checker', 0.4)
    scn = Scenario()
    scn.seed = seed
    scn.signal.scheme = scheme
    w = vmax / 6.7
    rng = np.random.default_rng(seed)
    traj = Shifted(Lissajous((5.0, 5.0), (3.0, 3.0), 2 * np.pi / w, 20.0), rng.uniform(0, 2 * np.pi / w), ramp)
    y, tr = synthesize(scn, traj, n_frames)
    fe, mdl = FrontEnd(scn), model_of(scn)
    lag = (build_pulses(fe, scn), scn.room.size, True)
    fs, T = scn.signal.fs, scn.signal.frame_len
    starts = [int(round((k * T * (1 + tr.eta) + tr.delta) * fs)) + 50 for k in range(n_frames)]
    res = {'CIPHER-track': [], 'CIPHER-no-motion-model': [], 'speed': [], 'yawrate': []}
    for est, t_loc in track(mdl, fe, y, starts, lag):
        p, psi = tr.pose(t_loc)
        res['CIPHER-track'].append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
        tt = tr.sys_time(t_loc)
        p2, psi2 = traj(tt + 1e-3)
        res['speed'].append(float(np.linalg.norm(p2 - p) / 1e-3))
        res['yawrate'].append(float(abs(wrap(psi2 - psi)) / 1e-3))
    # ablation: same tracking prior, but no epoch alignment / Doppler-coupling model (v = w = 0)
    prev = None
    for st in starts:
        obs = fe.process(y, st)
        est = cipher(mdl, obs, prior=prev, lag=lag)
        t_loc = st / fs + est.extra['t_ref']
        p, psi = tr.pose(t_loc)
        res['CIPHER-no-motion-model'].append((np.linalg.norm(est.p - p), abs(wrap(est.psi - psi))))
        prev = (est.p, est.psi)
    return res


def main(n_seeds=8, n_frames=20, workers=4):
    out = {}
    # checker is the operating waveform; all-up-chirp is kept only as the ablation that shows the
    # range-Doppler coupling problem (cold start while moving)
    configs = [('checker', 0.4, 'checker, slow start'), ('checker', 0.0, 'checker, moving start'),
               ('up', 0.0, 'up-chirp, moving start')]
    for vmax in [0.5, 1.0, 2.0]:
        for scheme, ramp, label in configs:
            t = time.time()
            with ProcessPoolExecutor(max_workers=workers) as ex:
                R = list(ex.map(one, [(vmax, s, n_frames, scheme, ramp) for s in range(n_seeds)]))
            s = {}
            for m in ('CIPHER-track', 'CIPHER-no-motion-model'):
                e = np.array([x for r in R for x in r[m]], float)
                s[m] = dict(pos_rmse=float(1e3 * np.sqrt(np.mean(e[:, 0] ** 2))),
                            pos_med=float(1e3 * np.median(e[:, 0])),
                            psi_rmse=float(np.rad2deg(np.sqrt(np.mean(e[:, 1] ** 2)))),
                            gross=float(np.mean(np.rad2deg(e[:, 1]) > 1.0)))
            sp = np.array([x for r in R for x in r['speed']])
            yr = np.array([x for r in R for x in r['yawrate']])
            s['speed_mean'], s['speed_max'] = float(sp.mean()), float(sp.max())
            s['yawrate_max_deg_s'] = float(np.rad2deg(yr.max()))
            out[f'vmax{vmax}/{label}'] = s
            v = s['CIPHER-track']
            print(f"vmax {vmax} | {label:24s}: CIPHER-track pos RMSE {v['pos_rmse']:8.2f} mm (med {v['pos_med']:.2f}) "
                  f"psi {v['psi_rmse']:.4f} deg gross {100*v['gross']:.0f}% | no-motion-model "
                  f"{s['CIPHER-no-motion-model']['pos_rmse']:.1f} mm ({time.time()-t:.0f} s)")
    save('exp03_motion', out)


if __name__ == '__main__':
    main()
