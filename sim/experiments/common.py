"""Shared experiment runner: synthesize -> front end -> estimators, per random pose (parallel)."""
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher.channel import synthesize
from sim.cipher.crlb import crlb_pose
from sim.cipher.estimators import (Model, aoa_closed_form, aoa_tdoa_nls, bearings_per_anchor,
                                   cipher_frame, drape_like, tdoa_solve)
from sim.cipher.frontend import FrontEnd
from sim.cipher.lagmodel import build_pulses
from sim.cipher.geometry import Static, wrap

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
METHODS = ('CIPHER', 'TDOA', 'DRAPE-like', 'SCALA-AOA', 'SCALA-AOA+TDOA')


def model_of(scn, sigma_toa=None, multipath=True):
    sigma_toa = max(scn.anchors.sync_jitter, 1e-6) if sigma_toa is None else sigma_toa
    return Model(scn.anchors.pos, scn.array.height, scn.array.mic_body(), scn.c, scn.signal.fs,
                 scn.signal.chirp_len, sigma_toa,
                 room_xy=((0, scn.room.size[0]), (0, scn.room.size[1])),
                 ceiling=scn.room.size[2] if multipath else None)


def random_pose(rng, scn, margin=1.0):
    lx, ly = scn.room.size[:2]
    return np.array([rng.uniform(margin, lx - margin), rng.uniform(margin, ly - margin)]), \
        rng.uniform(-np.pi, np.pi)


def errs(est_p, est_psi, p, psi):
    return float(np.linalg.norm(np.asarray(est_p) - p)), float(abs(wrap(est_psi - psi)))


def one_trial(args):
    """Run all methods on n_frames frames of one static pose. Returns dict of per-frame errors."""
    scn, p, psi, n_frames, seed, methods = args
    scn = copy.deepcopy(scn)
    scn.seed = seed
    y, tr = synthesize(scn, Static(p, psi), n_frames)
    fe, mdl = FrontEnd(scn), model_of(scn)
    lag = (build_pulses(fe, scn), scn.room.size, True)
    out = {m: [] for m in methods}
    out['ratio'] = []
    for k in range(n_frames):
        start = int(round((k * scn.signal.frame_len * (1 + tr.eta) + tr.delta) * scn.signal.fs)) + 50
        est, obs = cipher_frame(mdl, fe, y, start, passes=1, lag=lag)
        if 'CIPHER' in out:
            out['CIPHER'].append(errs(est.p, est.psi, p, psi))
            out['ratio'].append(float(est.ratio))
        if 'TDOA' in out:
            out['TDOA'].append(errs(est.p_tdoa, np.nan, p, psi))
        obs0 = fe.process(y, start)
        if 'DRAPE-like' in out:
            d = drape_like(mdl, obs0)
            out['DRAPE-like'].append(errs(d.p, d.psi, p, psi))
        if 'SCALA-AOA' in out or 'SCALA-AOA+TDOA' in out:
            th = bearings_per_anchor(mdl, obs0, est.p_tdoa)
            a = aoa_closed_form(mdl, th)
            if 'SCALA-AOA' in out:
                out['SCALA-AOA'].append(errs(a.p, a.psi, p, psi))
            if 'SCALA-AOA+TDOA' in out:
                h = aoa_tdoa_nls(mdl, obs0, th, a if np.all(np.isfinite(a.p)) else
                                 type(a)(est.p_tdoa, a.psi))
                out['SCALA-AOA+TDOA'].append(errs(h.p, h.psi, p, psi))
    return out


def run(scn, n_poses, n_frames=3, seed=0, methods=METHODS, workers=None):
    rng = np.random.default_rng(seed)
    jobs, poses = [], []
    for j in range(n_poses):
        p, psi = random_pose(rng, scn)
        poses.append((p, psi))
        jobs.append((scn, p, psi, n_frames, seed * 100003 + j, methods))
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        res = list(ex.map(one_trial, jobs))
    return poses, res


def summarize(res, methods=METHODS):
    """RMSE / median / 90th pct of position (mm) and heading (deg) per method."""
    s = {}
    for m in methods:
        e = np.array([x for r in res for x in r[m]], float)
        ep, eh = e[:, 0] * 1e3, np.rad2deg(e[:, 1])
        s[m] = dict(pos_rmse=float(np.sqrt(np.mean(ep ** 2))), pos_med=float(np.median(ep)),
                    pos_p90=float(np.percentile(ep, 90)),
                    psi_rmse=float(np.sqrt(np.nanmean(eh ** 2))) if np.any(np.isfinite(eh)) else None,
                    psi_med=float(np.nanmedian(eh)) if np.any(np.isfinite(eh)) else None,
                    psi_p90=float(np.nanpercentile(eh, 90)) if np.any(np.isfinite(eh)) else None,
                    gross=float(np.mean(eh > 1.0)) if np.any(np.isfinite(eh)) else None)
    return s


def crlb_avg(scn, poses, sigma_toa=5e-6, include=('coh', 'toa')):
    """RMS over poses of the CRLB standard deviations (pos mm, psi deg)."""
    cp, ch = [], []
    for p, psi in poses:
        cv = crlb_pose(scn, p, psi, sigma_toa=sigma_toa, include=include)
        cp.append(cv[0, 0] + cv[1, 1])
        ch.append(cv[2, 2])
    return float(1e3 * np.sqrt(np.mean(cp))), float(np.rad2deg(np.sqrt(np.mean(ch))))


def crlb_lag_avg(scn, poses, sigma_sync=None):
    """RMS over poses of the multipath-aware (estimator-structure) bound (pos mm, psi deg)."""
    from sim.cipher.crlb import crlb_lag
    cp, ch = [], []
    for p, psi in poses:
        cv = crlb_lag(scn, p, psi, sigma_sync=sigma_sync)
        cp.append(cv[0, 0] + cv[1, 1])
        ch.append(cv[2, 2])
    return float(1e3 * np.sqrt(np.mean(cp))), float(np.rad2deg(np.sqrt(np.mean(ch))))


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
