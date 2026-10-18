"""Exp06: motion-capture calibration on one board with realistic errors -- how many frames, and
what is left over. Variants: 12/24/48 calibration frames; oracle ports (gains/delays estimated);
oracle gains + ports from the true mic responses (floor of the per-band gain/delay model).
Run from repo root:  python -m sim.experiments.exp06_calibration"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cipher.channel import resonance_resp, synthesize
from sim.cipher.config import Scenario
from sim.cipher.frontend import FrontEnd
from sim.cipher.geometry import Static
from sim.cipher.lagmodel import build_pulses, calibrate
from sim.experiments.common import model_of, random_pose, save
from sim.experiments.exp04_mic_errors import REALISTIC, cal_obs, scenario, summarize, test_trial

HW = 2000


def band_response(scn, f_res):
    """Pulse-weighted complex response and group delay (s) of a mic resonance, per band."""
    g, d = [], []
    for f0, f1 in scn.signal.bands:
        f = np.linspace(f0, f1, 200)
        H = resonance_resp(f_res, scn.mic.q_res, scn.signal.fs, f)
        gd = -np.gradient(np.unwrap(np.angle(H)), 2 * np.pi * f)
        w = np.hanning(len(f) + 2)[1:-1]
        g.append(np.sum(H * w) / np.sum(w))
        d.append(np.sum(gd * w * np.abs(H) ** 2) / np.sum(w * np.abs(H) ** 2))
    return np.array(g), np.array(d)


def oracle(scn, tr):
    """Per-(band, mic) gains and delays from the simulated truth (resonance, phase, PDM skew)."""
    g0, d0 = band_response(scn, scn.mic.f_res)
    I, M = len(scn.signal.bands), len(tr.f_res)
    G, D = np.zeros((I, M), complex), np.zeros((I, M))
    for m in range(M):
        g, d = band_response(scn, tr.f_res[m])
        G[:, m], D[:, m] = g / g0 * np.exp(1j * tr.phase_err[m]), d - d0
    fc = np.array([0.5 * (a + b) for a, b in scn.signal.bands])
    G[:, 1::2] *= np.exp(-2j * np.pi * fc[:, None] * scn.mic.lr_skew)
    D[:, 1::2] += scn.mic.lr_skew
    return dict(mic_body=tr.mic_body.copy(), mic_gain=G / G.mean(1, keepdims=True),
                mic_delay=D - D.mean(1, keepdims=True))


def main(n_test=8, n_frames=2, workers=4):
    out = {}
    scn = scenario(REALISTIC, HW, 0)
    fe = FrontEnd(scn)
    _, tr = synthesize(scn, Static((5.0, 5.0), 0.0), 1)
    rng = np.random.default_rng(7)
    poses = [random_pose(rng, Scenario()) for _ in range(n_test)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        def test(name, gamma):
            R = list(ex.map(test_trial, [(REALISTIC, HW, 100 + j, p, psi, n_frames, gamma)
                                         for j, (p, psi) in enumerate(poses)]))
            s = summarize([e for r in R for e in r])
            out[name] = s
            print(f"{name:34s} pos RMSE {s['pos_rmse']:6.2f} mm (med {s['pos_med']:.2f})  "
                  f"psi RMSE {s['psi_rmse']:.4f} deg  gross {100*s['gross']:.0f}%", flush=True)
            return s

        test('uncalibrated', None)
        test('oracle gains + ports (floor)', oracle(scn, tr))
        frames_all = list(ex.map(cal_obs, [(REALISTIC, HW, 5000 + k, *random_pose(np.random.default_rng(900 + k), Scenario()))
                                           for k in range(48)]))
        for n_cal, oracle_ports in [(12, True), (12, False), (24, False), (48, False)]:
            mdl = model_of(scn)
            if oracle_ports:
                mdl.mic_body = tr.mic_body.copy()
            t = time.time()
            cal = calibrate(mdl, frames_all[:n_cal], build_pulses(fe, scn), scn.room.size, geometry=not oracle_ports)
            err = (cal['mic_body'] - tr.mic_body)[:, :2] * 1e3
            name = f"cal {n_cal} frames" + (" (oracle ports)" if oracle_ports else "")
            print(f"{name}: {time.time()-t:.0f} s, port error rms {np.sqrt(np.mean(err**2)):.3f} mm", flush=True)
            s = test(name, cal)
            s['port_err_rms_mm'], s['cal_time_s'] = float(np.sqrt(np.mean(err ** 2))), time.time() - t
    save('exp06_calibration', out)


if __name__ == '__main__':
    main()
