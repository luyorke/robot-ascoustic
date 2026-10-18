"""Smoke / invariance tests.  Run:  python -m sim.tests.test_core   (from repo root)"""
import time

import numpy as np

from sim.cipher.channel import synthesize
from sim.cipher.config import Scenario
from sim.cipher.crlb import crlb_pose
from sim.cipher.estimators import cipher_frame, drape_like
from sim.cipher.frontend import FrontEnd
from sim.cipher.lagmodel import build_pulses
from sim.experiments.common import model_of
from sim.cipher.geometry import Static, wrap


def run_static(scn, p, psi, n_frames=3):
    y, truth = synthesize(scn, Static(p, psi), n_frames)
    fe, mdl = FrontEnd(scn), model_of(scn)
    lag = (build_pulses(fe, scn), scn.room.size, True)
    out = []
    for k in range(n_frames):
        start = int(round((k * scn.signal.frame_len * (1 + truth.eta) + truth.delta) * scn.signal.fs)) + 50
        est, obs = cipher_frame(mdl, fe, y, start, passes=1, lag=lag)
        out.append((est, drape_like(mdl, fe.process(y, start)), obs))
    return out


def test_free_field_static():
    scn = Scenario()
    scn.room.reflection = 0.0
    scn.anchors.sync_jitter = 0.0
    p, psi = np.array([3.2, 6.1]), np.deg2rad(37.0)
    t0 = time.time()
    res = run_static(scn, p, psi)
    cov = crlb_pose(scn, p, psi, include=('coh',))
    print(f'CRLB: pos {1e3*np.sqrt(cov[0,0]+cov[1,1]):.3f} mm, psi {np.rad2deg(np.sqrt(cov[2,2])):.4f} deg')
    for est, dr, obs in res:
        ep = 1e3 * np.linalg.norm(est.p - p)
        eh = np.rad2deg(abs(wrap(est.psi - psi)))
        print(f'  CIPHER pos {ep:.3f} mm  psi {eh:.4f} deg  ratio {est.ratio:.3f} coh {est.coh:.3f} '
              f'tdoa-pos {1e3*np.linalg.norm(est.p_tdoa-p):.1f} mm  peakSNR(dB) {np.round(10*np.log10(obs.peak_snr),1)}'
              f' | DRAPE-like pos {1e3*np.linalg.norm(dr.p-p):.1f} mm psi {np.rad2deg(abs(wrap(dr.psi-psi))):.2f} deg')
        assert ep < 8 and eh < 0.05
    print(f'  ({time.time()-t0:.1f} s)')


def test_clock_immunity():
    """Coherent estimate must not depend on receiver offset/SFO or anchor sync error."""
    p, psi = np.array([6.4, 2.7]), np.deg2rad(-121.0)
    errs = []
    for off, ppm, jit in [(0.003, 0.0, 0.0), (0.0371, 80.0, 0.0), (0.0219, -50.0, 50e-6)]:
        scn = Scenario()
        scn.room.reflection = 0.0
        scn.clock.offset, scn.clock.drift_ppm, scn.anchors.sync_jitter = off, ppm, jit
        est = run_static(scn, p, psi, n_frames=2)[-1][0]
        errs.append((1e3 * np.linalg.norm(est.p - p), np.rad2deg(abs(wrap(est.psi - psi)))))
        print(f'  offset {off*1e3:.1f} ms, SFO {ppm:+.0f} ppm, sync jitter {jit*1e6:.0f} us -> '
              f'pos {errs[-1][0]:.3f} mm, psi {errs[-1][1]:.4f} deg')
    assert max(e[1] for e in errs) < 0.1


def test_frame_timing():
    """Autonomous window placement (no clock truth): every anchor's chirp lies inside each window
    for an unknown offset and a 300 ppm receiver drift."""
    from sim.cipher.channel import synthesize
    from sim.cipher.config import Scenario
    from sim.cipher.frontend import FrontEnd
    from sim.cipher.geometry import Static
    from sim.cipher.pipeline import frame_starts
    scn = Scenario()
    scn.clock.offset, scn.clock.drift_ppm = 0.0731, 300.0
    y, tr = synthesize(scn, Static((2.0, 7.5), 0.4), 12)
    fe = FrontEnd(scn)
    starts = frame_starts(fe, y, n_frames=10)
    for st in starts:
        obs = fe.process(y, st)
        assert obs.valid.all()
        e = (obs.n_arr - st) / scn.signal.fs
        assert e.min() > 1e-3 and e.max() + scn.signal.chirp_len < scn.signal.frame_len
    print(f'  {len(starts)} windows, earliest arrival {1e3 * min((fe.process(y, s).n_arr - s).min() for s in starts) / scn.signal.fs:.1f} ms into window')


if __name__ == '__main__':
    print('== free field static ==')
    test_free_field_static()
    print('== clock immunity ==')
    test_clock_immunity()
    print('== frame timing ==')
    test_frame_timing()
    print('OK')
