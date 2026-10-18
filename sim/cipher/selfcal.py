"""Self-calibration without motion capture.

The array errors (mounting tilt, port offsets, per-band mic gains / delays) and the AGV deck are
fixed in the body frame, while the directions to the anchors turn when the robot rotates in place.
So poses and array parameters can be estimated alternately from frames recorded at a few spots,
each with a slow in-place rotation (no ground truth needed):

  1. poses of all frames by CIPHER (cold start, current array model);
  2. with those poses as if they were truth: AGV deck depth (lagmodel.plate_depth_scan), then
     tilt + port offsets + gains / delays (lagmodel.calibrate);
  3. repeat with the calibrated model.

Frames whose pose-domain coherence is low (likely a wrong lobe) are left out of step 2.
Gauges: the array centre and heading zero are defined by the array itself (U1 direction), not by
an external frame -- exactly what the robot needs; per-anchor carrier constants for CIPHER-X can
then be calibrated with the self-estimated poses (carrier.calibrate_bias).
"""
import copy

import numpy as np

from .estimators import cipher
from .lagmodel import calibrate, plate_depth_scan


def apply_cal(model, cal, depth):
    m = copy.copy(model)
    m.mic_body, m.mic_gain, m.mic_delay = cal['mic_body'], cal['mic_gain'], cal['mic_delay']
    m.body_plate = depth
    return m


def self_calibrate(model, obs_list, pulses, room_size, n_iter=3, coh_keep=0.85, log=None):
    """obs_list: FrontEnd observations of frames recorded while rotating in place at a few spots.
    Returns (cal dict, deck depth or None, list of (p, psi) final pose estimates, history)."""
    lag = (pulses, room_size, True)
    mdl = copy.copy(model)
    cal, depth, hist = None, None, []
    for it in range(n_iter):
        ests = [cipher(mdl, obs, lag=lag) for obs in obs_list]
        coh = np.array([e.extra.get('coh_robust', e.coh) for e in ests])
        keep = [k for k in range(len(ests)) if np.isfinite(coh[k]) and coh[k] >= coh_keep]
        frames = [(obs_list[k], ests[k].p, ests[k].psi) for k in keep]
        depth, _ = plate_depth_scan(mdl if cal is not None else model, frames, pulses, room_size)
        base = copy.copy(model)
        base.body_plate = depth
        cal = calibrate(base, frames, pulses, room_size, tilt=True)
        mdl = apply_cal(model, cal, depth)
        hist.append(dict(iteration=it, frames_used=len(keep), deck_depth=depth, tilt_deg=cal['tilt_deg']))
        if log:
            log(f'  self-cal iteration {it}: {len(keep)}/{len(ests)} frames, deck {depth}, '
                f'tilt {np.round(cal["tilt_deg"], 3)} deg')
    poses = [(e.p, e.psi) for e in [cipher(mdl, obs, lag=lag) for obs in obs_list]]
    return cal, depth, poses, hist
