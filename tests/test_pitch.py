import numpy as np

from soccer_tracking import pitch


def test_keypoints_are_unique_and_inside_pitch():
    kps = pitch.keypoints(105, 68)
    keys = [k for k, _, _ in kps]
    assert len(kps) == 37
    assert len(set(keys)) == len(keys)
    for _, _, (x, y) in kps:
        assert 0 <= x <= 105 and 0 <= y <= 68


def test_left_and_right_landmarks_are_mirrored():
    xy = {k: np.array(p) for k, _, p in pitch.keypoints(105, 68)}
    for k in xy:
        if k.startswith("l_"):
            r = xy["r_" + k[2:]]
            assert np.allclose(xy[k], [105 - r[0], r[1]])


def test_penalty_arc_meets_penalty_area_line():
    # ペナルティマークから半径 9.15m の円が、ゴールラインから 16.5m の線と交わる点
    assert np.isclose((pitch.PA_DEPTH - pitch.SPOT_DIST) ** 2 + pitch.ARC_DY**2, pitch.CIRCLE_R**2)


def test_stripe_segments_are_evenly_spaced():
    segs = pitch.stripe_segments(105, 68, 0.0, 5.25)
    xs = sorted(seg[0][0][0] for seg in segs)
    assert np.allclose(np.diff(xs), 5.25)
