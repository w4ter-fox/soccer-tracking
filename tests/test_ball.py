import cv2
import numpy as np

from soccer_tracking.ball import filter_ball_candidates, fit_ball_trajectory, select_ball_path


def truth(f, s=4):
    """キックで向きが変わるボールの軌道（画像座標）"""
    t = f / s
    if t < 50:
        return 300 + 8 * t, 400 - 3 * t
    if t < 100:
        x0, y0 = truth(50 * s - 1, s)
        tt = t - 50
        return x0 - 10 * tt, y0 + 4 * tt + 0.05 * tt * tt
    x0, y0 = truth(100 * s - 1, s)
    tt = t - 100
    return x0 + 3 * tt, y0 - 6 * tt


def make_candidates(rng, n=600):
    cands = {}
    for f in range(n):
        cs = []
        if not (120 <= f < 200) and rng.random() < 0.6:  # 6割しか検出できず、長い見逃しもある
            x, y = truth(f)
            cs.append((x + rng.normal(0, 1), y + rng.normal(0, 1), 10, float(rng.uniform(0.08, 0.6)), None, None))
        cs.append((900 + 0.3 * np.sin(f / 7), 500, 8, 0.15, None, None))  # 止まっている誤検出
        if rng.random() < 0.3:
            cs.append((rng.uniform(0, 1280), rng.uniform(0, 720), 8, float(rng.uniform(0.05, 0.3)), None, None))
        cands[f] = cs
    return cands


def test_path_selection_prefers_moving_trajectory(cfg, rng):
    cands = make_candidates(rng)
    path = select_ball_path(cands, cfg)
    on_ball = sum(1 for f, x, y, _ in path if np.hypot(x - truth(f)[0], y - truth(f)[1]) < 10)
    n_true = sum(1 for f in cands for c in cands[f] if np.hypot(c[0] - truth(f)[0], c[1] - truth(f)[1]) < 10)
    assert on_ball >= 0.95 * n_true
    assert on_ball >= 0.8 * len(path)


def test_ballistic_fit_rejects_outliers_and_fills_gaps(cfg, rng):
    obs = []
    for f in range(240):
        if rng.random() < 0.45:
            continue
        x, y = truth(f)
        obs.append((f, x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 10))
    obs += [(30, 900, 100, 8), (95, 900, 100, 8)]  # 単発の誤検出
    fit = fit_ball_trajectory(sorted(obs), cfg)
    err = np.array([np.hypot(v[0] - truth(f)[0], v[1] - truth(f)[1]) for f, v in fit.items()])
    assert len(fit) >= 230
    assert np.median(err) < 2.0
    assert abs(fit[30][0] - truth(30)[0]) < 10


def test_spot_filter_uses_pitch_coordinates(cfg):
    # 画像→ピッチが単純な拡大のホモグラフィ（1px = 0.1m）
    H = np.diag([0.1, 0.1, 1.0])
    spots = [(11.0, 34.0)]
    cands = [(110, 340, 8, 0.2), (110, 340, 8, 0.9), (500, 340, 8, 0.2)]
    out = filter_ball_candidates(cands, H, spots, cfg)
    assert [(round(c[4]), c[3]) for c in out] == [(11, 0.9), (50, 0.2)]
    assert len(filter_ball_candidates(cands, None, spots, cfg)) == 3


def test_homography_sanity():
    # cv2 の perspectiveTransform と同じ向きで使っていることの確認
    H = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], float)
    assert np.allclose(cv2.perspectiveTransform(np.float32([[[1, 2]]]), H)[0, 0], [6, 2])
