import cv2
import numpy as np

from soccer_tracking import pitch
from soccer_tracking.registration import Refiner, distance_map, estimate_stripes, project
from soccer_tracking.vision import white_line_mask

W_IMG, H_IMG, L, W = 1280, 720, 105.0, 68.0


def camera():
    """ピッチ（m）→ 画像のホモグラフィ。右のペナルティエリア付近を斜め上から見る"""
    src = np.float32([[88, 20], [102, 20], [80, 50], [104, 50]])
    dst = np.float32([[150, 650], [1100, 650], [250, 200], [1050, 200]])
    return cv2.getPerspectiveTransform(src, dst).astype(float)


def render_lines(G):
    img = np.zeros((H_IMG, W_IMG, 3), np.uint8)
    img[:] = (40, 140, 40)
    pts, _ = pitch.sample_segments(pitch.line_segments(L, W), step=0.05)
    hom = np.c_[pts, np.ones(len(pts))] @ G.T
    uv = hom[:, :2] / hom[:, 2:]
    for u, v in uv:
        if 0 <= u < W_IMG and 0 <= v < H_IMG:
            cv2.circle(img, (int(u), int(v)), 2, (235, 235, 235), -1)
    return img


def test_refiner_recovers_perturbed_homography(rng):
    G = camera()
    img = render_lines(G)
    mask = white_line_mask(img, np.full((H_IMG, W_IMG), 255, np.uint8), np.zeros((0, 4)))
    dl = distance_map(mask)
    A = np.float32(
        [
            [0.1 * W_IMG, 0.55 * H_IMG],
            [0.9 * W_IMG, 0.55 * H_IMG],
            [0.1 * W_IMG, 0.95 * H_IMG],
            [0.9 * W_IMG, 0.95 * H_IMG],
        ]
    )
    ref = Refiner(A, L, W, None, None)
    P_true = project(np.linalg.inv(G), A)
    P0 = P_true + rng.normal(0, 0.8, (4, 2))  # 約 1m ずれた初期値
    P1, before, after = ref.refine(P0, dl, None)
    assert after["line"][0] > before["line"][0]
    assert np.abs(P1 - P_true).max() < 0.3


def test_stripe_period_snaps_to_half_pitch_division(rng):
    true_s = 105 / 2 / 10  # 5.25m
    xs = np.concatenate([k * true_s + rng.normal(0, 0.15, 200) for k in range(2, 18)])
    s, x0, r = estimate_stripes(xs, L)
    assert abs(s - true_s) < 1e-6
    assert min(x0 % s, s - x0 % s) < 0.2
