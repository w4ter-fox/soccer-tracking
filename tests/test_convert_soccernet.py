import cv2
import numpy as np
import pytest

from soccer_tracking.kp import convert_soccernet as cs

W_IMG, H_IMG = 960, 540


def camera(cx, cy, span):
    near, far = cy - span * 0.30, cy + span * 0.55
    src = np.float32(
        [[cx - span * 0.45, near], [cx + span * 0.45, near], [cx - span * 0.75, far], [cx + span * 0.75, far]]
    )
    dst = np.float32([[0, H_IMG], [W_IMG, H_IMG], [0, 70], [W_IMG, 70]])
    return cv2.getPerspectiveTransform(src, dst).astype(float)


def annotate(G, rng):
    """SoccerNet と同じ形式の注釈（白線のクラスごとの点、0〜1 に正規化）"""
    ann = {}
    for cls, m in cs.MODEL.items():
        uv = cs.project(G, m)
        ok = ~np.isnan(uv).any(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W_IMG) & (uv[:, 1] >= 70) & (uv[:, 1] < H_IMG)
        uv = uv[ok]
        if len(uv) < 3:
            continue
        sel = uv[np.linspace(0, len(uv) - 1, min(12, len(uv))).astype(int)] + rng.normal(0, 0.8, (min(12, len(uv)), 2))
        ann[cls] = [{"x": float(x / W_IMG), "y": float(y / H_IMG)} for x, y in sel]
    return ann


@pytest.mark.parametrize(
    "view",
    [(52.5, 41, 54), (90, 34, 40), (14, 32, 38), (75, 30, 45)],
    ids=["center_circle", "right_goal", "left_goal", "between"],
)
def test_keypoints_from_line_annotations(tmp_path, rng, view):
    import json

    G = camera(*view)
    ann_path = tmp_path / "00000.json"
    ann_path.write_text(json.dumps(annotate(G, rng)))
    out = cs.convert_one("dummy.jpg", str(ann_path), W_IMG, H_IMG)
    assert out is not None
    errs = [
        np.hypot(u - t[0], v - t[1])
        for k, (u, v) in out["keypoints"].items()
        for t in [cs.project(G, np.array([cs.KP_XY[k]]))[0]]
    ]
    assert len(errs) >= 5
    assert np.median(errs) < 1.5
    assert max(errs) < 5.0


def test_rejects_too_few_lines(tmp_path):
    import json

    ann_path = tmp_path / "00000.json"
    ann_path.write_text(json.dumps({"Middle line": [{"x": 0.5, "y": 0.2}, {"x": 0.5, "y": 0.9}]}))
    assert cs.convert_one("dummy.jpg", str(ann_path), W_IMG, H_IMG) is None
