from collections import Counter

import numpy as np

from soccer_tracking.config import UNKNOWN
from soccer_tracking.teams import occlusion_flags, resolve_teams


def pos(x, y):
    return {"img": (x * 10, y * 10), "pitch": (x, y), "h": 50}


def test_resolve_teams(cfg):
    cfg.slowdown = 4
    votes = {
        1: Counter({1: 200, 0: 20}),  # 十分な票 → 確定
        2: Counter({0: 5}),  # 票が少ない → 直前に近くで途切れた 1 から引き継ぐ
        3: Counter({0: 3, 1: 4}),  # 近くに誰もいない → 少ない票の最多
        4: Counter(),  # 票なし → 不明
    }
    spans = {
        1: {"f0": 0, "f1": 300, "start": pos(100, 40), "end": pos(105, 42)},
        2: {"f0": 320, "f1": 340, "start": pos(106, 43), "end": pos(107, 43)},
        3: {"f0": 500, "f1": 520, "start": pos(60, 10), "end": pos(61, 10)},
        4: {"f0": 600, "f1": 605, "start": pos(30, 30), "end": pos(30, 30)},
    }
    team = resolve_teams(votes, spans, real_fps=100, cfg=cfg)
    assert team[1] == (1, "votes")
    assert team[2] == (1, "inherited")
    assert team[3] == (1, "weak")
    assert team[4] == (UNKNOWN, "unknown")


def test_occlusion_flags():
    boxes = np.array([[0, 0, 10, 20], [5, 0, 15, 20], [100, 0, 110, 20]], dtype=float)
    assert occlusion_flags(boxes, 0.3).tolist() == [True, True, False]
    assert occlusion_flags(boxes[:1], 0.3).tolist() == [False]
