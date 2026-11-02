"""チームの決定（トラック単位）"""

import json
import os
from collections import Counter

import numpy as np

from .config import TEAM_NAMES, UNKNOWN


def occlusion_flags(xyxy, ratio_thr=0.3):
    """他の選手の箱と大きく重なっている検出を True にする（重なりは小さい方の箱の面積に対する比率）"""
    n = len(xyxy)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags
    x1, y1, x2, y2 = xyxy.T
    area = (x2 - x1) * (y2 - y1)
    iw = np.clip(np.minimum(x2[:, None], x2) - np.maximum(x1[:, None], x1), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2) - np.maximum(y1[:, None], y1), 0, None)
    inter = iw * ih
    ratio = inter / np.maximum(np.minimum(area[:, None], area), 1e-6)
    np.fill_diagonal(ratio, 0)
    flags = (ratio > ratio_thr).any(axis=1)
    return flags


def resolve_teams(votes, spans, real_fps, cfg):
    """トラックIDごとのチームを決める。
    1. 重なっていないフレームの色判定が十分あり、最多の色が十分多ければ、それで確定
    2. 足りなければ、直前に近くで途切れた（または直後に近くで始まった）トラックのチームを引き継ぐ
    3. それもなければ自分の少ない判定の最多、判定が1つもなければ UNKNOWN
    戻り値: {tid: (team, source)}  source は "votes" / "inherited" / "weak" / "unknown" """
    team = {}
    for tid, c in votes.items():
        n = sum(c.values())
        if n == 0:
            continue
        best, cnt = c.most_common(1)[0]
        if n >= cfg.team_min_votes * cfg.slowdown and cnt / n >= cfg.team_min_share:
            team[tid] = (best, "votes")

    max_gap = int(cfg.team_inherit_gap_s * real_fps)

    def distance(a, b):
        """a の終わり（または始まり）と b の位置の距離が閾値以内か、と距離を返す"""
        if a["pitch"] is not None and b["pitch"] is not None:
            d = float(np.hypot(*(np.subtract(a["pitch"], b["pitch"]))))
            return d, d <= cfg.team_inherit_dist_m
        d = float(np.hypot(*(np.subtract(a["img"], b["img"]))))
        return d, d <= cfg.team_inherit_dist_box * max(a["h"], b["h"])

    for tid in sorted(spans, key=lambda k: spans[k]["f0"]):
        if tid in team:
            continue
        s = spans[tid]
        cands = []
        for oid, o in spans.items():
            if oid == tid or oid not in team:
                continue
            if 0 < s["f0"] - o["f1"] <= max_gap:  # 直前に途切れたトラック
                d, ok = distance(o["end"], s["start"])
            elif 0 < o["f0"] - s["f1"] <= max_gap:  # 直後に始まったトラック
                d, ok = distance(s["end"], o["start"])
            else:
                continue
            if ok:
                cands.append((d, oid))
        if cands:
            team[tid] = (team[min(cands)[1]][0], "inherited")
        elif sum(votes.get(tid, Counter()).values()) > 0:
            team[tid] = (votes[tid].most_common(1)[0][0], "weak")
        else:
            team[tid] = (UNKNOWN, "unknown")
    return team


def load_team_overrides(base):
    """<name>_teams.json があれば、人手で付けたチームで上書きする。形式: {"12": "blue", "7": "white", ...}"""
    path = base + "_teams.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if v in TEAM_NAMES:
            out[int(k)] = TEAM_NAMES[v]
        else:
            print(f"  警告: {os.path.basename(path)} の {k}: {v!r} は不明なチーム名（white/blue/referee/gk）")
    print(f"  {os.path.basename(path)} から {len(out)} 件のチームを上書きします。")
    return out
