"""ピッチの寸法と目印（メートル座標）

座標系: x は左のゴールライン=0 → 右のゴールライン=L、y は手前（カメラ側）のタッチライン=0 → 奥のタッチライン=W。
mplsoccer の pitch_type="custom" と同じ向き（左下が原点）。
ピッチの長さ・幅は会場ごとに違うので引数で受け取り、ペナルティエリアなどの寸法は規格の値を使う。
"""

import numpy as np

PA_DEPTH = 16.5  # ペナルティエリアの奥行き
PA_HALF = 20.16  # ペナルティエリアの幅の半分（40.32 / 2）
GA_DEPTH = 5.5  # ゴールエリアの奥行き
GA_HALF = 9.16  # ゴールエリアの幅の半分（18.32 / 2）
GOAL_HALF = 3.66  # ゴールの幅の半分（7.32 / 2）
CIRCLE_R = 9.15  # センターサークルとペナルティアークの半径
SPOT_DIST = 11.0  # ゴールラインからペナルティマークまで
ARC_DY = float(np.sqrt(CIRCLE_R**2 - (PA_DEPTH - SPOT_DIST) ** 2))  # ペナルティアークとPAラインの交点の、中央からのずれ

DEFAULT_LENGTH = 105.0
DEFAULT_WIDTH = 68.0


def keypoints(L=DEFAULT_LENGTH, W=DEFAULT_WIDTH):
    """クリックで指定できる目印の一覧 [(key, 日本語名, (x, y)), ...]"""
    c = W / 2
    pts = [
        ("center_spot", "センターマーク", (L / 2, c)),
        ("half_touch_near", "ハーフウェイライン×タッチライン（手前）", (L / 2, 0)),
        ("half_touch_far", "ハーフウェイライン×タッチライン（奥）", (L / 2, W)),
        ("circle_half_near", "センターサークル×ハーフウェイライン（手前）", (L / 2, c - CIRCLE_R)),
        ("circle_half_far", "センターサークル×ハーフウェイライン（奥）", (L / 2, c + CIRCLE_R)),
        ("circle_left", "センターサークルの左端", (L / 2 - CIRCLE_R, c)),
        ("circle_right", "センターサークルの右端", (L / 2 + CIRCLE_R, c)),
    ]
    for side, jp, x0, s in [("l", "左", 0.0, 1.0), ("r", "右", L, -1.0)]:
        pts += [
            (f"{side}_corner_near", f"{jp}コーナー（手前）", (x0, 0)),
            (f"{side}_corner_far", f"{jp}コーナー（奥）", (x0, W)),
            (f"{side}_pa_goal_near", f"{jp}PA ゴールライン上（手前）", (x0, c - PA_HALF)),
            (f"{side}_pa_goal_far", f"{jp}PA ゴールライン上（奥）", (x0, c + PA_HALF)),
            (f"{side}_pa_field_near", f"{jp}PA フィールド側の角（手前）", (x0 + s * PA_DEPTH, c - PA_HALF)),
            (f"{side}_pa_field_far", f"{jp}PA フィールド側の角（奥）", (x0 + s * PA_DEPTH, c + PA_HALF)),
            (f"{side}_ga_goal_near", f"{jp}ゴールエリア ゴールライン上（手前）", (x0, c - GA_HALF)),
            (f"{side}_ga_goal_far", f"{jp}ゴールエリア ゴールライン上（奥）", (x0, c + GA_HALF)),
            (f"{side}_ga_field_near", f"{jp}ゴールエリア フィールド側の角（手前）", (x0 + s * GA_DEPTH, c - GA_HALF)),
            (f"{side}_ga_field_far", f"{jp}ゴールエリア フィールド側の角（奥）", (x0 + s * GA_DEPTH, c + GA_HALF)),
            (f"{side}_spot", f"{jp}ペナルティマーク", (x0 + s * SPOT_DIST, c)),
            (f"{side}_arc_near", f"{jp}ペナルティアーク×PAライン（手前）", (x0 + s * PA_DEPTH, c - ARC_DY)),
            (f"{side}_arc_far", f"{jp}ペナルティアーク×PAライン（奥）", (x0 + s * PA_DEPTH, c + ARC_DY)),
            (f"{side}_post_near", f"{jp}ゴールポスト（手前・地面との接点）", (x0, c - GOAL_HALF)),
            (f"{side}_post_far", f"{jp}ゴールポスト（奥・地面との接点）", (x0, c + GOAL_HALF)),
        ]
    return pts


def spots(L=DEFAULT_LENGTH, W=DEFAULT_WIDTH):
    """ボールと見間違えやすい白い点（ペナルティマーク2つとセンターマーク）"""
    return [(SPOT_DIST, W / 2), (L - SPOT_DIST, W / 2), (L / 2, W / 2)]


def line_segments(L=DEFAULT_LENGTH, W=DEFAULT_WIDTH):
    """白線を折れ線の集まりとして返す [(points, weight), ...]"""
    c = W / 2
    segs = [
        ([(0, 0), (L, 0)], 1.0),  # 手前のタッチライン
        ([(0, W), (L, W)], 1.0),  # 奥のタッチライン
        ([(0, 0), (0, W)], 1.0),  # 左のゴールライン
        ([(L, 0), (L, W)], 1.0),  # 右のゴールライン
        ([(L / 2, 0), (L / 2, W)], 1.0),  # ハーフウェイライン
    ]
    t = np.linspace(0, 2 * np.pi, 90)
    segs.append((list(zip(L / 2 + CIRCLE_R * np.cos(t), c + CIRCLE_R * np.sin(t))), 1.0))  # センターサークル
    th = np.arccos((PA_DEPTH - SPOT_DIST) / CIRCLE_R)
    for x0, s in [(0.0, 1.0), (L, -1.0)]:
        segs.append(
            (
                [
                    (x0, c - PA_HALF),
                    (x0 + s * PA_DEPTH, c - PA_HALF),
                    (x0 + s * PA_DEPTH, c + PA_HALF),
                    (x0, c + PA_HALF),
                ],
                1.0,
            )
        )  # PA
        segs.append(
            (
                [
                    (x0, c - GA_HALF),
                    (x0 + s * GA_DEPTH, c - GA_HALF),
                    (x0 + s * GA_DEPTH, c + GA_HALF),
                    (x0, c + GA_HALF),
                ],
                1.0,
            )
        )  # ゴールエリア
        base = 0.0 if s > 0 else np.pi
        t = np.linspace(base - th, base + th, 30)
        segs.append((list(zip(x0 + s * SPOT_DIST + CIRCLE_R * np.cos(t), c + CIRCLE_R * np.sin(t))), 1.0))  # PAアーク
    return segs


def sample_segments(segs, step=0.5):
    """折れ線を step メートル間隔の点にする。戻り値: (points (N,2), weights (N,))"""
    pts, wts = [], []
    for seg, w in segs:
        seg = np.asarray(seg, dtype=np.float64)
        for p0, p1 in zip(seg[:-1], seg[1:]):
            n = max(2, int(np.hypot(*(p1 - p0)) / step))
            for s in np.linspace(0, 1, n, endpoint=False):
                pts.append(p0 + (p1 - p0) * s)
                wts.append(w)
    return np.array(pts), np.array(wts)


def stripe_segments(L, W, x0, period):
    """芝の縞の境界（ゴールラインと平行な直線 x = x0 + k * period）"""
    if period is None or period <= 0:
        return []
    k0 = int(np.floor((0 - x0) / period))
    k1 = int(np.ceil((L - x0) / period))
    xs = [x0 + k * period for k in range(k0, k1 + 1)]
    return [([(x, 0), (x, W)], 1.0) for x in xs if 0.5 < x < L - 0.5]
