"""SoccerNet-Calibration の白線の注釈を、pitch.py の37個のキーポイントの学習データに変換する

SoccerNet-Calibration の注釈は「白線のクラスごとに、線上の点の座標（0〜1 に正規化）」で、
キーポイントそのものは入っていない。そこで画像ごとに次の手順でホモグラフィを求め、37個の目印を画像に投影する。

  1. 直線のクラス（タッチライン・ゴールライン・ハーフウェイライン・PA・ゴールエリア）を直線で当てはめ、
     交わる組み合わせの交点（コーナー、PAの角など）を求める
  2. 交点が4つ以上あれば、それでホモグラフィの初期値を作る
     センターサークル付近の画角で交点が足りない場合は、ハーフウェイライン上の点（タッチラインとの交点、
     サークルとの交点）の並び方からサークルの中心と左右の端を幾何学的に求めて初期値にする
  3. 注釈された全ての点（直線・サークル・アーク）が、投影したピッチのモデルに乗るように最適化する
  4. 当てはまりが悪い画像（放送のメインカメラ以外の画角、注釈の誤りなど）は捨てる
  5. 37個の目印を画像に投影し、画面内にあるものをラベルにする

使い方:
  uv run python -m soccer_tracking.kp.convert_soccernet --soccernet ~/data/soccernet/calibration-2023 --out ~/data/sn_keypoints
  （リポジトリの直下で実行する。--soccernet の下に train/ valid/ test/ があり、各フォルダに 00000.jpg と 00000.json が並ぶ想定）
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .. import pitch

L, W = 105.0, 68.0  # SoccerNet のピッチ寸法
C = W / 2
R = pitch.CIRCLE_R
MAX_RMSE_PX = 4.0  # 注釈点とモデルのずれ（px、90パーセンタイル）がこれを超える画像は捨てる
MIN_CLASSES = 2  # 最低限必要な白線のクラス数

# SoccerNet のクラス → pitch.py の座標でのモデル（"top" は画面の上 = 奥、"bottom" は手前）
STRAIGHT = {
    "Side line top": ((0, W), (L, W)),
    "Side line bottom": ((0, 0), (L, 0)),
    "Side line left": ((0, 0), (0, W)),
    "Side line right": ((L, 0), (L, W)),
    "Middle line": ((L / 2, 0), (L / 2, W)),
    "Big rect. left main": ((pitch.PA_DEPTH, C - pitch.PA_HALF), (pitch.PA_DEPTH, C + pitch.PA_HALF)),
    "Big rect. left top": ((0, C + pitch.PA_HALF), (pitch.PA_DEPTH, C + pitch.PA_HALF)),
    "Big rect. left bottom": ((0, C - pitch.PA_HALF), (pitch.PA_DEPTH, C - pitch.PA_HALF)),
    "Big rect. right main": ((L - pitch.PA_DEPTH, C - pitch.PA_HALF), (L - pitch.PA_DEPTH, C + pitch.PA_HALF)),
    "Big rect. right top": ((L, C + pitch.PA_HALF), (L - pitch.PA_DEPTH, C + pitch.PA_HALF)),
    "Big rect. right bottom": ((L, C - pitch.PA_HALF), (L - pitch.PA_DEPTH, C - pitch.PA_HALF)),
    "Small rect. left main": ((pitch.GA_DEPTH, C - pitch.GA_HALF), (pitch.GA_DEPTH, C + pitch.GA_HALF)),
    "Small rect. left top": ((0, C + pitch.GA_HALF), (pitch.GA_DEPTH, C + pitch.GA_HALF)),
    "Small rect. left bottom": ((0, C - pitch.GA_HALF), (pitch.GA_DEPTH, C - pitch.GA_HALF)),
    "Small rect. right main": ((L - pitch.GA_DEPTH, C - pitch.GA_HALF), (L - pitch.GA_DEPTH, C + pitch.GA_HALF)),
    "Small rect. right top": ((L, C + pitch.GA_HALF), (L - pitch.GA_DEPTH, C + pitch.GA_HALF)),
    "Small rect. right bottom": ((L, C - pitch.GA_HALF), (L - pitch.GA_DEPTH, C - pitch.GA_HALF)),
}
_th = np.arccos((pitch.PA_DEPTH - pitch.SPOT_DIST) / R)
CIRCLES = {
    "Circle central": ((L / 2, C), 0.0, 2 * np.pi),
    "Circle left": ((pitch.SPOT_DIST, C), -_th, _th),
    "Circle right": ((L - pitch.SPOT_DIST, C), np.pi - _th, np.pi + _th),
}

# 直線の組み合わせ → キーポイント
PAIRS = {
    ("Side line left", "Side line bottom"): "l_corner_near",
    ("Side line left", "Side line top"): "l_corner_far",
    ("Side line right", "Side line bottom"): "r_corner_near",
    ("Side line right", "Side line top"): "r_corner_far",
    ("Middle line", "Side line bottom"): "half_touch_near",
    ("Middle line", "Side line top"): "half_touch_far",
}
for side, sl in [("l", "left"), ("r", "right")]:
    PAIRS.update(
        {
            (f"Side line {sl}", f"Big rect. {sl} bottom"): f"{side}_pa_goal_near",
            (f"Side line {sl}", f"Big rect. {sl} top"): f"{side}_pa_goal_far",
            (f"Big rect. {sl} main", f"Big rect. {sl} bottom"): f"{side}_pa_field_near",
            (f"Big rect. {sl} main", f"Big rect. {sl} top"): f"{side}_pa_field_far",
            (f"Side line {sl}", f"Small rect. {sl} bottom"): f"{side}_ga_goal_near",
            (f"Side line {sl}", f"Small rect. {sl} top"): f"{side}_ga_goal_far",
            (f"Small rect. {sl} main", f"Small rect. {sl} bottom"): f"{side}_ga_field_near",
            (f"Small rect. {sl} main", f"Small rect. {sl} top"): f"{side}_ga_field_far",
        }
    )

KP = pitch.keypoints(L, W)
KP_XY = {k: xy for k, _, xy in KP}


def model_samples(step=0.2):
    """各クラスのモデルの曲線を細かい点列にする（距離の計算用）"""
    out = {}
    for k, (p0, p1) in STRAIGHT.items():
        n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
        t = np.linspace(0, 1, n)[:, None]
        out[k] = np.array(p0) * (1 - t) + np.array(p1) * t
    for k, ((cx, cy), a0, a1) in CIRCLES.items():
        t = np.linspace(a0, a1, max(20, int(R * (a1 - a0) / step)))
        out[k] = np.c_[cx + R * np.cos(t), cy + R * np.sin(t)]
    return out


MODEL = model_samples()


# ---------------------------------------------------------------- 幾何の道具


def fit_line(pts):
    """点列に直線を当てはめ、同次座標の直線 (a, b, c)（a^2+b^2=1）を返す"""
    vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    n = np.array([-vy, vx])
    return np.array([n[0], n[1], -(n[0] * x0 + n[1] * y0)], dtype=np.float64)


def intersect(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def ellipse_conic(pts):
    """点列に楕円を当てはめ、3x3 の円錐曲線の行列を返す"""
    if len(pts) < 5:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(np.float32(pts))
    a, b = MA / 2, ma / 2
    if a < 1 or b < 1:
        return None
    t = np.deg2rad(ang)
    c, s = np.cos(t), np.sin(t)
    A = c * c / a**2 + s * s / b**2
    B = 2 * c * s * (1 / a**2 - 1 / b**2)
    Cc = s * s / a**2 + c * c / b**2
    D = -2 * A * cx - B * cy
    E = -B * cx - 2 * Cc * cy
    F = A * cx * cx + B * cx * cy + Cc * cy * cy - 1
    return np.array([[A, B / 2, D / 2], [B / 2, Cc, E / 2], [D / 2, E / 2, F]])


def line_conic(line, Q):
    """直線と円錐曲線の交点（0〜2個）"""
    a, b, c = line
    p0 = np.array([-a * c, -b * c, 1.0])  # 直線上の原点に一番近い点
    d = np.array([-b, a, 0.0])  # 直線の向き
    qa, qb, qc = d @ Q @ d, 2 * p0 @ Q @ d, p0 @ Q @ p0
    disc = qb * qb - 4 * qa * qc
    if abs(qa) < 1e-12 or disc < 0:
        return []
    return [(p0 + t * d)[:2] for t in [(-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)]]


def center_view_points(lines, conic):
    """センターサークル付近の画角用。ハーフウェイライン上の既知の点（タッチラインとの交点、サークルとの交点）の
    並び方から、ハーフウェイライン方向の消失点を求め、サークルの中心と左右の端を出す"""
    lm = lines["Middle line"]
    ab = line_conic(lm, conic)
    if len(ab) != 2:
        return {}
    a, b = sorted(ab, key=lambda p: -p[1])  # 画面の下（手前）が先
    pts = {"circle_half_near": a, "circle_half_far": b}
    for cls, key in [("Side line bottom", "half_touch_near"), ("Side line top", "half_touch_far")]:
        if cls in lines:
            p = intersect(lm, lines[cls])
            if p is not None:
                pts[key] = p
    if len(pts) < 3:
        return {}
    # ハーフウェイライン上の位置 t（画像）とピッチの y の 1次元の射影変換 y = (p t + q) / (r t + 1)
    p0 = np.array([-lm[0] * lm[2], -lm[1] * lm[2]])
    d = np.array([-lm[1], lm[0]])
    ts = np.array([(pt - p0) @ d for pt in pts.values()])
    ys = np.array([KP_XY[k][1] for k in pts])
    M = np.c_[ts, np.ones(len(ts)), -ts * ys]
    try:
        pp, qq, rr = np.linalg.lstsq(M, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
    # y = C（中心）になる t と、y = ∞ になる t（消失点）
    t_c = (C - qq) / (pp - rr * C) if abs(pp - rr * C) > 1e-12 else None
    if t_c is None:
        return {}
    pts["center_spot"] = p0 + t_c * d
    vy = np.r_[p0 + (-1 / rr) * d, 1.0] if abs(rr) > 1e-9 else np.r_[d, 0.0]
    # 消失点からサークルへの接線の接点が、サークルの左右の端
    polar = conic @ vy
    lr = line_conic(polar / np.hypot(polar[0], polar[1]), conic)
    if len(lr) == 2:
        left, right = sorted(lr, key=lambda p: p[0])
        pts["circle_left"], pts["circle_right"] = left, right
    return pts


def _norm_T(pts):
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def _line_coef(mn, j):
    """H を9要素のベクトルにしたときの、(H^T m) の j 成分の係数"""
    c = np.zeros(9)
    for i in range(3):
        c[3 * i + j] = mn[i]
    return c


def dlt_points_lines(img_pts, pit_pts, img_lines, pit_lines):
    """点の対応と直線の対応の両方からホモグラフィ（画像→ピッチ）を線形に求める（正規化付き DLT）。
    交点が狭い範囲（ゴールエリアだけ等）にしかなくても、画面全体に伸びる白線が拘束になるので外挿に強い"""
    ref_img = list(img_pts) + [np.array([0.0, 0.0]), np.array([960.0, 540.0])]
    Ti = _norm_T(ref_img)
    Tp = _norm_T(list(pit_pts) + [np.array([0.0, 0.0]), np.array([L, W])])
    rows = []
    for p, q in zip(img_pts, pit_pts):
        x = Ti @ np.r_[p, 1.0]
        y = Tp @ np.r_[q, 1.0]
        # y × (H x) = 0
        rows.append(np.r_[np.zeros(3), -y[2] * x, y[1] * x])
        rows.append(np.r_[y[2] * x, np.zeros(3), -y[0] * x])
    Tiinv_T = np.linalg.inv(Ti).T
    Tpinv_T = np.linalg.inv(Tp).T
    for line_img, line_pit in zip(img_lines, pit_lines):
        ln = Tiinv_T @ line_img
        mn = Tpinv_T @ line_pit
        # H^T m ∝ l  →  l × (H^T m) = 0。H^T m の j 成分 = Σ_i H[i, j] m_i
        c0, c1, c2 = (_line_coef(mn, j) for j in range(3))
        rows.append(ln[1] * c2 - ln[2] * c1)
        rows.append(ln[2] * c0 - ln[0] * c2)
    A = np.array(rows)
    if len(A) < 8:
        return None
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tp) @ Hn @ Ti
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


# ---------------------------------------------------------------- 1枚の変換


def anchors(w, h):
    return np.float32([[0.1 * w, 0.55 * h], [0.9 * w, 0.55 * h], [0.1 * w, 0.95 * h], [0.9 * w, 0.95 * h]])


def project(H, pts):
    hom = np.c_[pts, np.ones(len(pts))] @ H.T
    ok = hom[:, 2] > 1e-9
    out = np.full((len(pts), 2), np.nan)
    out[ok] = hom[ok, :2] / hom[ok, 2:]
    return out


def refine(H0, obs, w, h):
    """注釈された点が、投影したモデルの曲線に乗るようにホモグラフィを最適化する（画像上の距離で）"""
    A = anchors(w, h)
    P0 = project(H0, A)
    if np.isnan(P0).any():
        return None, None

    def H_of(p):
        return cv2.getPerspectiveTransform(A, (P0 + p.reshape(4, 2)).astype(np.float32)).astype(np.float64)

    cap = [200.0]

    def resid(p):
        G = np.linalg.inv(H_of(p))
        res = []
        for cls, pts in obs.items():
            m = project(G, MODEL[cls])
            m = m[~np.isnan(m).any(1)]
            if len(m) < 2:
                res.append(np.full(len(pts), cap[0]))
                continue
            d, _ = cKDTree(m).query(pts)
            res.append(np.minimum(d, cap[0]))
        return np.concatenate(res)

    def jac(p):
        J = None
        for i in range(8):
            e = np.zeros(8)
            e[i] = 0.05
            col = (resid(p + e) - resid(p - e)) / 0.1
            if J is None:
                J = np.empty((len(col), 8))
            J[:, i] = col
        return J

    try:
        # 1段目: 初期値が交点だけから作られていると、交点から遠い白線（サークルなど）が大きくずれている。
        # 頑健な損失だとそれを外れ値として無視してしまうので、まず普通の二乗誤差で全体を引き寄せる
        r = least_squares(resid, np.zeros(8), jac=jac, loss="linear", x_scale=0.5, max_nfev=100)
        # 2段目: 注釈の誤りに引っ張られないよう、頑健な損失で仕上げる
        cap[0] = 50.0
        r = least_squares(resid, r.x, jac=jac, loss="soft_l1", f_scale=2.0, x_scale=0.5, max_nfev=100)
    except Exception:
        return None, None
    res = resid(r.x)
    return H_of(r.x), float(np.percentile(res, 90))


def orientation_ok(H, w, h):
    """メインカメラ（手前のタッチライン側から撮っている）の画角か。
    画面の左→右でピッチの x が増え、画面の下→上で y（奥）が増えること"""
    p = project(H, np.float64([[0.3 * w, 0.8 * h], [0.7 * w, 0.8 * h], [0.5 * w, 0.95 * h], [0.5 * w, 0.6 * h]]))
    if np.isnan(p).any():
        return False
    return p[1, 0] > p[0, 0] and p[3, 1] > p[2, 1]


def convert_one(img_path, ann_path, w, h):
    with open(ann_path) as f:
        ann = json.load(f)
    obs = {}
    for cls, pts in ann.items():
        if cls not in STRAIGHT and cls not in CIRCLES:
            continue
        a = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
        if len(a) == 0:
            continue
        if a.max() <= 1.5:  # 0〜1 に正規化されている
            a = a * [w, h]
        obs[cls] = a
    if len(obs) < MIN_CLASSES:
        return None

    lines = {k: fit_line(v) for k, v in obs.items() if k in STRAIGHT and len(v) >= 2}
    pts = {}
    for (a, b), key in PAIRS.items():
        if a in lines and b in lines:
            p = intersect(lines[a], lines[b])
            if p is not None and -0.05 * w <= p[0] <= 1.05 * w and -0.05 * h <= p[1] <= 1.05 * h:
                pts[key] = p
    if len(pts) < 4 and "Middle line" in lines and "Circle central" in obs:
        conic = ellipse_conic(obs["Circle central"])
        if conic is not None:
            for k, v in center_view_points(lines, conic).items():
                pts.setdefault(k, v)
    if len(pts) < 4:
        return None
    img = [np.asarray(p, dtype=np.float64) for p in pts.values()]
    pit = [np.asarray(KP_XY[k], dtype=np.float64) for k in pts]
    pit_lines = [np.cross(np.r_[STRAIGHT[k][0], 1.0], np.r_[STRAIGHT[k][1], 1.0]) for k in lines]
    H0 = dlt_points_lines(img, pit, list(lines.values()), pit_lines)
    if H0 is None or not orientation_ok(H0, w, h):
        return None
    H, rmse = refine(H0, obs, w, h)
    if H is None or rmse > MAX_RMSE_PX or not orientation_ok(H, w, h):
        return None

    G = np.linalg.inv(H)
    uv = project(G, np.array([KP_XY[k] for k, _, _ in KP]))
    kps = {}
    for (k, _, _), (u, v) in zip(KP, uv):
        if not np.isnan(u) and 0 <= u < w and 0 <= v < h:
            kps[k] = [float(u), float(v)]
    if len(kps) < 4:
        return None
    return {"image": img_path, "width": w, "height": h, "keypoints": kps, "rmse_px": rmse, "H_img2pitch": H.tolist()}


def _job(args):
    img_path, ann_path = args
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    try:
        return convert_one(img_path, ann_path, w, h)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soccernet", required=True, help="train/ valid/ test/ があるフォルダ")
    ap.add_argument("--out", required=True, help="変換結果（index_<split>.json）の出力先")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--preview", type=int, default=20, help="確認用に、ラベルを描いた画像を何枚出すか")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in args.splits:
        d = os.path.join(os.path.expanduser(args.soccernet), split)
        if not os.path.isdir(d):
            print(f"{d} がありません。スキップします。")
            continue
        jobs = []
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".jpg"):
                ann = os.path.join(d, fn[:-4] + ".json")
                if os.path.exists(ann):
                    jobs.append((os.path.join(d, fn), ann))
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_job, jobs, chunksize=32))
        items = [r for r in results if r is not None]
        with open(os.path.join(args.out, f"index_{split}.json"), "w") as f:
            json.dump(items, f)
        counts = np.zeros(len(KP), int)
        for it in items:
            for i, (k, _, _) in enumerate(KP):
                counts[i] += k in it["keypoints"]
        print(f"{split}: {len(items)}/{len(jobs)} 枚を採用")
        for (_, jp, _), c in zip(KP, counts):
            print(f"  {jp}: {c}")

        prev_dir = os.path.join(args.out, f"preview_{split}")
        os.makedirs(prev_dir, exist_ok=True)
        for it in items[:: max(1, len(items) // max(args.preview, 1))][: args.preview]:
            img = cv2.imread(it["image"])
            G = np.linalg.inv(np.array(it["H_img2pitch"]))
            for _cls, m in MODEL.items():
                uv = project(G, m)
                uv = uv[~np.isnan(uv).any(1)].astype(int)
                for u, v in uv:
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        img[v, u] = (255, 0, 255)
            for i, (k, _, _) in enumerate(KP):
                if k in it["keypoints"]:
                    u, v = map(int, it["keypoints"][k])
                    cv2.circle(img, (u, v), 4, (0, 255, 255), -1)
                    cv2.putText(img, str(i + 1), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(prev_dir, os.path.basename(it["image"])), img)
        print(f"  確認用の画像: {prev_dir}")


if __name__ == "__main__":
    main()
