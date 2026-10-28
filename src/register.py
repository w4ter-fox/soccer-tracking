"""ピッチの位置合わせ（全フレームの 画像→ピッチ のホモグラフィ）

pick_keyframes.py で指定したキーフレームを基準に、動画の全フレームについて
画像からピッチ（メートル座標）へのホモグラフィを求め、<動画名>_homography.npz に保存する。
track.py はこのファイルがあれば、それを使って2D座標を出す。

手順:
  1. フレーム間の動き（ORB 特徴点）を全フレームで推定する
  2. キーフレームから前向き・後ろ向きの両方向につなぎ、近いキーフレームほど重くして混ぜる
     （キーフレームの間は誤差を両側から分け合うので、ドリフトがたまらない）
  3. 各フレームで、投影したピッチの白線と芝の縞の境界を映像に合わせ込んで補正する
     （縞の幅と位置はキーフレームから自動で測る。縞はどれも同じ見た目なので、補正量には上限を付ける）
  4. 補正を時間方向に平滑化する

使い方:
  uv run python src/register.py video3.mov
  uv run python src/register.py video3.mov --debug-video   （投影したラインを重ねた確認用動画も出す）
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import supervision as sv
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitch
from track import CameraMotionEstimator, FFmpegWriter, get_pitch_mask, is_scene_cut, white_line_mask

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

PERSON_MODEL = "yolo11s.pt"   # 選手の位置（白線・縞・特徴点の検出から除くため）を取るだけなので小さいモデルで十分
PERSON_IMGSZ = 960

REFINE_EVERY = 2          # 何フレームごとに白線・縞で補正するか（間のフレームは補正量を補間する）
LINE_DIST_CAP = 20.0      # 白線・縞からの距離の上限（px）
INLIER_PX = 3.0           # この距離以内ならラインに乗っているとみなす
STRIPE_WEIGHT = 0.3       # 縞の重み（白線より弱い手がかりなので小さく）
MAX_CORRECTION_M = 2.0    # 1回の補正で動かしてよい量（m）。縞は1本ずれても当てはまるので、取り違えを防ぐ上限
KEYFRAME_MAX_CORRECTION_M = 5.0   # キーフレームの補正の上限（白線だけで合わせるので大きめでよい）
MIN_LINE_POINTS = 60      # 画面内の白線の点がこれより少ないと、白線での判定はしない
MIN_STRIPE_POINTS = 150   # 同、縞
SMOOTH_FRAMES = 15        # 平滑化の窓（フレーム数、奇数）
STRIPE_PERIOD_RANGE = (3.0, 9.0)   # 縞の幅の探索範囲（m）


# ---------------------------------------------------------------- 縞の検出

def stripe_edge_mask(frame, pitch_mask, person_boxes, line_mask):
    """芝の縞の境界（明るさの緩やかな段差）を取り出す。白線・選手・ピッチ外は除く"""
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), 3)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    valid = cv2.erode(pitch_mask, np.ones((15, 15), np.uint8)) > 0
    valid &= cv2.dilate(line_mask, np.ones((9, 9), np.uint8)) == 0
    for x1, y1, x2, y2 in np.asarray(person_boxes).astype(int):
        valid[max(y1 - 6, 0):y2 + 6, max(x1 - 6, 0):x2 + 6] = False
    if valid.sum() < 1000:
        return np.zeros(mag.shape, np.uint8)
    thr = max(np.percentile(mag[valid], 90), 1.5)
    mask = ((mag > thr) & valid).astype(np.uint8) * 255
    # 細かいノイズ（芝の模様、ボール、影）を除き、長くつながった段差だけ残す
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= frame.shape[0] * 0.08
    return (keep[lab] * 255).astype(np.uint8)


def estimate_stripes(xs, L=None):
    """縞の境界のピッチ上のx座標から、縞の幅と位置を推定する。
    境界は等間隔なので、x を周期 s で割った余りがそろう s を探す。
    s/2, s/3 でもそろってしまうので、十分そろう中で一番大きい s を選ぶ。"""
    xs = np.asarray(xs)
    if len(xs) < 300:
        return None, None, 0.0
    periods = np.arange(*STRIPE_PERIOD_RANGE, 0.01)
    R = np.array([abs(np.mean(np.exp(2j * np.pi * xs / s))) for s in periods])
    best = R.max()
    if best < 0.3:
        return None, None, float(best)
    s = periods[np.where(R >= 0.85 * best)[0].max()]
    x0 = (np.angle(np.mean(np.exp(2j * np.pi * xs / s))) / (2 * np.pi)) * s
    R_free = float(R[periods == s][0])
    # 縞は普通、ハーフウェイラインを基準に半面を等分して刈る。幅が (L/2)/n、境界かちょうど真ん中が
    # ハーフウェイラインに来る候補を試し、自由に推定した結果とほぼ同じくらい合えばそちらを使う
    # （少しの幅の誤差でも、ピッチの端では境界の位置が大きくずれるため）
    if L:
        best_snap = None
        for n in range(1, 40):
            sn = (L / 2) / n
            if not (STRIPE_PERIOD_RANGE[0] <= sn <= STRIPE_PERIOD_RANGE[1]):
                continue
            for x0n in [(L / 2) % sn, (L / 2 + sn / 2) % sn]:
                r = float(np.mean(np.cos(2 * np.pi * (xs - x0n) / sn)))
                if best_snap is None or r > best_snap[0]:
                    best_snap = (r, sn, x0n)
        if best_snap and best_snap[0] >= 0.8 * R_free:
            return float(best_snap[1]), float(best_snap[2]), best_snap[0]
    return float(s), float(x0 % s), R_free


# ---------------------------------------------------------------- 補正

def distance_map(mask):
    if mask is None or cv2.countNonZero(mask) < 100:
        return None
    return np.minimum(cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3), LINE_DIST_CAP).astype(np.float64)


class Refiner:
    """画像上の4つのアンカー点のピッチ座標（8変数、m）を動かして、
    投影した白線と縞の境界が映像の白線・縞に乗るようにする"""

    def __init__(self, anchors, L, W, stripe_period, stripe_x0):
        self.A = anchors.astype(np.float32)
        self.line_pts, self.line_w = pitch.sample_segments(pitch.line_segments(L, W), step=0.5)
        segs = pitch.stripe_segments(L, W, stripe_x0, stripe_period)
        self.stripe_pts = pitch.sample_segments(segs, step=0.5)[0] if segs else np.zeros((0, 2))

    def H_of(self, P):
        return cv2.getPerspectiveTransform(self.A, P.astype(np.float32)).astype(np.float64)

    @staticmethod
    def _sample(dist, H, pts):
        if dist is None or len(pts) == 0:
            return np.zeros(0), 0
        G = np.linalg.inv(H)
        hom = np.c_[pts, np.ones(len(pts))] @ G.T
        h, w = dist.shape
        ok = hom[:, 2] > 1e-6
        uv = hom[:, :2] / np.where(ok, hom[:, 2], 1)[:, None]
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        d = np.full(len(pts), LINE_DIST_CAP)
        if inside.any():
            d[inside] = map_coordinates(dist, [uv[inside, 1], uv[inside, 0]], order=1)
        return d, int(inside.sum()), inside

    def scores(self, H, dl, ds):
        out = {}
        for key, dist, pts in [("line", dl, self.line_pts), ("stripe", ds, self.stripe_pts)]:
            r = self._sample(dist, H, pts)
            if len(r) == 2:
                out[key] = (0.0, 0)
                continue
            d, n, inside = r
            out[key] = (float((d[inside] <= INLIER_PX).mean()) if n else 0.0, n)
        return out

    def refine(self, P0, dl, ds, bound=MAX_CORRECTION_M):
        """戻り値: (補正後のP, 補正前スコア, 補正後スコア) または None"""
        use_s = ds is not None and len(self.stripe_pts) > 0
        sw = np.sqrt(self.line_w)

        def resid(p):
            H = self.H_of(P0 + p.reshape(4, 2))
            parts = []
            if dl is not None:
                parts.append(self._sample(dl, H, self.line_pts)[0] * sw)
            if use_s:
                parts.append(self._sample(ds, H, self.stripe_pts)[0] * np.sqrt(STRIPE_WEIGHT))
            return np.concatenate(parts) if parts else np.zeros(1)

        def jac(p):
            J = None
            for i in range(8):
                e = np.zeros(8)
                e[i] = 0.1
                col = (resid(p + e) - resid(p - e)) / 0.2
                if J is None:
                    J = np.empty((len(col), 8))
                J[:, i] = col
            return J

        before = self.scores(self.H_of(P0), dl, ds)
        try:
            res = least_squares(resid, np.zeros(8), jac=jac, loss="soft_l1", f_scale=INLIER_PX,
                                x_scale=0.5, max_nfev=40,
                                bounds=(-bound, bound))
        except Exception:
            return None
        P1 = P0 + res.x.reshape(4, 2)
        after = self.scores(self.H_of(P1), dl, ds)
        return P1, before, after


def accept(before, after):
    """補正を採用するか。白線が見えていれば白線で、見えていなければ縞で判断する"""
    lb, nl = before["line"]
    la, nla = after["line"]
    sb, ns = before["stripe"]
    sa, nsa = after["stripe"]
    if nla >= MIN_LINE_POINTS:
        return la >= lb - 0.02 and (la > lb + 0.01 or sa > sb + 0.02)
    if nsa >= MIN_STRIPE_POINTS:
        return sa > sb + 0.02
    return False


# ---------------------------------------------------------------- 本体

def project(H, pts):
    hom = np.c_[pts, np.ones(len(pts))] @ H.T
    if (hom[:, 2] <= 1e-9).any():
        return None
    return hom[:, :2] / hom[:, 2:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--no-stripes", action="store_true", help="芝の縞を使わない")
    ap.add_argument("--refine-every", type=int, default=REFINE_EVERY)
    ap.add_argument("--smooth", type=int, default=SMOOTH_FRAMES, help="平滑化の窓（フレーム数、0で無効）")
    ap.add_argument("--debug-video", action="store_true", help="投影したピッチのラインと縞を重ねた確認用動画を出す")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    name = os.path.splitext(args.video)[0]
    path = os.path.join(data_dir, args.video)
    with open(os.path.join(data_dir, f"{name}_keyframes.json"), encoding="utf-8") as f:
        doc = json.load(f)
    L, W = doc["pitch_length"], doc["pitch_width"]
    kf = {k["frame"]: np.array(k["H_img2pitch"], dtype=np.float64) for k in doc["keyframes"]}
    if not kf:
        raise SystemExit("キーフレームがありません。先に pick_keyframes.py で指定してください。")
    print(f"ピッチ {L} m × {W} m / キーフレーム {sorted(kf)}")

    model = YOLO(PERSON_MODEL)
    motion = CameraMotionEstimator()

    # ---- 1. 全フレームのフレーム間の動きと選手の位置 ----
    print("1/3 フレーム間の動きを推定...")
    boxes, H_c2p, link_ok = [], [], []
    stripe_x = []
    kf_frames = {}
    prev_hist = None
    shape = None
    for f, frame in enumerate(sv.get_video_frames_generator(path)):
        shape = frame.shape
        r = model(frame, imgsz=PERSON_IMGSZ, conf=0.25, classes=[0], verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        boxes.append(b)
        H_c2p.append(motion.update(frame, b))
        cut, prev_hist = is_scene_cut(prev_hist, frame)
        link_ok.append(f > 0 and not cut)
        if f in kf:
            kf_frames[f] = frame.copy()
        if f in kf and not args.no_stripes:
            pm = get_pitch_mask(frame)
            lm = white_line_mask(frame, pm, b)
            ys, xs = np.nonzero(stripe_edge_mask(frame, pm, b, lm))
            if len(xs):
                sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 20000), replace=False)
                pxy = cv2.perspectiveTransform(np.float32(np.c_[xs[sel], ys[sel]]).reshape(-1, 1, 2), kf[f]).reshape(-1, 2)
                inside = (pxy[:, 0] > 0) & (pxy[:, 0] < L) & (pxy[:, 1] > 0) & (pxy[:, 1] < W)
                stripe_x += list(pxy[inside, 0])
    N = len(boxes)
    h_img, w_img = shape[:2]
    print(f"  {N} フレーム / カット {sum(1 for f in range(1, N) if not link_ok[f])} 箇所")

    period, x0 = None, None
    if not args.no_stripes:
        period, x0, R = estimate_stripes(stripe_x, L)
        if period:
            print(f"  芝の縞: 幅 {period:.2f} m、境界の位置 {x0:.2f} m（一致度 {R:.2f}）")
        else:
            print(f"  芝の縞は検出できませんでした（一致度 {R:.2f}）。白線だけで補正します。")

    A = np.float32([[0.1 * w_img, 0.55 * h_img], [0.9 * w_img, 0.55 * h_img],
                    [0.1 * w_img, 0.95 * h_img], [0.9 * w_img, 0.95 * h_img]])

    # ---- キーフレーム自体を白線で補正する ----
    # クリックは数 px ずれるうえ、4〜5点だと誤差がそのまま残る。キーフレームがずれていると
    # そこからつないだ全フレームがずれるので、つなぐ前に白線に合わせ込んでおく
    kref = Refiner(A, L, W, None, None)
    kf_dist, kf_score = {}, {}
    for f in sorted(kf):
        frame = kf_frames[f]
        pm = get_pitch_mask(frame)
        dl = distance_map(white_line_mask(frame, pm, boxes[f]))
        kf_dist[f] = dl
        P0 = project(kf[f], A)
        out = kref.refine(P0, dl, None, bound=KEYFRAME_MAX_CORRECTION_M) if (dl is not None and P0 is not None) else None
        if out is not None and out[2]["line"][1] >= MIN_LINE_POINTS and out[2]["line"][0] > out[1]["line"][0] + 0.01:
            kf[f] = kref.H_of(out[0])
            kf_score[f] = out[2]["line"][0]
            print(f"  キーフレーム {f}: 白線に乗っている割合 {out[1]['line'][0]:.2f} → {out[2]['line'][0]:.2f}（補正）")
        else:
            kf_score[f] = out[1]["line"][0] if out is not None else 0.0
            print(f"  キーフレーム {f}: 白線に乗っている割合 {kf_score[f]:.2f}（補正なし）")

    # 前のキーフレームから動きでつないだ結果の方が、そのキーフレーム自体より白線に合っているなら、
    # クリックが不正確だったとみなして、つないだ結果で置き換える
    order = sorted(kf)
    for a, b in zip(order[:-1], order[1:]):
        Hc = kf[a]
        broken = False
        for f in range(a + 1, b + 1):
            if not link_ok[f]:
                broken = True
                break
            Hc = Hc @ H_c2p[f]
        if broken or kf_dist.get(b) is None:
            continue
        Pc = project(Hc, A)
        if Pc is None:
            continue
        out = kref.refine(Pc, kf_dist[b], None)
        cand_H, cand_s = (kref.H_of(out[0]), out[2]["line"][0]) if out is not None else (Hc, 0.0)
        if cand_s > kf_score[b] + 0.05:
            print(f"  キーフレーム {b}: 前のキーフレームからつないだ方が白線に合う（{kf_score[b]:.2f} → {cand_s:.2f}）。"
                  f"クリックが不正確な可能性があるので置き換えます")
            kf[b], kf_score[b] = cand_H, cand_s
    kf_frames.clear()

    # ---- 2. キーフレームから前後両方向につなぐ ----
    fwd, fwd_d = [None] * N, [None] * N
    for f in range(N):
        if f in kf:
            fwd[f], fwd_d[f] = kf[f], 0
        elif f > 0 and link_ok[f] and fwd[f - 1] is not None:
            fwd[f], fwd_d[f] = fwd[f - 1] @ H_c2p[f], fwd_d[f - 1] + 1
    bwd, bwd_d = [None] * N, [None] * N
    for f in range(N - 1, -1, -1):
        if f in kf:
            bwd[f], bwd_d[f] = kf[f], 0
        elif f + 1 < N and link_ok[f + 1] and bwd[f + 1] is not None:
            bwd[f], bwd_d[f] = bwd[f + 1] @ np.linalg.inv(H_c2p[f + 1]), bwd_d[f + 1] + 1

    P_chain = [None] * N
    for f in range(N):
        pf = project(fwd[f], A) if fwd[f] is not None else None
        pb = project(bwd[f], A) if bwd[f] is not None else None
        if pf is not None and pb is not None:
            wf = bwd_d[f] / max(fwd_d[f] + bwd_d[f], 1)      # 近いキーフレームほど重く
            P_chain[f] = wf * pf + (1 - wf) * pb
        else:
            P_chain[f] = pf if pf is not None else pb
    n_valid = sum(p is not None for p in P_chain)
    print(f"  キーフレームからつながったフレーム: {n_valid}/{N}")

    # ---- 3. 白線と縞で補正 ----
    print("2/3 白線と芝の縞で補正...")
    ref = Refiner(A, L, W, period, x0)
    delta = {}
    line_score = np.zeros(N)
    n_acc = n_try = 0
    for f, frame in enumerate(sv.get_video_frames_generator(path)):
        if P_chain[f] is None or (f % args.refine_every and f not in kf):
            continue
        pm = get_pitch_mask(frame)
        lm = white_line_mask(frame, pm, boxes[f])
        dl = distance_map(lm)
        ds = distance_map(stripe_edge_mask(frame, pm, boxes[f], lm)) if period else None
        out = ref.refine(P_chain[f], dl, ds)
        n_try += 1
        if out is None:
            continue
        P1, before, after = out
        if accept(before, after):
            delta[f] = P1 - P_chain[f]
            line_score[f] = after["line"][0]
            n_acc += 1
        else:
            line_score[f] = before["line"][0]
    print(f"  補正を採用: {n_acc}/{n_try} フレーム")

    # 補正量を、つながっている区間ごとに補間して全フレームに広げ、平滑化する
    P = np.full((N, 4, 2), np.nan)
    segments, s = [], None
    for f in range(N):
        if P_chain[f] is None:
            if s is not None:
                segments.append((s, f))
            s = None
            continue
        if s is None:
            s = f
        elif not link_ok[f]:          # カットをまたいで補間・平滑化しない
            segments.append((s, f))
            s = f
    if s is not None:
        segments.append((s, N))
    for s, e in segments:
        frames = np.arange(s, e)
        base = np.array([P_chain[f] for f in frames])
        kfs = [f for f in frames if f in delta]
        if kfs:
            D = np.array([delta[f] for f in kfs]).reshape(len(kfs), 8)
            corr = np.stack([np.interp(frames, kfs, D[:, i]) for i in range(8)], axis=1).reshape(-1, 4, 2)
        else:
            corr = np.zeros_like(base)
        seq = base + corr
        win = min(args.smooth, len(frames))
        win -= 1 - win % 2           # 奇数にする
        if args.smooth and win >= 5:
            seq = savgol_filter(seq.reshape(len(frames), 8), win, 2, axis=0).reshape(-1, 4, 2)
        P[s:e] = seq
        sc = line_score[s:e]
        idx = np.nonzero(sc)[0]
        if len(idx):
            line_score[s:e] = np.interp(np.arange(e - s), idx, sc[idx])

    H = np.full((N, 3, 3), np.nan)
    valid = ~np.isnan(P[:, 0, 0])
    for f in np.nonzero(valid)[0]:
        H[f] = cv2.getPerspectiveTransform(A, P[f].astype(np.float32))
    out_npz = os.path.join(data_dir, f"{name}_homography.npz")
    np.savez(out_npz, H=H, valid=valid, score=line_score, pitch_length=L, pitch_width=W,
             stripe_period=period if period else np.nan, stripe_x0=x0 if x0 is not None else np.nan,
             keyframes=np.array(sorted(kf)))
    print(f"保存: {out_npz}（2D座標が出るフレーム {valid.sum()}/{N}）")

    # ---- 4. 確認用動画 ----
    if args.debug_video:
        print("3/3 確認用動画...")
        line_pts, _ = pitch.sample_segments(pitch.line_segments(L, W), step=0.3)
        stripe_pts = pitch.sample_segments(pitch.stripe_segments(L, W, x0, period), step=0.3)[0] if period else np.zeros((0, 2))
        meta = json.load(open(os.path.join(data_dir, f"{name}_meta.json"))) if os.path.exists(
            os.path.join(data_dir, f"{name}_meta.json")) else {}
        fps = meta.get("file_fps") or sv.VideoInfo.from_video_path(path).fps or 30
        out_mp4 = os.path.join(data_dir, f"{name}_registration.mp4")
        with FFmpegWriter(out_mp4, w_img, h_img, fps) as sink:
            for f, frame in enumerate(sv.get_video_frames_generator(path)):
                img = frame.copy()
                if valid[f]:
                    G = np.linalg.inv(H[f])
                    for pts, col in [(stripe_pts, (255, 255, 0)), (line_pts, (255, 0, 255))]:
                        if len(pts) == 0:
                            continue
                        hom = np.c_[pts, np.ones(len(pts))] @ G.T
                        ok = hom[:, 2] > 1e-6
                        uv = (hom[ok, :2] / hom[ok, 2:]).astype(int)
                        m = (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
                        img[uv[m, 1], uv[m, 0]] = col
                    cv2.putText(img, f"frame {f}  line {line_score[f]:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(img, f"frame {f}  no registration", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                sink.write_frame(img)
        print(f"保存: {out_mp4}（マゼンタ: 白線 / 水色: 縞の境界）")


if __name__ == "__main__":
    main()
