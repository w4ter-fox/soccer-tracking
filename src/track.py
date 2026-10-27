import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import json
import subprocess
import pandas as pd
from collections import defaultdict, deque, Counter
from trackers import ByteTrackTracker
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

import argparse

# 既定のデータ置き場（入力動画・キーポイントJSON・出力はすべてここ）。--data-dir で変更可
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
VIDEO_FILES = ["video1.mp4", "video2.mp4"]

# ---- 検出 ----
# 大きいモデル＋高解像度推論で、小さい・重なった選手とボールを拾いやすくする
# T4で遅すぎる場合は "yolo11l.pt" / "yolo11m.pt"、IMGSZ=960 に下げる
MODEL_NAME = "yolo11x.pt"
IMGSZ = 1280
PERSON_CONF = 0.1   # 低めにする。低信頼の検出はByteTrackが隠れた選手の追跡継続にだけ使う
NMS_IOU = 0.8       # 高めにすると、重なった選手の箱がNMSで消されにくくなる
OCCLUSION_RATIO = 0.3  # 他の選手と（小さい方の箱の面積比で）これ以上重なったら色判定を履歴に入れない

# ---- ボール ----
BALL_CONF = 0.05        # 候補として拾う最低信頼度（位置の予測で誤検出を弾く前提で低め）
BALL_INIT_CONF = 0.25   # 追跡を開始するのに必要な信頼度
BALL_GATE_PX = 80       # 予測位置からこの距離（px）以内の候補だけ採用。見失うほど広げる
BALL_MAX_MISSES = 12    # 実時間換算で何フレーム見失ったら追跡をやめるか
BALL_ROI = 320          # 全体で見つからないとき、予測位置の周りをこの大きさ（px）で切り出し拡大して再検出
BALL_TRAIL = 15         # ボールの軌跡を描くフレーム数

# ---- スロー化した入力 ----
# 入力動画が実時間の何倍に引き伸ばされているか（0.25倍速なら 4、通常の動画なら 1）
# 推論は全フレームで行い、出力動画は SLOWDOWN フレームごとに1枚書き出して1倍速に戻す
SLOWDOWN = 4

# ---- ボールの弾道フィット（全フレーム処理後にまとめて実行）----
BALL_FIT_TOL_PX = 6.0   # 弾道（2次式）の予測からこれ以上外れたら、キック・バウンド・誤検出のどれか
BALL_FIT_WINDOW = 12    # 予測に使う直近の検出数
BALL_SEG_MAX_GAP = 8    # 検出の間がこのフレーム数（実時間換算）を超えたら別の軌道として扱う

# ---- 出力 ----
OUTPUT_CRF = 18         # H.264の画質（小さいほど高画質。18でほぼ劣化なし）

# 内部クラスID（ラベルには出さない。ボックス色の決定にだけ使う）
TEAM_WHITE, TEAM_BLUE, REFEREE, GOALKEEPER, BALL = 0, 1, 2, 3, 4

# クラスID順のボックス色（RGB）
BOX_PALETTE = sv.ColorPalette(colors=[
    sv.Color(r=255, g=255, b=255),  # 0: 白チーム
    sv.Color(r=30,  g=90,  b=255),  # 1: 青チーム
    sv.Color(r=255, g=215, b=0),    # 2: 審判（黄）
    sv.Color(r=0,   g=200, b=80),   # 3: GK（緑）
    sv.Color(r=255, g=140, b=0),    # 4: ボール（オレンジ）
])
BALL_BGR = (0, 140, 255)

# GK判定に必要な胴体の緑ピクセル比率（芝の写り込みとの区別用）
GK_GREEN_RATIO = 0.4
# クラス多数決に使う直近フレーム数
VOTE_WINDOW = 30

# カメラがパン・ズームする映像なら True（フレーム間の動きを推定してホモグラフィを更新する）
# 固定カメラなら False（キーポイントを取ったフレームのホモグラフィをそのまま使う）
CAMERA_MOVES = True

# ---- 動くカメラでのホモグラフィ補正 ----
# フレーム間の動き推定は少しずつ誤差がたまるので、REFINE_EVERY フレームごとに
# 投影したピッチのライン（PA・ゴールエリア・ゴールライン・アーク）を映像の白線に合わせ込んで補正する
REFINE_EVERY = 30       # 小さくするほど2Dのずれは減るが遅くなる（1回あたり約0.1秒）
REFINE_ACCEPT = 0.4     # 補正後、投影したライン点のうち白線に乗っている割合がこれ以上なら採用
LOST_SCORE = 0.2        # これを下回ったら位置合わせを見失ったとみなし、2D座標を出さない
MIN_LINE_POINTS = 80    # 画面内に投影されたライン点がこれより少なければ判定しない
DEBUG_OVERLAY = False   # True で推定中のピッチのラインを細線で重ねる（位置合わせの確認用）


def get_pitch_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pitch_mask = np.zeros_like(mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(pitch_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return pitch_mask


def classify_by_jersey_color(crop):
    """胴体部分のユニフォーム色で 白/青/黄(審判)/緑(GK) を判定する"""
    h, w = crop.shape[:2]
    # 胴体（上15〜50%、横中央50%）だけを見る。脚・芝・頭を避ける
    torso = crop[int(h * 0.15):int(h * 0.5), int(w * 0.25):int(w * 0.75)]
    if torso.size == 0:
        return -1
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    total = torso.shape[0] * torso.shape[1]

    blue = cv2.inRange(hsv, (95, 70, 40), (130, 255, 255))   # 紺は明度が低いので下限を低めに
    white = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
    yellow = cv2.inRange(hsv, (18, 80, 80), (34, 255, 255))
    green = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))

    ratios = {
        TEAM_WHITE: cv2.countNonZero(white) / total,
        TEAM_BLUE: cv2.countNonZero(blue) / total,
        REFEREE: cv2.countNonZero(yellow) / total,
        GOALKEEPER: cv2.countNonZero(green) / total,
    }
    # 緑は芝の写り込みでも増えるので、十分に支配的なときだけGKにする
    if ratios[GOALKEEPER] < GK_GREEN_RATIO:
        ratios.pop(GOALKEEPER)
    return max(ratios, key=ratios.get)


def load_homography(input_path):
    """キーポイント指定セルで保存した 画像→ピッチ のホモグラフィと設定を読み込む"""
    json_path = os.path.splitext(input_path)[0] + "_keypoints.json"
    if not os.path.exists(json_path):
        print(f"  警告: {os.path.basename(json_path)} がないため2D座標は出力しません。")
        return None
    with open(json_path, encoding="utf-8") as f:
        kp = json.load(f)
    if "goal_on_screen" not in kp:
        print("  警告: 古い形式のキーポイントです。ピッカーセルで取り直してください（ゴールは画面右として扱います）。")
    return {"H": np.array(kp["H_img2pitch"], dtype=np.float64),
            "ref_frame": kp.get("reference_frame", 0),
            "goal_on_screen": kp.get("goal_on_screen", "right"),
            "pitch_width_m": kp.get("pitch_width_m", 68.0)}


class CameraMotionEstimator:
    """ORB特徴点のフレーム間マッチングで、現在フレーム→前フレームのホモグラフィを推定する"""

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp, self.prev_des = None, None

    def update(self, frame, moving_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 選手やボールなど動く物体の上の特徴点はカメラの動きと無関係なので除外する
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in moving_boxes.astype(int):
            mask[max(y1, 0):y2, max(x1, 0):x2] = 0
        kp, des = self.orb.detectAndCompute(gray, mask)

        H_cur2prev = np.eye(3)
        if self.prev_des is not None and des is not None and len(kp) >= 10:
            matches = self.matcher.match(des, self.prev_des)
            if len(matches) >= 15:
                src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([self.prev_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is not None:
                    H_cur2prev = H
        self.prev_kp, self.prev_des = kp, des
        return H_cur2prev


def occlusion_flags(xyxy):
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
    flags = (ratio > OCCLUSION_RATIO).any(axis=1)
    return flags


class BallTracker:
    """ボール専用の追跡。ボールは1つしかないので、IDを振るのではなく
    等速モデルのカルマンフィルタで次の位置を予測し、予測に一番近い候補だけを採用する。"""

    def __init__(self):
        self.active = False
        self.misses = 0
        self.size = 10.0
        self.trail = deque(maxlen=BALL_TRAIL)
        self.recent = deque(maxlen=BALL_FIT_WINDOW)  # 直近の実測 (frame, x, y)
        self.frame = 0

    def _init_kf(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # キックで急加速するので速度のプロセスノイズを大きめにする
        kf.processNoiseCov = np.diag([1, 1, 30, 30]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        kf.errorCovPost = np.diag([10, 10, 400, 400]).astype(np.float32)
        self.kf = kf

    def predict(self, frame_idx):
        """このフレームの予測位置。直近の実測が4点以上あれば弾道（2次式）で外挿する。追跡中でなければ None"""
        self.frame = frame_idx
        if not self.active:
            return None
        p = self.kf.predict()
        pred = (float(p[0, 0]), float(p[1, 0]))
        if len(self.recent) >= 4:
            q = quadratic_predict(list(self.recent), frame_idx)
            if q is not None:
                pred = q
        return pred

    def gate(self):
        return BALL_GATE_PX * (1 + 0.5 * self.misses / SLOWDOWN)

    def update(self, pred, cands):
        """cands: [(cx, cy, size, conf), ...]。戻り値: (x, y, size, 予測のみか) または None"""
        if not self.active:
            good = [c for c in cands if c[3] >= BALL_INIT_CONF]
            if not good:
                return None
            cx, cy, size, _ = max(good, key=lambda c: c[3])
            self._init_kf(cx, cy)
            self.active, self.misses, self.size = True, 0, size
            self.trail.clear()
            self.trail.append((cx, cy))
            self.recent.clear()
            self.recent.append((self.frame, cx, cy))
            return cx, cy, size, False

        best = None
        if cands:
            d = [np.hypot(c[0] - pred[0], c[1] - pred[1]) for c in cands]
            k = int(np.argmin(d))
            if d[k] <= self.gate():
                best = cands[k]
        if best is not None:
            cx, cy, size, _ = best
            self.kf.correct(np.array([[cx], [cy]], np.float32))
            self.misses, self.size = 0, 0.7 * self.size + 0.3 * size
            self.trail.append((cx, cy))
            self.recent.append((self.frame, cx, cy))
            return cx, cy, self.size, False

        self.misses += 1
        if self.misses > BALL_MAX_MISSES * SLOWDOWN:
            self.active = False
            self.trail.clear()
            self.recent.clear()
            return None
        self.trail.append(pred)
        return pred[0], pred[1], self.size, True


def quadratic_predict(pts, f):
    """(frame, x, y) の列に x(t), y(t) の2次式（点が少なければ1次式）を当てはめ、フレーム f の位置を返す"""
    if len(pts) < 2:
        return None
    a = np.asarray(pts, dtype=np.float64)
    t = a[:, 0] - a[-1, 0]
    deg = 2 if len(a) >= 4 and np.ptp(t) >= 3 else 1
    try:
        cx = np.polyfit(t, a[:, 1], deg)
        cy = np.polyfit(t, a[:, 2], deg)
    except Exception:
        return None
    tf = f - a[-1, 0]
    return float(np.polyval(cx, tf)), float(np.polyval(cy, tf))


def fit_ball_trajectory(obs):
    """ボールの実測 [(frame, x, y, size), ...] を弾道ごとに区切って2次式で当てはめ直す。
    - 予測から外れた1点だけの検出は誤検出として捨てる
    - 続けて外れたらキックかバウンドとみなし、そこで軌道を区切る
    - 軌道の中の欠けたフレームは2次式で、軌道と軌道の間の短い欠けは直線で埋める
    戻り値: {frame: (x, y, size, kind)}  kind は "fit" / "gap" / "raw" """
    obs = sorted(obs)
    max_gap = BALL_SEG_MAX_GAP * SLOWDOWN
    segments, cur = [], []
    i = 0
    while i < len(obs):
        f, x, y, s = obs[i]
        if cur and f - cur[-1][0] > max_gap:
            segments.append(cur)
            cur = []
        if len(cur) >= 3:
            win = [(p[0], p[1], p[2]) for p in cur[-BALL_FIT_WINDOW:]]
            gap = (f - cur[-1][0]) / SLOWDOWN
            tol = BALL_FIT_TOL_PX * (1 + 0.5 * max(gap - 1, 0))
            px, py = quadratic_predict(win, f)
            if np.hypot(x - px, y - py) > tol:
                nxt = obs[i + 1] if i + 1 < len(obs) else None
                if nxt is not None:
                    qx, qy = quadratic_predict(win, nxt[0])
                    if np.hypot(nxt[1] - qx, nxt[2] - qy) <= tol * 1.5:
                        i += 1  # この1点だけ外れている → 誤検出として捨てる
                        continue
                segments.append(cur)   # 軌道が変わった（キック・バウンド）
                cur = [cur[-1]]        # 接触点は前後の軌道で共有する
        cur.append(obs[i])
        i += 1
    if cur:
        segments.append(cur)

    out = {}
    for seg in segments:
        a = np.asarray(seg, dtype=np.float64)
        f0, f1 = int(a[0, 0]), int(a[-1, 0])
        size = float(np.median(a[:, 3]))
        if len(a) >= 4:
            t = a[:, 0]
            keep = np.ones(len(a), dtype=bool)
            for _ in range(2):  # 外れ点を除いて当てはめ直す
                cx = np.polyfit(t[keep] - f0, a[keep, 1], 2)
                cy = np.polyfit(t[keep] - f0, a[keep, 2], 2)
                res = np.hypot(np.polyval(cx, t - f0) - a[:, 1], np.polyval(cy, t - f0) - a[:, 2])
                new_keep = res <= BALL_FIT_TOL_PX * 2
                if new_keep.sum() < 4 or (new_keep == keep).all():
                    break
                keep = new_keep
            for f in range(f0, f1 + 1):
                out[f] = (float(np.polyval(cx, f - f0)), float(np.polyval(cy, f - f0)), size, "fit")
        else:
            for f, x, y, s in seg:
                out[int(f)] = (float(x), float(y), float(s), "raw")

    # 軌道と軌道の間の短い欠けを直線で埋める
    frames = sorted(out)
    for fa, fb in zip(frames[:-1], frames[1:]):
        if 1 < fb - fa <= max_gap:
            xa, ya, sa, _ = out[fa]
            xb, yb, sb, _ = out[fb]
            for f in range(fa + 1, fb):
                r = (f - fa) / (fb - fa)
                out[f] = (xa + (xb - xa) * r, ya + (yb - ya) * r, sa + (sb - sa) * r, "gap")
    return out


def ball_candidates(boxes, confs, offset=(0, 0)):
    ox, oy = offset
    return [((b[0] + b[2]) / 2 + ox, (b[1] + b[3]) / 2 + oy, max(b[2] - b[0], b[3] - b[1]), float(c))
            for b, c in zip(boxes, confs)]


def search_ball_roi(model, frame, center):
    """予測位置の周りを切り出して拡大推論する（小さく速いボールの見逃し対策）"""
    h, w = frame.shape[:2]
    half = BALL_ROI // 2
    x0 = int(min(max(center[0] - half, 0), max(w - BALL_ROI, 0)))
    y0 = int(min(max(center[1] - half, 0), max(h - BALL_ROI, 0)))
    crop = frame[y0:y0 + BALL_ROI, x0:x0 + BALL_ROI]
    r = model(crop, imgsz=640, conf=BALL_CONF, classes=[32], verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    return ball_candidates(boxes, confs, offset=(x0, y0))


def draw_ball(img, ball, trail):
    if len(trail) > 1:
        cv2.polylines(img, [np.int32(trail)], False, BALL_BGR, 2, cv2.LINE_AA)
    if ball is None:
        return
    x, y, size, kind = ball
    r = int(max(6, size * 0.75))
    # 実測に当てはめた位置は太い円、検出の間を補った位置は細い円
    cv2.circle(img, (int(x), int(y)), r, BALL_BGR, 3 if kind in ("fit", "raw") else 1, cv2.LINE_AA)


class FFmpegWriter:
    """OpenCVのmp4vより高画質なH.264で書き出す（Drive上でもプレビュー再生できる）"""

    def __init__(self, path, width, height, fps, crf=OUTPUT_CRF):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def write_frame(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.stdin.close()
        self.proc.wait()


# ---- 白線によるホモグラフィの再推定（ドリフト補正）----
LINE_DIST_CAP = 25.0   # 白線からの距離の上限（px）。これより遠い点は外れ扱い
INLIER_PX = 3.0        # 白線からこの距離以内なら「ラインに乗っている」とみなす


def line_model_points(goal_on_screen="right", pitch_width_m=68.0, step=0.5):
    """ゴール側のライン（ゴールライン・PA・ゴールエリア・PAアーク・タッチライン）上の点と重み"""
    c = (pitch_width_m / 2) / 0.9144
    segs = [
        ([(120, 40 - c), (120, 40 + c)], 1.0),                 # ゴールライン
        ([(120, 18), (102, 18), (102, 62), (120, 62)], 1.0),   # PA
        ([(120, 30), (114, 30), (114, 50), (120, 50)], 1.0),   # ゴールエリア
        ([(90, 40 - c), (120, 40 - c)], 0.5),                  # タッチライン（幅の誤差があるので重み小）
        ([(90, 40 + c), (120, 40 + c)], 0.5),
    ]
    t = np.linspace(np.pi - 0.927, np.pi + 0.927, 20)
    segs.append((list(zip(108 + 10 * np.cos(t), 40 + 10 * np.sin(t))), 1.0))  # PAアーク

    pts, wts = [], []
    for seg, w in segs:
        for (x0, y0), (x1, y1) in zip(seg[:-1], seg[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
            for s in np.linspace(0, 1, n, endpoint=False):
                pts.append((x0 + (x1 - x0) * s, y0 + (y1 - y0) * s))
                wts.append(w)
    pts = np.array(pts, np.float64)
    if goal_on_screen == "left":
        pts[:, 1] = 80 - pts[:, 1]
    return pts, np.array(wts)


def white_line_mask(frame, pitch_mask, person_boxes):
    """芝の上の細く明るい線（白線）を抽出する。白ユニフォームと誤認しないよう選手の箱は消す"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = max(9, (frame.shape[1] // 80) | 1)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = ((tophat > 25) & (hsv[..., 1] < 80) & (pitch_mask > 0)).astype(np.uint8) * 255
    for x1, y1, x2, y2 in np.asarray(person_boxes).astype(int):
        mask[max(y1 - 3, 0):y2 + 3, max(x1 - 3, 0):x2 + 3] = 0
    return mask


class PitchLineRefiner:
    """推定中のホモグラフィで投影したピッチのラインを、映像の白線に合わせ込んで補正する。
    パラメータはPA4角の画像上の位置のずれ（8変数）で、白線までの距離を最小化する。"""

    def __init__(self, goal_on_screen="right", pitch_width_m=68.0):
        self.pts, self.wts = line_model_points(goal_on_screen, pitch_width_m)
        ctrl = np.float32([[102, 18], [102, 62], [120, 18], [120, 62]])
        if goal_on_screen == "left":
            ctrl[:, 1] = 80 - ctrl[:, 1]
        self.ctrl = ctrl

    @staticmethod
    def _project(G, pts):
        hom = np.c_[pts, np.ones(len(pts))] @ G.T
        valid = hom[:, 2] > 1e-6
        uv = hom[:, :2] / np.where(valid, hom[:, 2], 1)[:, None]
        return uv, valid

    def _sample(self, dist, G):
        h, w = dist.shape
        uv, valid = self._project(G, self.pts)
        inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        d = np.full(len(self.pts), LINE_DIST_CAP)
        if inside.any():
            d[inside] = map_coordinates(dist, [uv[inside, 1], uv[inside, 0]], order=1)
        return d, inside

    def score(self, dist, H_img2pitch):
        """画面内に投影されたライン点のうち、白線に乗っている割合と画面内の点数"""
        d, inside = self._sample(dist, np.linalg.inv(H_img2pitch))
        n = int(inside.sum())
        if n == 0:
            return 0.0, 0
        return float((d[inside] <= INLIER_PX).mean()), n

    def distance_map(self, frame, pitch_mask, person_boxes):
        mask = white_line_mask(frame, pitch_mask, person_boxes)
        if cv2.countNonZero(mask) < 200:
            return None
        dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        return np.minimum(dist, LINE_DIST_CAP).astype(np.float64)

    def refine(self, dist, H_img2pitch):
        """戻り値: (補正後のH, 補正前スコア, 補正後スコア, 画面内の点数) / 失敗時 None"""
        G0 = np.linalg.inv(H_img2pitch)
        c0, valid = self._project(G0, self.ctrl.astype(np.float64))
        if not valid.all():
            return None
        c0 = c0.astype(np.float32)

        def G_of(p):
            return cv2.getPerspectiveTransform(self.ctrl, (c0 + p.reshape(4, 2)).astype(np.float32)).astype(np.float64)

        sw = np.sqrt(self.wts)

        def resid(p):
            d, _ = self._sample(dist, G_of(p))
            return d * sw

        def jac(p):
            # 距離マップは区分線形なので、1px幅の中心差分で勾配をとる
            J = np.empty((len(self.pts), 8))
            for i in range(8):
                e = np.zeros(8)
                e[i] = 1.0
                J[:, i] = (resid(p + e) - resid(p - e)) / 2.0
            return J

        try:
            res = least_squares(resid, np.zeros(8), jac=jac, loss="soft_l1", f_scale=INLIER_PX,
                                x_scale=5.0, max_nfev=100)
        except Exception:
            return None
        H1 = np.linalg.inv(G_of(res.x))
        before, _ = self.score(dist, H_img2pitch)
        after, n = self.score(dist, H1)
        return H1, before, after, n


def is_scene_cut(prev_hist, frame, threshold=0.5):
    """色ヒストグラムの急変でカット（リプレイ・別カメラへの切り替え）を検出する"""
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < threshold
    return cut, hist


def process_video(input_path, output_path):
    print(f"\n処理開始: {os.path.basename(input_path)}")
    model = YOLO(MODEL_NAME)
    video_info = sv.VideoInfo.from_video_path(video_path=input_path)
    fps = video_info.fps or 30          # ファイル上のfps
    real_fps = fps * SLOWDOWN           # 実時間1秒あたりのフレーム数
    # 隠れて見えなくなった選手のIDを実時間で2秒間保持する
    tracker = ByteTrackTracker(frame_rate=real_fps, lost_track_buffer=int(real_fps * 2))
    ball_tracker = BallTracker()

    # トラッカーIDごとのクラス履歴（フレームごとの色判定のチラつきを多数決で抑える）
    class_history = defaultdict(lambda: deque(maxlen=VOTE_WINDOW * SLOWDOWN))

    kp = load_homography(input_path)
    H_ref = kp["H"] if kp else None
    ref_frame = kp["ref_frame"] if kp else 0
    motion = CameraMotionEstimator() if (H_ref is not None and CAMERA_MOVES) else None
    refiner = PitchLineRefiner(kp["goal_on_screen"], kp["pitch_width_m"]) if (kp and CAMERA_MOVES) else None
    H_img2pitch, H_last_good = None, None
    h_ok, h_score = False, 0.0
    prev_hist = None
    n_refined, n_lost_frames = 0, 0

    # 1パス目の結果をフレームごとに保存し、ボールの弾道フィット後に2パス目で描画・記録する
    per_frame = []      # (persons, tracker_ids, H_for_2d, h_score)
    ball_obs = []       # ボールの実測 (frame, x, y, size)

    print(f"  1/2 推論とトラッキング（{video_info.total_frames} フレーム, 実時間 {video_info.total_frames / real_fps:.1f} 秒）...")
    for frame_idx, frame in enumerate(sv.get_video_frames_generator(input_path)):
        pitch_mask = get_pitch_mask(frame)
        h_img, w_img = pitch_mask.shape[:2]

        results = model(frame, imgsz=IMGSZ, conf=min(PERSON_CONF, BALL_CONF), iou=NMS_IOU,
                        classes=[0, 32], verbose=False)[0]
        raw = sv.Detections.from_ultralytics(results)

        # ---- 画像→ピッチのホモグラフィを現在フレーム用に更新する ----
        # （キーポイントを取ったフレームより前は2D座標を出さない）
        if H_ref is not None and motion is None:
            H_img2pitch, h_ok, h_score = H_ref, True, 1.0
        elif H_ref is not None:
            H_cur2prev = motion.update(frame, raw.xyxy)
            cut, prev_hist = is_scene_cut(prev_hist, frame)
            force = False
            if frame_idx == ref_frame:
                H_img2pitch, H_last_good, h_ok, force = H_ref.copy(), H_ref.copy(), True, True
            elif frame_idx > ref_frame:
                H_img2pitch = H_img2pitch @ H_cur2prev
                if cut:  # リプレイや別カメラへの切り替え。動き推定は使えないので見失い扱い
                    h_ok, force = False, True

            due = frame_idx >= ref_frame and (frame_idx - ref_frame) % REFINE_EVERY == 0
            if frame_idx >= ref_frame and (force or due or not h_ok):
                person_boxes = raw.xyxy[raw.class_id == 0]
                dist = refiner.distance_map(frame, pitch_mask, person_boxes)
                best = None
                if dist is not None:
                    # 見失っているときは、最後に合っていたHからも試す
                    inits = [H_img2pitch] if h_ok else [H_img2pitch, H_last_good]
                    for H0 in inits:
                        out = refiner.refine(dist, H0)
                        if out is not None and out[3] >= MIN_LINE_POINTS and (best is None or out[2] > best[2]):
                            best = out
                if best is not None and best[2] >= REFINE_ACCEPT and best[2] >= best[1]:
                    H_img2pitch, H_last_good = best[0], best[0].copy()
                    h_ok, h_score = True, best[2]
                    n_refined += 1
                elif best is not None and best[1] < LOST_SCORE:
                    h_ok = False
            if frame_idx >= ref_frame and not h_ok:
                n_lost_frames += 1
        H_for_2d = H_img2pitch.copy() if (h_ok and H_img2pitch is not None) else None

        # ---- 選手 ----
        persons = raw[(raw.class_id == 0) & (raw.confidence >= PERSON_CONF)]
        keep, team_ids = [], []
        for i, xyxy in enumerate(persons.xyxy):
            feet_x = min(max(int((xyxy[0] + xyxy[2]) / 2), 0), w_img - 1)
            feet_y = min(max(int(xyxy[3]), 0), h_img - 1)
            if pitch_mask[feet_y, feet_x] == 0:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            team_id = classify_by_jersey_color(frame[max(y1, 0):y2, max(x1, 0):x2])
            if team_id == -1:
                continue
            keep.append(i)
            team_ids.append(team_id)

        persons = persons[np.array(keep, dtype=int)]
        persons.class_id = np.array(team_ids, dtype=int)
        persons.data["occluded"] = occlusion_flags(persons.xyxy)
        persons = tracker.update(persons)

        # 確定したトラッカーIDは直近フレームの多数決でクラス（=ボックス色）を決める
        # 他の選手と重なっているフレームの色判定は、隣の選手の色が混ざるので履歴に入れない
        tracker_ids = persons.tracker_id if persons.tracker_id is not None else np.full(len(persons), -1)
        occluded = persons.data.get("occluded", np.zeros(len(persons), dtype=bool))
        voted = persons.class_id.copy()
        for k, (cid, tid) in enumerate(zip(persons.class_id, tracker_ids)):
            if tid < 0:
                continue
            hist = class_history[int(tid)]
            if not occluded[k] or len(hist) == 0:
                hist.append(int(cid))
            voted[k] = Counter(hist).most_common(1)[0][0]
        persons.class_id = voted

        # 表示と記録はIDが確定した選手だけ（低信頼の検出は追跡の継続にだけ使う）
        confirmed = tracker_ids >= 0
        per_frame.append((persons.xyxy[confirmed].copy(), persons.class_id[confirmed].copy(),
                          tracker_ids[confirmed].copy(), H_for_2d, h_score))

        # ---- ボール（オンラインでは見失い対策の探索と候補選びだけ行う）----
        pred = ball_tracker.predict(frame_idx)
        balls = raw[raw.class_id == 32]
        cands = ball_candidates(balls.xyxy, balls.confidence)
        near = pred is not None and any(np.hypot(c[0] - pred[0], c[1] - pred[1]) <= ball_tracker.gate() for c in cands)
        if pred is not None and not near:
            cands += search_ball_roi(model, frame, pred)
        ball = ball_tracker.update(pred, cands)
        if ball is not None and not ball[3]:
            ball_obs.append((frame_idx, ball[0], ball[1], ball[2]))

    if refiner is not None:
        print(f"  白線による補正: {n_refined} 回採用 / 位置合わせを見失っていたフレーム: {n_lost_frames}")

    # ---- ボールの弾道フィット ----
    ball_fit = fit_ball_trajectory(ball_obs)
    n_fit = sum(1 for v in ball_fit.values() if v[3] == "fit")
    print(f"  ボール: 実測 {len(ball_obs)} フレーム → 弾道で補完後 {len(ball_fit)} フレーム（うち2次式フィット {n_fit}）")

    # ---- 2パス目: 描画（1倍速に戻す）と2D座標の記録 ----
    print("  2/2 描画と2D座標の記録...")
    box_annotator = sv.BoxAnnotator(color=BOX_PALETTE, thickness=2, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=BOX_PALETTE, text_color=sv.Color.BLACK,
                                        text_scale=0.5, color_lookup=sv.ColorLookup.CLASS)
    records = []
    with FFmpegWriter(output_path, video_info.width, video_info.height, fps) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(input_path)):
            if frame_idx >= len(per_frame):
                break
            xyxy, cls, tids, H_for_2d, score = per_frame[frame_idx]
            ball = ball_fit.get(frame_idx)
            t_sec = frame_idx / real_fps

            if H_for_2d is not None:
                if len(xyxy) > 0:
                    feet = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]].astype(np.float32)
                    pxy = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H_for_2d).reshape(-1, 2)
                    for (px, py), (ix, iy), cid, tid in zip(pxy, feet, cls, tids):
                        records.append({"frame": frame_idx, "time_s": t_sec, "tracker_id": int(tid),
                                        "class_id": int(cid), "image_x": float(ix), "image_y": float(iy),
                                        "pitch_x": float(px), "pitch_y": float(py), "ball_kind": "",
                                        "h_score": score})
                if ball is not None:
                    bx, by, _, kind = ball
                    px, py = cv2.perspectiveTransform(np.float32([[[bx, by]]]), H_for_2d)[0, 0]
                    records.append({"frame": frame_idx, "time_s": t_sec, "tracker_id": -1, "class_id": BALL,
                                    "image_x": float(bx), "image_y": float(by),
                                    "pitch_x": float(px), "pitch_y": float(py), "ball_kind": kind,
                                    "h_score": score})

            if frame_idx % SLOWDOWN != 0:   # 出力はSLOWDOWNフレームごと → 1倍速
                continue
            annotated = frame.copy()
            if len(xyxy) > 0:
                dets = sv.Detections(xyxy=xyxy, class_id=cls, tracker_id=tids)
                annotated = box_annotator.annotate(scene=annotated, detections=dets)
                annotated = label_annotator.annotate(scene=annotated, detections=dets,
                                                     labels=[str(int(t)) for t in tids])
            trail = [ball_fit[f][:2] for f in range(max(0, frame_idx - BALL_TRAIL * SLOWDOWN), frame_idx + 1)
                     if f in ball_fit]
            draw_ball(annotated, ball, trail)
            if DEBUG_OVERLAY and refiner is not None and H_for_2d is not None:
                uv, valid = refiner._project(np.linalg.inv(H_for_2d), refiner.pts)
                h_img, w_img = annotated.shape[:2]
                for u, v in uv[valid].astype(int):
                    if 0 <= u < w_img and 0 <= v < h_img:
                        annotated[v, u] = (255, 0, 255)
            sink.write_frame(annotated)

    print(f"  完了！ {os.path.basename(output_path)} を保存しました（1倍速）。")

    base = os.path.splitext(input_path)[0]
    with open(base + "_meta.json", "w") as f:
        json.dump({"file_fps": fps, "slowdown": SLOWDOWN, "real_fps": real_fps}, f)
    if records:
        pd.DataFrame(records).to_csv(base + "_positions.csv", index=False)
        print(f"  2D座標: {os.path.basename(base)}_positions.csv を保存しました（{len(records)} 行）。")


def main():
    global BASE_DIR, VIDEO_FILES, MODEL_NAME, IMGSZ, SLOWDOWN, CAMERA_MOVES, DEBUG_OVERLAY
    ap = argparse.ArgumentParser(description="選手・ボールのトラッキングと2D座標の出力")
    ap.add_argument("videos", nargs="*", help="動画ファイル名（省略時は VIDEO_FILES）")
    ap.add_argument("--data-dir", default=BASE_DIR)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--slowdown", type=int, default=SLOWDOWN, help="入力が実時間の何倍にスロー化されているか")
    ap.add_argument("--fixed-camera", action="store_true", help="固定カメラ（動き推定と白線補正を行わない）")
    ap.add_argument("--debug-overlay", action="store_true", help="推定中のピッチのラインを重ねる")
    args = ap.parse_args()
    BASE_DIR = os.path.abspath(args.data_dir)
    if args.videos:
        VIDEO_FILES = args.videos
    MODEL_NAME, IMGSZ, SLOWDOWN = args.model, args.imgsz, args.slowdown
    CAMERA_MOVES, DEBUG_OVERLAY = not args.fixed_camera, args.debug_overlay

    for filename in VIDEO_FILES:
        input_path = os.path.join(BASE_DIR, filename)
        name, ext = os.path.splitext(filename)
        output_name = f"{name}_tracking.mp4"
        output_path = os.path.join(BASE_DIR, output_name)

        if not os.path.exists(input_path):
            print(f"\nエラー: {input_path} が見つかりません。")
            continue

        process_video(input_path, output_path)


if __name__ == "__main__":
    main()
