import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import sys
import json
import subprocess
import pandas as pd
from collections import defaultdict, deque, Counter
from trackers import ByteTrackTracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitch
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
# 画面を BALL_TILES×BALL_TILES に分割し、各タイルを拡大してボールだけ探す（0で無効）
# 2なら推論が5回/フレームになり遅くなるが、小さいボールの検出率が上がる
BALL_TILES = 2
BALL_TILE_OVERLAP = 0.2

# ---- スロー化した入力 ----
# 入力動画が実時間の何倍に引き伸ばされているか（0.25倍速なら 4、通常の動画なら 1）
# 推論は全フレームで行い、出力動画は SLOWDOWN フレームごとに1枚書き出して1倍速に戻す
SLOWDOWN = 4

# ---- ボールの経路選択（全フレームの候補から、もっともらしい1本の経路を選ぶ）----
BALL_KEEP_TOPK = 5          # フレームごとに残す候補数（信頼度順）
BALL_MAX_SPEED_PX = 60      # 実時間1フレームあたりの最大移動量（px）。これより速い移動は同じボールとみなさない
BALL_LINK_GAP = 8           # 候補同士をつなげる最大の間隔（実時間換算のフレーム数）
BALL_LINK_BONUS = 0.15      # 動きとして自然につながる候補への加点（低信頼でも軌道上なら採用されやすくする）
BALL_CONF_OFFSET = 0.15     # 信頼度からこれを引いた値が候補の得点（つながりのない低信頼の候補は採用されない）
BALL_JUMP_PENALTY = 0.5     # つながらない位置へ飛ぶ（見失い→再発見）ときの減点
BALL_STATIC_PX = 2.0        # 実時間1フレームあたりこれ未満しか動かない候補のつながりは加点しない（ペナルティスポット等の誤検出対策）
BALL_STATIC_YD = 0.05       # 同上をピッチ座標で判定する場合（カメラが動いても止まっている物を見分けられる）

# ---- ピッチ上の白い点（ペナルティマーク・センターマーク）の誤検出除去 ----
# 2D座標がわかるフレームでは、ボール候補をピッチ座標に変換し、マークの近くにある候補を捨てる
PITCH_SPOTS = [(108, 40), (12, 40), (60, 40)]   # ペナルティマーク（両側）とセンターマーク（StatsBomb座標）
BALL_SPOT_RADIUS_YD = 1.0   # マークからこの距離以内の候補を捨てる
BALL_SPOT_KEEP_CONF = 0.6   # ただし信頼度がこれ以上なら残す（PK でマーク上に置かれた本物のボール用）

# ---- 出力動画の速度 ----
# "input": 入力と同じ速度（全フレームを書き出す）
# "realtime": SLOWDOWN フレームごとに1枚だけ書き出して実時間に戻す（入力全体が一定倍率でスロー化されている場合のみ正しい）
OUTPUT_SPEED = "input"

# ---- ボールの弾道フィット（全フレーム処理後にまとめて実行）----
BALL_FIT_TOL_PX = 6.0   # 弾道（2次式）の予測からこれ以上外れたら、キック・バウンド・誤検出のどれか
BALL_FIT_WINDOW = 12    # 予測に使う直近の検出数
BALL_SEG_MAX_GAP = 8    # 検出の間がこのフレーム数（実時間換算）を超えたら別の軌道として扱う

# ---- 出力 ----
OUTPUT_CRF = 18         # H.264の画質（小さいほど高画質。18でほぼ劣化なし）

# 内部クラスID（ラベルには出さない。ボックス色の決定にだけ使う）
TEAM_WHITE, TEAM_BLUE, REFEREE, GOALKEEPER, BALL, UNKNOWN = 0, 1, 2, 3, 4, 5
TEAM_NAMES = {"white": TEAM_WHITE, "blue": TEAM_BLUE, "referee": REFEREE, "gk": GOALKEEPER}

# クラスID順のボックス色（RGB）
BOX_PALETTE = sv.ColorPalette(colors=[
    sv.Color(r=255, g=255, b=255),  # 0: 白チーム
    sv.Color(r=30,  g=90,  b=255),  # 1: 青チーム
    sv.Color(r=255, g=215, b=0),    # 2: 審判（黄）
    sv.Color(r=230, g=30,  b=30),   # 3: GK（赤）
    sv.Color(r=255, g=140, b=0),    # 4: ボール（オレンジ）
    sv.Color(r=150, g=150, b=150),  # 5: 判定できない（灰）
])
BALL_BGR = (0, 140, 255)

# ---- ユニフォームの色（HSV の範囲。OpenCV の H は 0〜180）----
# 試合ごとにここを書き換える。1つのクラスに複数の範囲を書ける（赤は H が 0 付近と 180 付近に分かれる）
JERSEY_COLORS = {
    TEAM_WHITE: [((0, 0, 150), (180, 45, 255))],                                # 白（日陰でも拾えるよう明度の下限を低めに）
    TEAM_BLUE: [((95, 60, 40), (130, 255, 255))],                               # 青（紺も含む）
    REFEREE: [((18, 80, 80), (34, 255, 255))],                                  # 審判: 黄
    GOALKEEPER: [((0, 110, 60), (10, 255, 255)), ((165, 110, 60), (180, 255, 255))],  # GK: 赤（肌と区別するため彩度の下限を高めに）
}
JERSEY_MIN_RATIO = 0.08   # どの色もこの割合に届かなければ、そのフレームの色判定はしない（票に数えない）
# ---- 途切れたトラックのつなぎ直し（全フレーム処理後）----
# 見失った選手が再検出されると別のIDになるので、途切れたトラックの最後の位置と速度から
# 「そのまま進んでいたらどこにいるか」を予測し、その近くで新しく始まったトラックを同じ選手としてつなぐ
LINK_MAX_GAP_S = 3.0      # 途切れてからこの秒数（実時間）以内に始まったトラックだけを候補にする
LINK_VEL_WINDOW_S = 0.5   # 速度の推定に使う、途切れる直前の区間の長さ（秒）
LINK_R0_M = 1.5           # 予測位置からの許容距離（m、2D座標がある場合）。途切れていた時間に応じて広げる
LINK_R_PER_S_M = 2.0      # 1秒あたりに広げる許容距離（m）。向きや速さが変わった分の余裕
LINK_R0_BOX = 1.0         # 2D座標がない場合の許容距離（箱の高さの何倍か）
LINK_R_PER_S_BOX = 1.0

# ---- チームの決定（全フレーム処理後にトラック単位で行う）----
# チームは試合中に変わらないので、トラックIDごとに全フレームの色判定を集計して1つに決める
TEAM_MIN_VOTES = 8        # 自分の色判定だけで決めるのに必要な判定数（実時間換算のフレーム数）
TEAM_MIN_SHARE = 0.6      # 最多の色が全判定に占める割合がこれ以上なら確定
TEAM_INHERIT_GAP_S = 1.5  # 判定が足りないトラックは、この秒数以内に近くで途切れたトラックのチームを引き継ぐ
TEAM_INHERIT_DIST_YD = 5  # 引き継ぐ距離の上限（2D座標がある場合、ヤード）
TEAM_INHERIT_DIST_BOX = 1.5  # 引き継ぐ距離の上限（2D座標がない場合、箱の高さの何倍か）

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
    """胴体部分のユニフォーム色を JERSEY_COLORS のどれかに判定する。判定できなければ UNKNOWN。
    芝のピクセルは分母から除くので、細い選手や遠くの選手でも比率が薄まらない"""
    h, w = crop.shape[:2]
    # 胴体（上15〜50%、横中央50%）だけを見る。脚・芝・頭を避ける
    torso = crop[int(h * 0.15):int(h * 0.5), int(w * 0.25):int(w * 0.75)]
    if torso.size == 0:
        return UNKNOWN
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    total = torso.shape[0] * torso.shape[1] - cv2.countNonZero(grass)
    if total < 20:
        return UNKNOWN

    ratios = {}
    for cls, ranges in JERSEY_COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        ratios[cls] = cv2.countNonZero(mask & ~grass) / total
    best = max(ratios, key=ratios.get)
    return best if ratios[best] >= JERSEY_MIN_RATIO else UNKNOWN


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
        if len(self.recent) >= 4 and frame_idx - self.recent[-1][0] <= 3 * SLOWDOWN:
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
        if best is None:
            # 予測が外れている（キックで向きが変わった等）ときに、映っているボールを無視し続けないよう
            # 予測から遠くても信頼度の高い候補があれば、そこで追跡をやり直す
            strong = [c for c in cands if c[3] >= BALL_INIT_CONF]
            if strong:
                self.active = False
                return self.update(None, strong)
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


def filter_ball_candidates(cands, H_img2pitch):
    """ボール候補にピッチ座標を付け、ペナルティマーク・センターマークの上にある低信頼の候補を捨てる。
    戻り値: [(x, y, size, conf, pitch_x, pitch_y), ...]（2D座標がないフレームでは pitch は None）"""
    if not cands:
        return []
    if H_img2pitch is None:
        return [(*c, None, None) for c in cands]
    pts = np.float32([[c[0], c[1]] for c in cands]).reshape(-1, 1, 2)
    pxy = cv2.perspectiveTransform(pts, H_img2pitch).reshape(-1, 2)
    out = []
    for c, (px, py) in zip(cands, pxy):
        near_spot = any(np.hypot(px - sx, py - sy) <= BALL_SPOT_RADIUS_YD for sx, sy in PITCH_SPOTS)
        if near_spot and c[3] < BALL_SPOT_KEEP_CONF:
            continue
        out.append((*c, float(px), float(py)))
    return out


def select_ball_path(cands_per_frame):
    """全フレームのボール候補から、動きとして一番もっともらしい経路を動的計画法で選ぶ。
    オンライン追跡と違い、予測が外れても「その後に映っているボール」を取りこぼさない。
    - 候補の得点: 信頼度 - BALL_CONF_OFFSET
    - 前の候補から自然な速さでつながれば BALL_LINK_BONUS を加点（間隔が空くほど減点）
    - つながらない位置への移動は BALL_JUMP_PENALTY を払えば常に可能（見失い→再発見）
    cands_per_frame: {frame: [(x, y, size, conf, pitch_x, pitch_y), ...]}
    戻り値: 選ばれた候補 [(frame, x, y, size), ...]"""
    max_gap = BALL_LINK_GAP * SLOWDOWN
    nodes = []
    for f in sorted(cands_per_frame):
        for c in sorted(cands_per_frame[f], key=lambda c: -c[3])[:BALL_KEEP_TOPK]:
            nodes.append((f, c))
    if not nodes:
        return []
    n = len(nodes)
    dp = np.full(n, -np.inf)
    prev = np.full(n, -1)
    best_before = []          # (frame, これまでの最高得点, そのノード)
    run_best, run_idx = -np.inf, -1
    i0 = 0                    # 前のフレームまでのノードの開始位置
    for j, (fj, cj) in enumerate(nodes):
        # 前のフレームまでの最高得点を更新（同じフレーム内のノードからは飛べない）
        while i0 < j and nodes[i0][0] < fj:
            if dp[i0] > run_best:
                run_best, run_idx = dp[i0], i0
            i0 += 1
        emit = cj[3] - BALL_CONF_OFFSET
        score, p = emit, -1                                   # ここから新しく始める
        if run_idx >= 0 and run_best - BALL_JUMP_PENALTY + emit > score:
            score, p = run_best - BALL_JUMP_PENALTY + emit, run_idx   # 見失い→再発見
        k = j - 1
        while k >= 0 and fj - nodes[k][0] <= max_gap:
            fk, ck = nodes[k]
            if fk < fj:
                gap_real = (fj - fk) / SLOWDOWN
                dist = np.hypot(cj[0] - ck[0], cj[1] - ck[1])
                if dist <= BALL_MAX_SPEED_PX * max(gap_real, 1 / SLOWDOWN):
                    speed = dist / max(gap_real, 1e-6)
                    # 止まっている物（白いソックス、ピッチ上の白い汚れ等）が毎フレーム続けて検出されても得をしないよう、
                    # ほぼ動いていないつながりには加点しない。2D座標があればピッチ上で判定する
                    # （画像上だとカメラのパンで止まっている物も動いて見えるため）
                    if cj[4] is not None and ck[4] is not None:
                        moving = min(1.0, np.hypot(cj[4] - ck[4], cj[5] - ck[5]) / max(gap_real, 1e-6) / BALL_STATIC_YD)
                    else:
                        moving = min(1.0, speed / BALL_STATIC_PX)
                    link = BALL_LINK_BONUS * moving - 0.02 * max(gap_real - 1, 0) - 0.05 * speed / BALL_MAX_SPEED_PX
                    s = dp[k] + emit + link
                    if s > score:
                        score, p = s, k
            k -= 1
        dp[j], prev[j] = score, p
    # 最高得点のノードから逆にたどる
    j = int(np.argmax(dp))
    path = []
    while j >= 0:
        f, c = nodes[j]
        path.append((f, c[0], c[1], c[2]))
        j = prev[j]
    return path[::-1]


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


def detect_ball_tiled(model, frame):
    """画面をタイルに分割して拡大推論し、ボールの候補を返す（遠く小さいボールの見逃し対策）"""
    if BALL_TILES < 2:
        return []
    h, w = frame.shape[:2]
    th, tw = int(h / BALL_TILES * (1 + BALL_TILE_OVERLAP)), int(w / BALL_TILES * (1 + BALL_TILE_OVERLAP))
    ys = np.linspace(0, h - th, BALL_TILES).astype(int)
    xs = np.linspace(0, w - tw, BALL_TILES).astype(int)
    tiles = [(x0, y0, frame[y0:y0 + th, x0:x0 + tw]) for y0 in ys for x0 in xs]
    results = model([t[2] for t in tiles], imgsz=IMGSZ, conf=BALL_CONF, classes=[32], verbose=False)
    cands = []
    for (x0, y0, _), r in zip(tiles, results):
        cands += ball_candidates(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), offset=(x0, y0))
    return cands


def merge_candidates(cands, dist_px=8):
    """全体推論・タイル推論・切り出し推論で重複した候補を、信頼度の高い方に1つにまとめる"""
    out = []
    for c in sorted(cands, key=lambda c: -c[3]):
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) > dist_px for o in out):
            out.append(c)
    return out


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


def link_tracklets(track_pts, team_votes, real_fps):
    """途切れたトラックと、その後に始まったトラックを同じ選手としてつなぐ。
    track_pts: {tid: [(frame, img_x, img_y, pitch_x, pitch_y, box_h), ...]}（pitch は無ければ None）
    戻り値: {古いID: つないだ先のID（チェーンの最初のID）}"""
    from scipy.optimize import linear_sum_assignment

    max_gap = LINK_MAX_GAP_S * real_fps
    vel_n = max(2, int(LINK_VEL_WINDOW_S * real_fps))

    def team_of(tid):
        c = team_votes.get(tid)
        if not c or sum(c.values()) < TEAM_MIN_VOTES * SLOWDOWN:
            return None
        best, cnt = c.most_common(1)[0]
        return best if cnt / sum(c.values()) >= TEAM_MIN_SHARE else None

    def endpoint(pts, last):
        """トラックの端の位置と速度（1フレームあたり）。2D座標が揃っていればピッチ座標、なければ画像座標"""
        seg = pts[-vel_n:] if last else pts[:vel_n]
        use_pitch = all(p[3] is not None for p in seg)
        a = np.array([[p[0], p[3], p[4]] if use_pitch else [p[0], p[1], p[2]] for p in seg], dtype=np.float64)
        pos = a[-1, 1:] if last else a[0, 1:]
        vel = np.zeros(2)
        if len(a) >= 3 and np.ptp(a[:, 0]) > 0:
            vel = np.array([np.polyfit(a[:, 0], a[:, 1], 1)[0], np.polyfit(a[:, 0], a[:, 2], 1)[0]])
        return pos, vel, use_pitch, float(np.median([p[5] for p in seg]))

    tids = sorted(track_pts, key=lambda t: track_pts[t][0][0])
    ends = {t: endpoint(track_pts[t], True) for t in tids}
    starts = {t: endpoint(track_pts[t], False) for t in tids}
    INF = 1e9
    cost = np.full((len(tids), len(tids)), INF)
    for i, a in enumerate(tids):
        fa = track_pts[a][-1][0]
        pa, va, ua, ha = ends[a]
        for j, b in enumerate(tids):
            fb = track_pts[b][0][0]
            gap = fb - fa
            if a == b or gap <= 0 or gap > max_gap:
                continue
            pb, vb, ub, hb = starts[b]
            if ua != ub:                     # 片方だけ2D座標がある組み合わせは比べられない
                continue
            ta, tb = team_of(a), team_of(b)
            if ta is not None and tb is not None and ta != tb:
                continue
            gap_s = gap / real_fps
            # 途切れた方は前へ、始まった方は後ろへ、それぞれの速度で外挿して、真ん中の時刻で比べる
            mid = gap / 2
            d = np.linalg.norm((pa + va * mid) - (pb - vb * mid))
            if ua:
                r = LINK_R0_M + LINK_R_PER_S_M * gap_s
            else:
                r = (LINK_R0_BOX + LINK_R_PER_S_BOX * gap_s) * max(ha, hb)
            if d <= r:
                cost[i, j] = d / r + 0.1 * gap_s   # 近いほど・途切れが短いほど優先
    rows, cols = linear_sum_assignment(np.where(cost >= INF, 1e6, cost))
    nxt = {tids[i]: tids[j] for i, j in zip(rows, cols) if cost[i, j] < INF}

    # チェーンをたどって、つながった全トラックを最初のIDにそろえる
    prev = {b: a for a, b in nxt.items()}
    mapping = {}
    for t in tids:
        root = t
        while root in prev:
            root = prev[root]
        if root != t:
            mapping[t] = root
    return mapping


def resolve_teams(votes, spans, real_fps):
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
        if n >= TEAM_MIN_VOTES * SLOWDOWN and cnt / n >= TEAM_MIN_SHARE:
            team[tid] = (best, "votes")

    max_gap = int(TEAM_INHERIT_GAP_S * real_fps)

    def distance(a, b):
        """a の終わり（または始まり）と b の位置の距離が閾値以内か、と距離を返す"""
        if a["pitch"] is not None and b["pitch"] is not None:
            d = float(np.hypot(*(np.subtract(a["pitch"], b["pitch"]))))
            return d, d <= TEAM_INHERIT_DIST_YD
        d = float(np.hypot(*(np.subtract(a["img"], b["img"]))))
        return d, d <= TEAM_INHERIT_DIST_BOX * max(a["h"], b["h"])

    for tid in sorted(spans, key=lambda k: spans[k]["f0"]):
        if tid in team:
            continue
        s = spans[tid]
        cands = []
        for oid, o in spans.items():
            if oid == tid or oid not in team:
                continue
            if 0 < s["f0"] - o["f1"] <= max_gap:        # 直前に途切れたトラック
                d, ok = distance(o["end"], s["start"])
            elif 0 < o["f0"] - s["f1"] <= max_gap:      # 直後に始まったトラック
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


def process_video(input_path, output_path):
    print(f"\n処理開始: {os.path.basename(input_path)}")
    model = YOLO(MODEL_NAME)
    video_info = sv.VideoInfo.from_video_path(video_path=input_path)
    fps = video_info.fps or 30          # ファイル上のfps
    real_fps = fps * SLOWDOWN           # 実時間1秒あたりのフレーム数
    # 隠れて見えなくなった選手のIDを実時間で2秒間保持する
    tracker = ByteTrackTracker(frame_rate=real_fps, lost_track_buffer=int(real_fps * 2))
    ball_tracker = BallTracker()

    # トラックIDごとの色判定の集計と、トラックの始まり・終わりの位置
    team_votes = defaultdict(Counter)
    spans = {}
    track_pts = defaultdict(list)   # トラックIDごとの足元の位置の履歴（つなぎ直しに使う）

    global PITCH_SPOTS
    base = os.path.splitext(input_path)[0]
    # register.py の結果（全フレームのホモグラフィ、メートル座標）があればそれを使う。
    # なければ従来の方式（pick_keypoints.py の1フレーム + 前向きの追従、StatsBomb座標）
    reg = None
    if os.path.exists(base + "_homography.npz"):
        reg = np.load(base + "_homography.npz")
        PITCH_SPOTS = pitch.spots(float(reg["pitch_length"]), float(reg["pitch_width"]))
        pitch_meta = {"units": "m", "length": float(reg["pitch_length"]), "width": float(reg["pitch_width"])}
        print(f"  位置合わせ: {os.path.basename(base)}_homography.npz（{int(reg['valid'].sum())}/{len(reg['valid'])} フレーム、メートル座標）")
        kp = None
    else:
        kp = load_homography(input_path)
        pitch_meta = {"units": "statsbomb"}
    H_ref = kp["H"] if kp else None
    ref_frame = kp["ref_frame"] if kp else 0
    motion = CameraMotionEstimator() if (H_ref is not None and CAMERA_MOVES) else None
    refiner = PitchLineRefiner(kp["goal_on_screen"], kp["pitch_width_m"]) if (kp and CAMERA_MOVES) else None
    H_img2pitch, H_last_good = None, None
    h_ok, h_score = False, 0.0
    prev_hist = None
    n_refined, n_lost_frames = 0, 0

    # 1パス目の結果をフレームごとに保存し、ボールの弾道フィット後に2パス目で描画・記録する
    per_frame = []      # (xyxy, tracker_ids, H_for_2d, h_score)
    ball_cands = {}     # フレームごとのボール候補 [(x, y, size, conf), ...]

    print(f"  1/2 推論とトラッキング（{video_info.total_frames} フレーム, 実時間 {video_info.total_frames / real_fps:.1f} 秒）...")
    for frame_idx, frame in enumerate(sv.get_video_frames_generator(input_path)):
        pitch_mask = get_pitch_mask(frame)
        h_img, w_img = pitch_mask.shape[:2]

        results = model(frame, imgsz=IMGSZ, conf=min(PERSON_CONF, BALL_CONF), iou=NMS_IOU,
                        classes=[0, 32], verbose=False)[0]
        raw = sv.Detections.from_ultralytics(results)

        # ---- 画像→ピッチのホモグラフィを現在フレーム用に更新する ----
        # （キーポイントを取ったフレームより前は2D座標を出さない）
        if reg is not None:
            ok_f = frame_idx < len(reg["valid"]) and bool(reg["valid"][frame_idx])
            H_img2pitch, h_ok = (reg["H"][frame_idx], True) if ok_f else (None, False)
            h_score = float(reg["score"][frame_idx]) if ok_f else 0.0
        elif H_ref is not None and motion is None:
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
            # 色が判定できないフレームでも人としては追跡を続ける（票には数えない）
            team_id = classify_by_jersey_color(frame[max(y1, 0):y2, max(x1, 0):x2])
            keep.append(i)
            team_ids.append(team_id)

        persons = persons[np.array(keep, dtype=int)]
        persons.class_id = np.array(team_ids, dtype=int)
        persons.data["occluded"] = occlusion_flags(persons.xyxy)
        persons = tracker.update(persons)

        # 色判定はトラックIDごとに集計だけしておき、チームは全フレーム処理後にトラック単位で決める
        # 他の選手と重なっているフレームの色判定は、隣の選手の色が混ざるので数えない
        tracker_ids = persons.tracker_id if persons.tracker_id is not None else np.full(len(persons), -1)
        occluded = persons.data.get("occluded", np.zeros(len(persons), dtype=bool))
        confirmed = tracker_ids >= 0
        xyxy_c, tids_c = persons.xyxy[confirmed].copy(), tracker_ids[confirmed].copy()
        for box, cid, tid, occ in zip(xyxy_c, persons.class_id[confirmed], tids_c, occluded[confirmed]):
            tid = int(tid)
            if not occ and int(cid) != UNKNOWN:
                team_votes[tid][int(cid)] += 1
            foot = ((box[0] + box[2]) / 2, box[3])
            foot_pitch = None
            if H_for_2d is not None:
                foot_pitch = tuple(cv2.perspectiveTransform(np.float32([[foot]]), H_for_2d)[0, 0])
            pos = {"img": foot, "pitch": foot_pitch, "h": float(box[3] - box[1])}
            if tid not in spans:
                spans[tid] = {"f0": frame_idx, "start": pos}
            spans[tid]["f1"], spans[tid]["end"] = frame_idx, pos
            track_pts[tid].append((frame_idx, foot[0], foot[1],
                                   foot_pitch[0] if foot_pitch else None, foot_pitch[1] if foot_pitch else None,
                                   float(box[3] - box[1])))

        # 表示と記録はIDが確定した選手だけ（低信頼の検出は追跡の継続にだけ使う）
        per_frame.append((xyxy_c, tids_c, H_for_2d, h_score))

        # ---- ボール（オンラインでは見失い対策の探索と候補選びだけ行う）----
        pred = ball_tracker.predict(frame_idx)
        balls = raw[raw.class_id == 32]
        cands = ball_candidates(balls.xyxy, balls.confidence) + detect_ball_tiled(model, frame)
        near = pred is not None and any(np.hypot(c[0] - pred[0], c[1] - pred[1]) <= ball_tracker.gate() for c in cands)
        if pred is not None and not near:
            cands += search_ball_roi(model, frame, pred)
        cands = filter_ball_candidates(merge_candidates(cands), H_for_2d)
        # オンライン追跡は、予測位置の切り出し探索を当てるためだけに使う。採否は全フレーム後に経路選択で決める
        ball_tracker.update(pred, [c[:4] for c in cands])
        if cands:
            ball_cands[frame_idx] = cands

    if refiner is not None:
        print(f"  白線による補正: {n_refined} 回採用 / 位置合わせを見失っていたフレーム: {n_lost_frames}")

    # ---- チームの決定 ----
    base = os.path.splitext(input_path)[0]
    link = link_tracklets(track_pts, team_votes, real_fps)
    if link:
        for old, new in link.items():
            team_votes[new].update(team_votes.pop(old, Counter()))
            so, sn = spans.pop(old), spans[new]
            if so["f0"] < sn["f0"]:
                sn["f0"], sn["start"] = so["f0"], so["start"]
            if so["f1"] > sn["f1"]:
                sn["f1"], sn["end"] = so["f1"], so["end"]
        per_frame = [(xyxy, np.array([link.get(int(t), int(t)) for t in tids]), H, s)
                     for xyxy, tids, H, s in per_frame]
    print(f"  つなぎ直し: {len(link)} 個のトラックを前のトラックにつなげました（{len(spans)} トラックに）")
    team_of = resolve_teams(team_votes, spans, real_fps)
    for tid, team in load_team_overrides(base).items():
        if tid in spans:
            team_of[tid] = (team, "manual")
    src_count = Counter(s for _, s in team_of.values())
    print(f"  チーム: {len(team_of)} トラック（色判定で確定 {src_count['votes']} / 引き継ぎ {src_count['inherited']} / "
          f"判定少 {src_count['weak']} / 不明 {src_count['unknown']} / 手動 {src_count['manual']}）")

    # ---- ボールの経路選択と弾道フィット ----
    ball_obs = select_ball_path(ball_cands)
    ball_fit = fit_ball_trajectory(ball_obs)
    n_fit = sum(1 for v in ball_fit.values() if v[3] == "fit")
    n_total = len(per_frame)
    print(f"  ボール: 実測 {len(ball_obs)}/{n_total} フレーム（{100 * len(ball_obs) / max(n_total, 1):.0f}%）"
          f" → 弾道で補完後 {len(ball_fit)} フレーム（{100 * len(ball_fit) / max(n_total, 1):.0f}%、うち2次式フィット {n_fit}）")

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
            xyxy, tids, H_for_2d, score = per_frame[frame_idx]
            cls = np.array([team_of.get(int(t), (UNKNOWN, "unknown"))[0] for t in tids], dtype=int)
            src = [team_of.get(int(t), (UNKNOWN, "unknown"))[1] for t in tids]
            ball = ball_fit.get(frame_idx)
            t_sec = frame_idx / real_fps

            if H_for_2d is not None:
                if len(xyxy) > 0:
                    feet = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]].astype(np.float32)
                    pxy = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H_for_2d).reshape(-1, 2)
                    for (px, py), (ix, iy), cid, tid, s in zip(pxy, feet, cls, tids, src):
                        records.append({"frame": frame_idx, "time_s": t_sec, "tracker_id": int(tid),
                                        "class_id": int(cid), "team_source": s,
                                        "image_x": float(ix), "image_y": float(iy),
                                        "pitch_x": float(px), "pitch_y": float(py), "ball_kind": "",
                                        "h_score": score})
                if ball is not None:
                    bx, by, _, kind = ball
                    px, py = cv2.perspectiveTransform(np.float32([[[bx, by]]]), H_for_2d)[0, 0]
                    records.append({"frame": frame_idx, "time_s": t_sec, "tracker_id": -1, "class_id": BALL,
                                    "team_source": "",
                                    "image_x": float(bx), "image_y": float(by),
                                    "pitch_x": float(px), "pitch_y": float(py), "ball_kind": kind,
                                    "h_score": score})

            if OUTPUT_SPEED == "realtime" and frame_idx % SLOWDOWN != 0:
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

    speed_note = "入力と同じ速度" if OUTPUT_SPEED == "input" else "実時間"
    print(f"  完了！ {os.path.basename(output_path)} を保存しました（{speed_note}）。")

    with open(base + "_meta.json", "w") as f:
        json.dump({"file_fps": fps, "slowdown": SLOWDOWN, "real_fps": real_fps,
                   "output_step": SLOWDOWN if OUTPUT_SPEED == "realtime" else 1, "pitch": pitch_meta}, f)
    tracks = [{"tracker_id": tid, "team": next(k for k, v in TEAM_NAMES.items() if v == team) if team in TEAM_NAMES.values() else "unknown",
               "source": s, "start_frame": spans[tid]["f0"], "end_frame": spans[tid]["f1"],
               "votes": dict(team_votes.get(tid, {}))} for tid, (team, s) in sorted(team_of.items())]
    pd.DataFrame(tracks).to_csv(base + "_tracks.csv", index=False)
    if records:
        pd.DataFrame(records).to_csv(base + "_positions.csv", index=False)
        print(f"  2D座標: {os.path.basename(base)}_positions.csv を保存しました（{len(records)} 行）。")


def main():
    global BASE_DIR, VIDEO_FILES, MODEL_NAME, IMGSZ, SLOWDOWN, CAMERA_MOVES, DEBUG_OVERLAY, BALL_TILES, OUTPUT_SPEED
    ap = argparse.ArgumentParser(description="選手・ボールのトラッキングと2D座標の出力")
    ap.add_argument("videos", nargs="*", help="動画ファイル名（省略時は VIDEO_FILES）")
    ap.add_argument("--data-dir", default=BASE_DIR)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--slowdown", type=int, default=SLOWDOWN, help="入力が実時間の何倍にスロー化されているか")
    ap.add_argument("--fixed-camera", action="store_true", help="固定カメラ（動き推定と白線補正を行わない）")
    ap.add_argument("--debug-overlay", action="store_true", help="推定中のピッチのラインを重ねる")
    ap.add_argument("--output-speed", choices=["input", "realtime"], default=OUTPUT_SPEED,
                    help="出力動画の速度（input: 入力と同じ / realtime: SLOWDOWN で割って実時間に戻す）")
    ap.add_argument("--kp-model", default=None,
                    help="ピッチのキーポイント検出モデル。指定すると、位置合わせ（register.py）を自動で実行してから追跡する")
    ap.add_argument("--kp-every", type=int, default=5, help="キーポイントを何フレームごとに推論するか")
    ap.add_argument("--re-register", action="store_true", help="位置合わせの結果があってもやり直す")
    ap.add_argument("--ball-tiles", type=int, default=BALL_TILES, help="ボール用のタイル分割数（0で無効、2で2×2）")
    args = ap.parse_args()
    BASE_DIR = os.path.abspath(args.data_dir)
    if args.videos:
        VIDEO_FILES = args.videos
    MODEL_NAME, IMGSZ, SLOWDOWN = args.model, args.imgsz, args.slowdown
    CAMERA_MOVES, DEBUG_OVERLAY = not args.fixed_camera, args.debug_overlay
    BALL_TILES = args.ball_tiles
    OUTPUT_SPEED = args.output_speed

    for filename in VIDEO_FILES:
        input_path = os.path.join(BASE_DIR, filename)
        name, ext = os.path.splitext(filename)
        output_name = f"{name}_tracking.mp4"
        output_path = os.path.join(BASE_DIR, output_name)

        if not os.path.exists(input_path):
            print(f"\nエラー: {input_path} が見つかりません。")
            continue

        npz = os.path.join(BASE_DIR, f"{name}_homography.npz")
        if args.kp_model and (args.re_register or not os.path.exists(npz)):
            import register
            print(f"\n位置合わせ: {filename}")
            register.run(filename, BASE_DIR, kp_model=args.kp_model, kp_every=args.kp_every)

        process_video(input_path, output_path)


if __name__ == "__main__":
    main()