"""ボールの検出・追跡・経路選択・弾道フィット

オンラインでは候補を集めるだけにして（予測位置の切り出し探索のために軽い追跡は行う）、
全フレームが終わってから、動的計画法で最ももっともらしい経路を選び、2次式（弾道）で欠けを埋める。
"""

from collections import deque

import cv2
import numpy as np

from .config import BALL_BGR


class BallTracker:
    """ボール専用の追跡。ボールは1つしかないので、IDを振るのではなく
    等速モデルのカルマンフィルタで次の位置を予測し、予測に一番近い候補だけを採用する。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self.misses = 0
        self.size = 10.0
        self.trail = deque(maxlen=self.cfg.ball_trail)
        self.recent = deque(maxlen=self.cfg.ball_fit_window)  # 直近の実測 (frame, x, y)
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
        if len(self.recent) >= 4 and frame_idx - self.recent[-1][0] <= 3 * self.cfg.slowdown:
            q = quadratic_predict(list(self.recent), frame_idx)
            if q is not None:
                pred = q
        return pred

    def gate(self):
        return self.cfg.ball_gate_px * (1 + 0.5 * self.misses / self.cfg.slowdown)

    def update(self, pred, cands):
        """cands: [(cx, cy, size, conf), ...]。戻り値: (x, y, size, 予測のみか) または None"""
        if not self.active:
            good = [c for c in cands if c[3] >= self.cfg.ball_init_conf]
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
            strong = [c for c in cands if c[3] >= self.cfg.ball_init_conf]
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
        if self.misses > self.cfg.ball_max_misses * self.cfg.slowdown:
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


def filter_ball_candidates(cands, H_img2pitch, spots, cfg):
    """ボール候補にピッチ座標を付け、ペナルティマーク・センターマークの上にある低信頼の候補を捨てる。
    戻り値: [(x, y, size, conf, pitch_x, pitch_y), ...]（2D座標がないフレームでは pitch は None）"""
    if not cands:
        return []
    if H_img2pitch is None or not spots:
        return [(*c, None, None) for c in cands]
    pts = np.float32([[c[0], c[1]] for c in cands]).reshape(-1, 1, 2)
    pxy = cv2.perspectiveTransform(pts, H_img2pitch).reshape(-1, 2)
    out = []
    for c, (px, py) in zip(cands, pxy):
        near_spot = any(np.hypot(px - sx, py - sy) <= cfg.ball_spot_radius_m for sx, sy in spots)
        if near_spot and c[3] < cfg.ball_spot_keep_conf:
            continue
        out.append((*c, float(px), float(py)))
    return out


def select_ball_path(cands_per_frame, cfg):
    """全フレームのボール候補から、動きとして一番もっともらしい経路を動的計画法で選ぶ。
    オンライン追跡と違い、予測が外れても「その後に映っているボール」を取りこぼさない。
    - 候補の得点: 信頼度 - cfg.ball_conf_offset
    - 前の候補から自然な速さでつながれば cfg.ball_link_bonus を加点（間隔が空くほど減点）
    - つながらない位置への移動は cfg.ball_jump_penalty を払えば常に可能（見失い→再発見）
    cands_per_frame: {frame: [(x, y, size, conf, pitch_x, pitch_y), ...]}
    戻り値: 選ばれた候補 [(frame, x, y, size), ...]"""
    max_gap = cfg.ball_link_gap * cfg.slowdown
    nodes = []
    for f in sorted(cands_per_frame):
        for c in sorted(cands_per_frame[f], key=lambda c: -c[3])[: cfg.ball_keep_topk]:
            nodes.append((f, c))
    if not nodes:
        return []
    n = len(nodes)
    dp = np.full(n, -np.inf)
    prev = np.full(n, -1)
    run_best, run_idx = -np.inf, -1
    i0 = 0  # 前のフレームまでのノードの開始位置
    for j, (fj, cj) in enumerate(nodes):
        # 前のフレームまでの最高得点を更新（同じフレーム内のノードからは飛べない）
        while i0 < j and nodes[i0][0] < fj:
            if dp[i0] > run_best:
                run_best, run_idx = dp[i0], i0
            i0 += 1
        emit = cj[3] - cfg.ball_conf_offset
        score, p = emit, -1  # ここから新しく始める
        if run_idx >= 0 and run_best - cfg.ball_jump_penalty + emit > score:
            score, p = run_best - cfg.ball_jump_penalty + emit, run_idx  # 見失い→再発見
        k = j - 1
        while k >= 0 and fj - nodes[k][0] <= max_gap:
            fk, ck = nodes[k]
            if fk < fj:
                gap_real = (fj - fk) / cfg.slowdown
                dist = np.hypot(cj[0] - ck[0], cj[1] - ck[1])
                if dist <= cfg.ball_max_speed_px * max(gap_real, 1 / cfg.slowdown):
                    speed = dist / max(gap_real, 1e-6)
                    # 止まっている物（白いソックス、ピッチ上の白い汚れ等）が毎フレーム続けて検出されても得をしないよう、
                    # ほぼ動いていないつながりには加点しない。2D座標があればピッチ上で判定する
                    # （画像上だとカメラのパンで止まっている物も動いて見えるため）
                    if cj[4] is not None and ck[4] is not None:
                        moving = min(
                            1.0, np.hypot(cj[4] - ck[4], cj[5] - ck[5]) / max(gap_real, 1e-6) / cfg.ball_static_m
                        )
                    else:
                        moving = min(1.0, speed / cfg.ball_static_px)
                    link = (
                        cfg.ball_link_bonus * moving
                        - 0.02 * max(gap_real - 1, 0)
                        - 0.05 * speed / cfg.ball_max_speed_px
                    )
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


def fit_ball_trajectory(obs, cfg):
    """ボールの実測 [(frame, x, y, size), ...] を弾道ごとに区切って2次式で当てはめ直す。
    - 予測から外れた1点だけの検出は誤検出として捨てる
    - 続けて外れたらキックかバウンドとみなし、そこで軌道を区切る
    - 軌道の中の欠けたフレームは2次式で、軌道と軌道の間の短い欠けは直線で埋める
    戻り値: {frame: (x, y, size, kind)}  kind は "fit" / "gap" / "raw" """
    obs = sorted(obs)
    max_gap = cfg.ball_seg_max_gap * cfg.slowdown
    segments, cur = [], []
    i = 0
    while i < len(obs):
        f, x, y, s = obs[i]
        if cur and f - cur[-1][0] > max_gap:
            segments.append(cur)
            cur = []
        if len(cur) >= 3:
            win = [(p[0], p[1], p[2]) for p in cur[-cfg.ball_fit_window :]]
            gap = (f - cur[-1][0]) / cfg.slowdown
            tol = cfg.ball_fit_tol_px * (1 + 0.5 * max(gap - 1, 0))
            px, py = quadratic_predict(win, f)
            if np.hypot(x - px, y - py) > tol:
                nxt = obs[i + 1] if i + 1 < len(obs) else None
                if nxt is not None:
                    qx, qy = quadratic_predict(win, nxt[0])
                    if np.hypot(nxt[1] - qx, nxt[2] - qy) <= tol * 1.5:
                        i += 1  # この1点だけ外れている → 誤検出として捨てる
                        continue
                segments.append(cur)  # 軌道が変わった（キック・バウンド）
                cur = [cur[-1]]  # 接触点は前後の軌道で共有する
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
                new_keep = res <= cfg.ball_fit_tol_px * 2
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
    return [
        ((b[0] + b[2]) / 2 + ox, (b[1] + b[3]) / 2 + oy, max(b[2] - b[0], b[3] - b[1]), float(c))
        for b, c in zip(boxes, confs)
    ]


def detect_ball_tiled(model, frame, cfg):
    """画面をタイルに分割して拡大推論し、ボールの候補を返す（遠く小さいボールの見逃し対策）"""
    if cfg.ball_tiles < 2:
        return []
    h, w = frame.shape[:2]
    th, tw = (
        int(h / cfg.ball_tiles * (1 + cfg.ball_tile_overlap)),
        int(w / cfg.ball_tiles * (1 + cfg.ball_tile_overlap)),
    )
    ys = np.linspace(0, h - th, cfg.ball_tiles).astype(int)
    xs = np.linspace(0, w - tw, cfg.ball_tiles).astype(int)
    tiles = [(x0, y0, frame[y0 : y0 + th, x0 : x0 + tw]) for y0 in ys for x0 in xs]
    results = model([t[2] for t in tiles], imgsz=cfg.imgsz, conf=cfg.ball_conf, classes=[32], verbose=False)
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


def search_ball_roi(model, frame, center, cfg):
    """予測位置の周りを切り出して拡大推論する（小さく速いボールの見逃し対策）"""
    h, w = frame.shape[:2]
    half = cfg.ball_roi // 2
    x0 = int(min(max(center[0] - half, 0), max(w - cfg.ball_roi, 0)))
    y0 = int(min(max(center[1] - half, 0), max(h - cfg.ball_roi, 0)))
    crop = frame[y0 : y0 + cfg.ball_roi, x0 : x0 + cfg.ball_roi]
    r = model(crop, imgsz=640, conf=cfg.ball_conf, classes=[32], verbose=False)[0]
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
