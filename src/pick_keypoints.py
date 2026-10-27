"""ピッチのキーポイント指定（ローカル版）

OpenCV のウィンドウで動画のフレームを移動しながら、片側ゴールの PA 4角とコーナー 2点をクリックする。
WSL2 では WSLg（Windows 11）でウィンドウが開く。

操作:
  クリック        現在の項目の点を置く
  a / d           1フレーム戻る / 進む
  j / l           1秒戻る / 進む
  u / i           10秒戻る / 進む
  トラックバー    フレームを直接指定
  s               現在の項目をスキップ（映っていないとき）
  b               1つ戻る
  r               点をリセット
  Enter           確定してホモグラフィを計算・保存
  q / Esc         保存せずに終了
"""
import argparse
import json
import os
from collections import OrderedDict

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PREVIEW_WIDTH = 1280

# 映像の中でゴールが画面の左右どちらにあるか（"right" / "left"）
GOAL_ON_SCREEN = "right"   # --goal で上書き
# 実際のピッチの幅（m）。コーナーの位置だけに使う（PAの寸法は規格で固定）
PITCH_WIDTH_M = 68.0       # --pitch-width で上書き

# ピッチ座標は StatsBomb（120×80ヤード）。映っているゴールは常に右端（x=120）に置く
# 「奥」はカメラから遠い側、「手前」はカメラに近い側
# 画面左にゴールがある映像は180°回転させて右端に置く（鏡像にならないよう奥と手前のyも入れ替える）
_corner_off = (PITCH_WIDTH_M / 2) / 0.9144   # 中央からコーナーまでの距離（ヤード）


def _y(far_y):
    """奥側を基準にしたy座標を、ゴールの向きに合わせて変換する"""
    return far_y if GOAL_ON_SCREEN == "right" else 80 - far_y


PITCH_KEYPOINTS = [
    ("PA ゴールライン上・奥", (120, _y(18))),
    ("PA ゴールライン上・手前", (120, _y(62))),
    ("PA フィールド側・奥", (102, _y(18))),
    ("PA フィールド側・手前", (102, _y(62))),
    ("コーナー・奥（映っていなければスキップ）", (120, _y(40 - _corner_off))),
    ("コーナー・手前（映っていなければスキップ）", (120, _y(40 + _corner_off))),
]


class FrameReader:
    """フレーム番号を指定して読む。トラッキングセルと番号がずれないよう、シークせず先頭から順に読む。
    一度読んだフレームはキャッシュするので、行き来は速い。"""

    def __init__(self, path, cache_size=30):
        if not os.path.exists(path):
            raise FileNotFoundError(f"動画が見つかりません: {path}（Driveのマウントとファイル名を確認）")
        self.path = path
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません（コーデック非対応の可能性）: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def _open(self):
        self.cap = cv2.VideoCapture(self.path)
        self.pos = 0  # 次に読むフレーム番号

    def get(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx < self.pos:  # 戻るときは開き直して先頭から
            self.cap.release()
            self._open()
        while self.pos < idx:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        self.cache[idx] = frame
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame


def pitch_line_segments():
    """検証用に描くゴール側半面のライン（StatsBomb座標）"""
    c = _corner_off
    segs = [
        [(60, 40 - c), (120, 40 - c), (120, 40 + c), (60, 40 + c)],
        [(60, 40 - c), (60, 40 + c)],
        [(120, 18), (102, 18), (102, 62), (120, 62)],
        [(120, 30), (114, 30), (114, 50), (120, 50)],
    ]
    t = np.linspace(-np.pi / 2, np.pi / 2, 30)
    segs.append(list(zip(60 + 10 * np.cos(t), 40 + 10 * np.sin(t))))
    # PAアーク（ペナルティスポット x=108 から半径10、PA外側の部分だけ）
    t = np.linspace(np.pi - 0.927, np.pi + 0.927, 20)
    segs.append(list(zip(108 + 10 * np.cos(t), 40 + 10 * np.sin(t))))
    if GOAL_ON_SCREEN == "left":  # 180°回転に合わせて y を反転（x は右ゴール基準のまま）
        segs = [[(x, 80 - y) for x, y in seg] for seg in segs]
    return segs


def draw_projected_pitch(frame, H_img2pitch):
    """ホモグラフィの逆変換でピッチのラインを映像に重ね、点の取り方が正しいか確認する"""
    H_inv = np.linalg.inv(H_img2pitch)
    out = frame.copy()
    for seg in pitch_line_segments():
        dense = []
        for (x0, y0), (x1, y1) in zip(seg[:-1], seg[1:]):
            for s in np.linspace(0, 1, 20):
                dense.append((x0 + (x1 - x0) * s, y0 + (y1 - y0) * s))
        hom = np.c_[np.array(dense), np.ones(len(dense))] @ H_inv.T
        run = []
        for (u, v, w) in hom:
            if w > 1e-6:  # カメラの後ろに回る点は描かない
                run.append((u / w, v / w))
            elif len(run) > 1:
                cv2.polylines(out, [np.int32(run)], False, (255, 0, 255), 2)
                run = []
            else:
                run = []
        if len(run) > 1:
            cv2.polylines(out, [np.int32(run)], False, (255, 0, 255), 2)
    return out




def compute_and_save(reader, video_name, data_dir, frame_idx, clicked):
    frame = reader.get(frame_idx)
    points = [{"name": name, "pitch": list(pxy), "image": ixy}
              for (name, pxy), ixy in zip(PITCH_KEYPOINTS, clicked) if ixy is not None]
    print(f"フレーム {frame_idx}（{frame_idx / reader.fps:.2f} 秒）  取得した点: {len(points)}")
    if len(points) < 4:
        print("4点以上必要です。PAの4角は必ず取ってください。")
        return None
    img_pts = np.array([p["image"] for p in points], np.float32)
    pitch_pts = np.array([p["pitch"] for p in points], np.float32)
    H, inliers = cv2.findHomography(img_pts, pitch_pts, cv2.RANSAC, 3.0)
    if H is None:
        print("ホモグラフィを計算できません（点が一直線上に並んでいる可能性）。")
        return None
    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(proj - pitch_pts, axis=1)
    print(f"再投影誤差（120×80スケール）: 平均 {errors.mean():.2f} / 最大 {errors.max():.2f}")
    for p, e, inl in zip(points, errors, inliers.ravel()):
        print(f"  {p['name']}: {e:.2f}{'' if inl else '  ← 外れ値（クリック位置か対応付けを確認）'}")
    if len(points) == 4:
        print("※ 4点ちょうどなので誤差は必ず0になります。確認ウィンドウのラインで判断してください。")

    out_json = os.path.join(data_dir, f"{os.path.splitext(video_name)[0]}_keypoints.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"video": video_name, "reference_frame": frame_idx, "goal_on_screen": GOAL_ON_SCREEN,
                   "pitch_width_m": PITCH_WIDTH_M, "points": points, "H_img2pitch": H.tolist()},
                  f, ensure_ascii=False, indent=2)
    print(f"保存: {out_json}")
    if frame_idx > 0:
        print(f"※ 2D座標はフレーム {frame_idx} 以降のみ出力されます（カメラが動く設定の場合）。")

    check = draw_projected_pitch(frame, H)
    for p in points:
        cv2.circle(check, tuple(int(v) for v in p["image"]), 6, (0, 255, 255), -1)
    out_png = os.path.join(data_dir, f"{os.path.splitext(video_name)[0]}_keypoints_check.png")
    cv2.imwrite(out_png, check)
    print(f"確認画像: {out_png}（黄: クリックした点 / マゼンタ: 投影したピッチのライン）")
    return check


def main():
    global GOAL_ON_SCREEN, PITCH_WIDTH_M, PITCH_KEYPOINTS, _corner_off
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="動画ファイル名（例: video1.mp4）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--goal", choices=["right", "left"], default=GOAL_ON_SCREEN, help="ゴールが画面のどちら側に映っているか")
    ap.add_argument("--pitch-width", type=float, default=PITCH_WIDTH_M, help="実際のピッチ幅（m）。コーナーの位置にだけ使う")
    args = ap.parse_args()
    GOAL_ON_SCREEN, PITCH_WIDTH_M = args.goal, args.pitch_width
    _corner_off = (PITCH_WIDTH_M / 2) / 0.9144
    PITCH_KEYPOINTS = [
        ("PA ゴールライン上・奥", (120, _y(18))),
        ("PA ゴールライン上・手前", (120, _y(62))),
        ("PA フィールド側・奥", (102, _y(18))),
        ("PA フィールド側・手前", (102, _y(62))),
        ("コーナー・奥（映っていなければ s でスキップ）", (120, _y(40 - _corner_off))),
        ("コーナー・手前（映っていなければ s でスキップ）", (120, _y(40 + _corner_off))),
    ]
    data_dir = os.path.abspath(args.data_dir)
    reader = FrameReader(os.path.join(data_dir, args.video))
    print(f"{args.video}: {reader.n_frames} フレーム, {reader.fps:.2f} fps")

    state = {"frame": 0, "idx": 0, "pts": [], "scale": 1.0, "img": None, "dirty": True}
    win = "keypoints"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def load(f):
        f = int(max(0, min(f, reader.n_frames - 1)))
        img = reader.get(f)
        if img is None:
            print(f"フレーム {f} は読めませんでした")
            return
        if f != state["frame"] and state["pts"]:
            print("フレームを変えたので点をリセットしました")
        if f != state["frame"]:
            state["pts"], state["idx"] = [], 0
        state["frame"] = f
        state["scale"] = min(1.0, PREVIEW_WIDTH / img.shape[1])
        state["img"] = cv2.resize(img, None, fx=state["scale"], fy=state["scale"]) if state["scale"] < 1 else img
        state["dirty"] = True
        cv2.setTrackbarPos("frame", win, f)

    def on_trackbar(v):
        if v != state["frame"]:
            load(v)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["idx"] < len(PITCH_KEYPOINTS):
            state["pts"].append([x / state["scale"], y / state["scale"]])
            state["idx"] += 1
            state["dirty"] = True

    cv2.createTrackbar("frame", win, 0, max(reader.n_frames - 1, 1), on_trackbar)
    cv2.setMouseCallback(win, on_mouse)
    load(0)

    def render():
        img = state["img"].copy()
        s = state["scale"]
        for i, p in enumerate(state["pts"]):
            if p is None:
                continue
            x, y = int(p[0] * s), int(p[1] * s)
            cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(img, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        n = sum(1 for p in state["pts"] if p is not None)
        idx = state["idx"]
        # OpenCV は日本語を描けないので、項目名はターミナルに出し、画面には番号だけ出す
        msg = (f"point {idx + 1}/{len(PITCH_KEYPOINTS)}  (got {n})   frame {state['frame']}  "
               f"{state['frame'] / reader.fps:.2f}s") if idx < len(PITCH_KEYPOINTS) else \
              f"done ({n} points)  press Enter to save   frame {state['frame']}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (40, 40, 40), -1)
        cv2.putText(img, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    last_idx = -1
    fps_i = max(1, int(round(reader.fps)))
    while True:
        if state["dirty"]:
            cv2.imshow(win, render())
            state["dirty"] = False
            if state["idx"] != last_idx:
                last_idx = state["idx"]
                if state["idx"] < len(PITCH_KEYPOINTS):
                    print(f"({state['idx'] + 1}/{len(PITCH_KEYPOINTS)}) {PITCH_KEYPOINTS[state['idx']][0]} をクリック")
                else:
                    print("全項目終了。Enter で確定")
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("a"): load(state["frame"] - 1)
        elif k == ord("d"): load(state["frame"] + 1)
        elif k == ord("j"): load(state["frame"] - fps_i)
        elif k == ord("l"): load(state["frame"] + fps_i)
        elif k == ord("u"): load(state["frame"] - 10 * fps_i)
        elif k == ord("i"): load(state["frame"] + 10 * fps_i)
        elif k == ord("s") and state["idx"] < len(PITCH_KEYPOINTS):
            state["pts"].append(None); state["idx"] += 1; state["dirty"] = True
        elif k == ord("b") and state["idx"] > 0:
            state["pts"].pop(); state["idx"] -= 1; state["dirty"] = True
        elif k == ord("r"):
            state["pts"], state["idx"], state["dirty"] = [], 0, True
        elif k in (13, 10):
            clicked = state["pts"] + [None] * (len(PITCH_KEYPOINTS) - len(state["pts"]))
            check = compute_and_save(reader, args.video, data_dir, state["frame"], clicked)
            if check is not None:
                s = state["scale"]
                cv2.imshow("check (any key to close)", cv2.resize(check, None, fx=s, fy=s) if s < 1 else check)
                cv2.waitKey(0)
                cv2.destroyWindow("check (any key to close)")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
