"""キーフレームの指定（全面・複数フレーム対応）

動画の中の何フレームか（例: 冒頭のセンターサークル、後半のゴール前）で、映っている目印をクリックする。
目印はセンターマーク・センターサークル・ハーフウェイライン・コーナー・PA・ゴールエリア・ペナルティマークなど37種類。
キーフレームが2つ以上あれば、register.py がその間を前後両方向からつないで全フレームの位置合わせをする。

操作:
  クリック        今選んでいる目印の位置を置く（置いたら次の目印に進む）
  n / p           次 / 前の目印を選ぶ（名前はターミナルに出る）
  g               目印の一覧を表示し、番号を入力して選ぶ
  x               今選んでいる目印の点を消す
  r               このフレームの点を全部消す
  a / d           1フレーム戻る / 進む
  j / l           1秒戻る / 進む
  u / i           10秒戻る / 進む
  Enter           このフレームをキーフレームとして保存（4点以上必要）
  k               保存済みのキーフレームの一覧
  q / Esc         終了
"""
""" """
import argparse
import json
import os

import cv2
import numpy as np

import pitch
from pick_keypoints import FrameReader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PREVIEW_WIDTH = 1280


def compute_homography(points):
    """クリックした点から 画像→ピッチ（m）のホモグラフィを計算し、誤差を表示する"""
    img = np.array([p["image"] for p in points], np.float32)
    pit = np.array([p["pitch"] for p in points], np.float32)
    H, inl = cv2.findHomography(img, pit, cv2.RANSAC if len(points) > 4 else 0, 1.5)
    if H is None:
        print("  ホモグラフィを計算できません（点が一直線上に並んでいる可能性）。")
        return None
    err = np.linalg.norm(cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2) - pit, axis=1)
    inl = np.ones(len(points), bool) if inl is None else inl.ravel().astype(bool)
    print(f"  再投影誤差（m）: 平均 {err.mean():.2f} / 最大 {err.max():.2f}")
    for p, e, ok in zip(points, err, inl):
        print(f"    {p['name']}: {e:.2f}{'' if ok else '  ← 外れ値（クリック位置か目印の選択を確認）'}")
    if len(points) == 4:
        print("  ※ 4点ちょうどなので誤差は必ず0です。確認画面のラインで判断してください。")
    return H


def draw_model(img, H_img2pitch, L, W, scale=1.0):
    """ピッチの白線をホモグラフィで映像に投影して描く（確認用）"""
    G = np.linalg.inv(H_img2pitch)
    out = img.copy()
    for seg, _ in pitch.line_segments(L, W):
        pts, _ = pitch.sample_segments([(seg, 1.0)], step=0.5)
        hom = np.c_[pts, np.ones(len(pts))] @ G.T
        run = []
        for u, v, w in hom:
            if w > 1e-6:
                run.append((u / w * scale, v / w * scale))
            else:
                if len(run) > 1:
                    cv2.polylines(out, [np.int32(run)], False, (255, 0, 255), 2)
                run = []
        if len(run) > 1:
            cv2.polylines(out, [np.int32(run)], False, (255, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="動画ファイル名（例: video3.mov）")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--pitch-length", type=float, default=None, help=f"ピッチの長さ（m、既定 {pitch.DEFAULT_LENGTH}）")
    ap.add_argument("--pitch-width", type=float, default=None, help=f"ピッチの幅（m、既定 {pitch.DEFAULT_WIDTH}）")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    name = os.path.splitext(args.video)[0]
    out_json = os.path.join(data_dir, f"{name}_keyframes.json")

    # 既存のキーフレームがあれば読み込んで追記する
    doc = {"video": args.video, "pitch_length": pitch.DEFAULT_LENGTH, "pitch_width": pitch.DEFAULT_WIDTH, "keyframes": []}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as f:
            doc = json.load(f)
        print(f"{os.path.basename(out_json)} を読み込みました（キーフレーム {len(doc['keyframes'])} 個）")
    if args.pitch_length:
        doc["pitch_length"] = args.pitch_length
    if args.pitch_width:
        doc["pitch_width"] = args.pitch_width
    L, W = doc["pitch_length"], doc["pitch_width"]
    KP = pitch.keypoints(L, W)
    print(f"ピッチ: {L} m × {W} m")

    def save():
        doc["keyframes"].sort(key=lambda k: k["frame"])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

    reader = FrameReader(os.path.join(data_dir, args.video))
    fps_i = max(1, int(round(reader.fps)))
    print(f"{args.video}: {reader.n_frames} フレーム, {reader.fps:.2f} fps")

    st = {"frame": -1, "cur": 0, "pts": {}, "scale": 1.0, "img": None, "dirty": True, "warned": False, "saved": True}
    win = "keyframes"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def announce():
        k, jp, xy = KP[st["cur"]]
        mark = "（済）" if k in st["pts"] else ""
        print(f"[{st['cur'] + 1:2d}/{len(KP)}] {jp}{mark}")

    def load(f):
        f = int(max(0, min(f, reader.n_frames - 1)))
        if f == st["frame"]:
            return
        if st["pts"] and not st["saved"] and not st["warned"]:
            print("  まだ保存していない点があります。Enter で保存するか、もう一度移動キーを押すと破棄して移動します。")
            st["warned"] = True
            return
        img = reader.get(f)
        if img is None:
            print(f"  フレーム {f} は読めませんでした")
            return
        st.update(frame=f, warned=False, saved=True, dirty=True)
        st["scale"] = min(1.0, PREVIEW_WIDTH / img.shape[1])
        st["img"] = cv2.resize(img, None, fx=st["scale"], fy=st["scale"]) if st["scale"] < 1 else img
        # 保存済みのキーフレームならその点を読み込む
        kf = next((k for k in doc["keyframes"] if k["frame"] == f), None)
        st["pts"] = {p["key"]: p["image"] for p in kf["points"]} if kf else {}
        if kf:
            print(f"  フレーム {f} は保存済みのキーフレームです（{len(st['pts'])} 点）")
        cv2.setTrackbarPos("frame", win, f)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["pts"][KP[st["cur"]][0]] = [x / st["scale"], y / st["scale"]]
            st["saved"], st["dirty"] = False, True
            st["cur"] = (st["cur"] + 1) % len(KP)
            announce()

    cv2.createTrackbar("frame", win, 0, max(reader.n_frames - 1, 1), lambda v: load(v))
    cv2.setMouseCallback(win, on_mouse)
    load(0)
    announce()

    def render():
        img = st["img"].copy()
        s = st["scale"]
        for i, (k, jp, xy) in enumerate(KP):
            if k not in st["pts"]:
                continue
            x, y = int(st["pts"][k][0] * s), int(st["pts"][k][1] * s)
            col = (0, 255, 255) if i != st["cur"] else (0, 128, 255)
            cv2.circle(img, (x, y), 5, col, -1)
            cv2.putText(img, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, str(i + 1), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        kfs = [k["frame"] for k in doc["keyframes"]]
        msg = (f"frame {st['frame']} ({st['frame'] / reader.fps:.2f}s)  points {len(st['pts'])}  "
               f"target #{st['cur'] + 1}  keyframes {kfs}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (40, 40, 40), -1)
        cv2.putText(img, msg, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img

    while True:
        if st["dirty"]:
            cv2.imshow(win, render())
            st["dirty"] = False
        k = cv2.waitKey(30) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        nav = {ord("a"): -1, ord("d"): 1, ord("j"): -fps_i, ord("l"): fps_i, ord("u"): -10 * fps_i, ord("i"): 10 * fps_i}
        if k in nav:
            load(st["frame"] + nav[k])
        elif k == ord("n"):
            st["cur"] = (st["cur"] + 1) % len(KP); st["dirty"] = True; announce()
        elif k == ord("p"):
            st["cur"] = (st["cur"] - 1) % len(KP); st["dirty"] = True; announce()
        elif k == ord("g"):
            for i, (key, jp, _) in enumerate(KP):
                print(f"  {i + 1:2d}: {jp}{'（済）' if key in st['pts'] else ''}")
            try:
                st["cur"] = int(input("番号: ")) - 1
                st["cur"] = max(0, min(st["cur"], len(KP) - 1))
            except ValueError:
                pass
            st["dirty"] = True
            announce()
        elif k == ord("x"):
            st["pts"].pop(KP[st["cur"]][0], None); st["saved"] = False; st["dirty"] = True
        elif k == ord("r"):
            st["pts"] = {}; st["saved"] = False; st["dirty"] = True
        elif k == ord("k"):
            for kf in doc["keyframes"]:
                print(f"  フレーム {kf['frame']}（{kf['frame'] / reader.fps:.2f} 秒）: {len(kf['points'])} 点")
        elif k in (13, 10):
            points = [{"key": key, "name": jp, "pitch": list(xy), "image": st["pts"][key]}
                      for key, jp, xy in KP if key in st["pts"]]
            print(f"フレーム {st['frame']}: {len(points)} 点")
            if len(points) < 4:
                print("  4点以上必要です。")
                continue
            H = compute_homography(points)
            if H is None:
                continue
            doc["keyframes"] = [kf for kf in doc["keyframes"] if kf["frame"] != st["frame"]]
            doc["keyframes"].append({"frame": st["frame"], "points": points, "H_img2pitch": H.tolist()})
            save()
            st["saved"] = True
            print(f"  保存: {out_json}（キーフレーム {len(doc['keyframes'])} 個）")
            check = draw_model(st["img"], H, L, W, scale=st["scale"])
            cv2.imshow("check (any key to close)", check)
            cv2.waitKey(0)
            cv2.destroyWindow("check (any key to close)")
            st["dirty"] = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
