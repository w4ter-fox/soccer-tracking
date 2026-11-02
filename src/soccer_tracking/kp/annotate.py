"""自分の映像のキーポイントのアノテーション（追加学習用）

動画から一定間隔でフレームを取り出し、37個の目印の位置を付ける。
初期値として、位置合わせの結果（<動画名>_homography.npz）か、学習済みモデルの推論結果を表示するので、
ずれている点をドラッグで直し、見えていない点を消し、足りない点を足すだけでよい。

使い方（src/ で実行）:
  uv run python -m kp.annotate video3.mov --every 15
  uv run python -m kp.annotate video3.mov --every 15 --kp-model ../models/pitch_kp/best.pt

操作:
  ドラッグ        点を動かす（点の近くを押して動かす）
  クリック        何もないところを押すと、今選んでいる目印をそこに置く
  n / p / g       置く目印を選ぶ（g は一覧から番号で）
  x               最後に触った点を消す
  c               このフレームの点を全部消す
  l               点からホモグラフィを計算して、ピッチの白線を重ねて確認する（もう一度押すと消える）
  a / d           前 / 次のフレーム（移動時に自動で保存）
  s               保存
  q / Esc         保存して終了

出力: data/<動画名>_kp/ に画像と index.json（kp.train の --extra に渡す）
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pitch  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
PREVIEW_WIDTH = 1280
PICK_RADIUS = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--every", type=int, default=15, help="何フレームごとに取り出すか")
    ap.add_argument("--kp-model", default=None, help="初期値に使う学習済みモデル（位置合わせの結果がないとき）")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    name = os.path.splitext(args.video)[0]
    out_dir = os.path.join(data_dir, f"{name}_kp")
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, "index.json")

    npz_path = os.path.join(data_dir, f"{name}_homography.npz")
    reg = np.load(npz_path) if os.path.exists(npz_path) else None
    L = float(reg["pitch_length"]) if reg is not None else pitch.DEFAULT_LENGTH
    W = float(reg["pitch_width"]) if reg is not None else pitch.DEFAULT_WIDTH
    KP = pitch.keypoints(L, W)
    KEYS = [k for k, _, _ in KP]
    detector = None
    if reg is None and args.kp_model:
        from kp.detect import KeypointDetector
        detector = KeypointDetector(args.kp_model)
    print(f"初期値: {'位置合わせの結果' if reg is not None else ('モデルの推論' if detector else 'なし')}")

    # 対象フレームを取り出して保存（まだなければ）
    cap = cv2.VideoCapture(os.path.join(data_dir, args.video))
    frames = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % args.every == 0:
            p = os.path.join(out_dir, f"{name}_{f:06d}.jpg")
            if not os.path.exists(p):
                cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append((f, p))
        f += 1
    cap.release()
    print(f"{len(frames)} フレームを対象にします（{out_dir}）")

    ann = {}
    if os.path.exists(index_path):
        with open(index_path) as fh:
            for it in json.load(fh):
                ann[it["image"]] = it["keypoints"]

    def initial(fi, img):
        h, w = img.shape[:2]
        if reg is not None and fi < len(reg["valid"]) and reg["valid"][fi]:
            G = np.linalg.inv(reg["H"][fi])
            out = {}
            for k, _, (x, y) in KP:
                hv = G @ [x, y, 1]
                if hv[2] > 1e-9:
                    u, v = hv[:2] / hv[2]
                    if 0 <= u < w and 0 <= v < h:
                        out[k] = [float(u), float(v)]
            return out
        if detector is not None:
            return {k: [u, v] for k, (u, v, s) in detector(img).items()}
        return {}

    def save():
        items = []
        for fi, p in frames:
            if p in ann and ann[p]:
                img_h, img_w = st["shape"]
                items.append({"image": p, "width": img_w, "height": img_h, "keypoints": ann[p]})
        with open(index_path, "w") as fh:
            json.dump(items, fh)
        print(f"保存: {index_path}（{len(items)} 枚）")

    st = {"i": 0, "cur": 0, "drag": None, "last": None, "lines": False, "scale": 1.0, "img": None, "shape": None}
    win = "annotate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def load(i):
        st["i"] = max(0, min(i, len(frames) - 1))
        fi, p = frames[st["i"]]
        img = cv2.imread(p)
        st["shape"] = img.shape[:2]
        st["scale"] = min(1.0, PREVIEW_WIDTH / img.shape[1])
        st["img"] = cv2.resize(img, None, fx=st["scale"], fy=st["scale"]) if st["scale"] < 1 else img
        if p not in ann:
            ann[p] = initial(fi, img)
        print(f"[{st['i'] + 1}/{len(frames)}] フレーム {fi}: {len(ann[p])} 点")

    def pts():
        return ann[frames[st["i"]][1]]

    def nearest(x, y):
        best, bd = None, PICK_RADIUS
        for k, (u, v) in pts().items():
            d = np.hypot(u * st["scale"] - x, v * st["scale"] - y)
            if d < bd:
                best, bd = k, d
        return best

    def on_mouse(event, x, y, flags, param):
        s = st["scale"]
        if event == cv2.EVENT_LBUTTONDOWN:
            k = nearest(x, y)
            if k is not None:
                st["drag"] = st["last"] = k
            else:
                k = KEYS[st["cur"]]
                pts()[k] = [x / s, y / s]
                st["last"] = k
                st["cur"] = (st["cur"] + 1) % len(KEYS)
                print(f"  置く目印: [{st['cur'] + 1}] {KP[st['cur']][1]}")
        elif event == cv2.EVENT_MOUSEMOVE and st["drag"]:
            pts()[st["drag"]] = [x / s, y / s]
        elif event == cv2.EVENT_LBUTTONUP:
            st["drag"] = None

    cv2.setMouseCallback(win, on_mouse)
    load(0)
    print(f"  置く目印: [{st['cur'] + 1}] {KP[st['cur']][1]}")

    while True:
        img = st["img"].copy()
        s = st["scale"]
        P = pts()
        if st["lines"] and len(P) >= 4:
            src = np.float32([P[k] for k in P])
            dst = np.float32([dict((k, xy) for k, _, xy in KP)[k] for k in P])
            H, _ = cv2.findHomography(src, dst, 0)
            if H is not None:
                G = np.linalg.inv(H)
                lp, _ = pitch.sample_segments(pitch.line_segments(L, W), step=0.3)
                hom = np.c_[lp, np.ones(len(lp))] @ G.T
                ok = hom[:, 2] > 1e-9
                uv = (hom[ok, :2] / hom[ok, 2:] * s).astype(int)
                m = (uv[:, 0] >= 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < img.shape[0])
                img[uv[m, 1], uv[m, 0]] = (255, 0, 255)
        for k, (u, v) in P.items():
            i = KEYS.index(k)
            col = (0, 128, 255) if k == st["last"] else (0, 255, 255)
            cv2.circle(img, (int(u * s), int(v * s)), 5, col, -1)
            cv2.putText(img, str(i + 1), (int(u * s) + 6, int(v * s) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, str(i + 1), (int(u * s) + 6, int(v * s) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (40, 40, 40), -1)
        cv2.putText(img, f"{st['i'] + 1}/{len(frames)}  points {len(P)}  next #{st['cur'] + 1}",
                    (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            save()
            break
        if k == ord("a"):
            save(); load(st["i"] - 1)
        elif k == ord("d"):
            save(); load(st["i"] + 1)
        elif k == ord("s"):
            save()
        elif k == ord("n"):
            st["cur"] = (st["cur"] + 1) % len(KEYS); print(f"  置く目印: [{st['cur'] + 1}] {KP[st['cur']][1]}")
        elif k == ord("p"):
            st["cur"] = (st["cur"] - 1) % len(KEYS); print(f"  置く目印: [{st['cur'] + 1}] {KP[st['cur']][1]}")
        elif k == ord("g"):
            for i, (kk, jp, _) in enumerate(KP):
                print(f"  {i + 1:2d}: {jp}{'（済）' if kk in P else ''}")
            try:
                st["cur"] = max(0, min(int(input("番号: ")) - 1, len(KEYS) - 1))
            except ValueError:
                pass
            print(f"  置く目印: [{st['cur'] + 1}] {KP[st['cur']][1]}")
        elif k == ord("x") and st["last"] in P:
            P.pop(st["last"]); st["last"] = None
        elif k == ord("c"):
            P.clear()
        elif k == ord("l"):
            st["lines"] = not st["lines"]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
