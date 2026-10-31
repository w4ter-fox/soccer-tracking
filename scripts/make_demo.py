"""README 用のデモ GIF を作る

  1. 元動画とトラッキング動画を左右に並べた GIF（docs/media/demo.gif）
  2. 2D のピッチ図の GIF（docs/media/demo_2d.gif）

元動画は可変フレームレートのことがあり、ffmpeg で時刻を基準に並べると左右がずれる。
track.py の出力（入力と同じ速度）はフレーム番号が元動画と1対1なので、OpenCV でフレーム番号をそろえて並べる。

使い方（リポジトリの直下で実行）:
  uv run python scripts/make_demo.py video3.mov
  uv run python scripts/make_demo.py video3.mov --start 200 --frames 180 --width 960 --fps 12
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read_frames(path, start, n, step):
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i >= start and (i - start) % step == 0:
            out.append(f)
        i += 1
    cap.release()
    return out


def label(img, text):
    cv2.rectangle(img, (0, 0), (12 + 14 * len(text), 34), (30, 30, 30), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def to_gif(frames, out_path, fps, width):
    """フレーム列を、パレットを最適化した GIF にする（ffmpeg の2段階変換）"""
    h, w = frames[0].shape[:2]
    tmp = tempfile.mkdtemp()
    try:
        mp4 = os.path.join(tmp, "clip.mp4")
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", mp4],
            stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        proc.wait()
        palette = os.path.join(tmp, "palette.png")
        scale = f"fps={fps},scale={width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", f"{scale},palettegen=stats_mode=diff",
                        palette], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i", palette, "-lavfi",
                        f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                        out_path], check=True)
    finally:
        shutil.rmtree(tmp)
    print(f"保存: {out_path}（{os.path.getsize(out_path) / 1e6:.1f} MB）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="data/ にある元動画のファイル名")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs", "media"))
    ap.add_argument("--start", type=int, default=0, help="使い始めるフレーム番号")
    ap.add_argument("--frames", type=int, default=180, help="GIF に使う元動画のフレーム数")
    ap.add_argument("--step", type=int, default=3, help="何フレームごとに使うか（GIF を軽くする）")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=960, help="並べた GIF の横幅（px）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.splitext(args.video)[0]
    src = read_frames(os.path.join(args.data_dir, args.video), args.start, args.frames // args.step, args.step)
    trk = read_frames(os.path.join(args.data_dir, f"{name}_tracking.mp4"), args.start, args.frames // args.step, args.step)
    n = min(len(src), len(trk))
    if n == 0:
        raise SystemExit("フレームを読めませんでした。元動画と _tracking.mp4 が data/ にあるか確認してください。")
    h, w = src[0].shape[:2]
    pairs = []
    for a, b in zip(src[:n], trk[:n]):
        b = cv2.resize(b, (w, h)) if b.shape[:2] != (h, w) else b
        pairs.append(np.hstack([label(a.copy(), "Input"), label(b.copy(), "Tracking")]))
    to_gif(pairs, os.path.join(args.out_dir, "demo.gif"), args.fps, args.width)

    p2d = os.path.join(args.data_dir, f"{name}_2d.mp4")
    if os.path.exists(p2d):
        frames = read_frames(p2d, 0, args.frames // args.step, args.step)
        if frames:
            to_gif(frames, os.path.join(args.out_dir, "demo_2d.gif"), args.fps, min(args.width, 800))
    else:
        print(f"{name}_2d.mp4 がないので、2D の GIF は作りません（visualize.py を先に実行）")


if __name__ == "__main__":
    main()
