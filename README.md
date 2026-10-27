# soccer-tracking（ローカル版）

Colab で動かしていたノートブックを、WSL2（Ubuntu）上のスクリプトにしたもの。

## 構成

```
soccer-tracking/
├── pyproject.toml
├── data/                 入力動画と出力（<name>_keypoints.json, _tracking.mp4, _positions.csv, _2d.mp4）
└── src/
    ├── pick_keypoints.py   片側ゴールの PA 4角 + コーナー 2点をクリックしてホモグラフィを保存
    ├── track.py            検出・追跡・2D座標の出力（Colab のトラッキングセル）
    └── visualize.py        2D の静止画と動画（Colab の視覚化セル）
```

## 初回セットアップ

ffmpeg（出力動画の書き出しに使う）
```
sudo apt install -y ffmpeg
```

Python 環境
```
cd ~/dev/personal/soccer-tracking
uv sync
```

GPU の確認（True NVIDIA GeForce RTX 3080 と出れば OK）
```
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 使い方

動画を data/ に置く（Windows 側のファイルは cp で Linux 側へコピーする。/mnt/c を直接読むと遅い）
```
cp /mnt/c/Users/KEN/Videos/video1.mp4 data/
```

1. キーポイント指定（ゴールが画面左なら --goal left）
```
uv run python src/pick_keypoints.py video1.mp4 --goal right
```

2. トラッキング（0.25倍速の動画なら --slowdown 4、通常速度なら --slowdown 1）
```
uv run python src/track.py video1.mp4 --slowdown 4
```

3. 2D の視覚化
```
uv run python src/visualize.py video1.mp4
```

設定値（モデル名、閾値、ボールの弾道フィットなど）は各スクリプト冒頭の定数で変える。
