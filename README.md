# soccer-tracking（ローカル版）

Colab で動かしていたノートブックを、WSL2（Ubuntu）上のスクリプトにしたもの。

## 構成

```
soccer-tracking/
├── pyproject.toml
├── data/                 入力動画と出力（<name>_keypoints.json, _tracking.mp4, _positions.csv, _2d.mp4）
└── src/
    ├── pitch.py            ピッチの寸法・目印・白線のモデル（メートル座標）
    ├── pick_keyframes.py   全面の目印を複数フレームでクリック（<name>_keyframes.json）
    ├── register.py         全フレームの位置合わせ（<name>_homography.npz、確認用 _registration.mp4）
    ├── kp/                 ピッチのキーポイント検出
    │   ├── convert_soccernet.py  SoccerNet-Calibration → 学習データ
    │   ├── model.py              軽量 U-Net（MobileNetV3 + Coordinate Attention）
    │   ├── dataset.py            データ拡張（左右反転・画質劣化など）
    │   ├── train.py              学習・追加学習
    │   ├── annotate.py           自分の映像のアノテーション
    │   └── detect.py             推論とホモグラフィ
    ├── pick_keypoints.py   従来: 片側ゴールの PA 4角 + コーナー 2点をクリックしてホモグラフィを保存
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

## 使い方（全面・カメラが動く映像: 推奨）

1. キーフレームの指定（冒頭のセンターサークル、ゴール前など、映っている目印を数フレームでクリック）
```
uv run python src/pick_keyframes.py video3.mov
```

2. 全フレームの位置合わせ（確認用動画も出す）
```
uv run python src/register.py video3.mov --debug-video
```

3. トラッキング（`<動画名>_homography.npz` があれば自動で使い、2D座標はメートルで出る）
```
uv run python src/track.py video3.mov --slowdown 4
```

4. 2D の視覚化
```
uv run python src/visualize.py video3.mov
```

## ピッチのキーポイント検出モデル（自動の位置合わせ）

SoccerNet-Calibration の白線の注釈から37個の目印の学習データを作り、軽量 U-Net（MobileNetV3 + Coordinate Attention）を
フルスクラッチで学習する。学習したモデルを使うと、手動のキーフレームなしで位置合わせができる。

1. SoccerNet-Calibration のダウンロード（利用条件を確認すること）
```
uv run --with SoccerNet python -c "from SoccerNet.Downloader import SoccerNetDownloader as D; D(LocalDirectory='$HOME/data/soccernet').downloadDataTask(task='calibration-2023', split=['train','valid','test'])"
```
zip で落ちてくるので、`~/data/soccernet/calibration-2023/train/` などに展開する（00000.jpg と 00000.json が並ぶ形）。

2. 学習データへの変換（src/ で実行。確認用の画像が preview_<split>/ に出る）
```
cd src
uv run python -m kp.convert_soccernet --soccernet ~/data/soccernet/calibration-2023 --out ~/data/sn_keypoints
```

3. 動作確認（少ない枚数で1エポックだけ回して、エラーが出ないか見る）
```
uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/smoke --max-train 64 --epochs 1 --workers 0
```

4. 本番の学習
```
uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/pitch_kp
```

5. 自分の映像での追加学習（任意。位置合わせの結果を初期値にして、ずれた点だけ直す）
```
uv run python -m kp.annotate video3.mov --every 15
uv run python -m kp.train --data ~/data/sn_keypoints --extra ../data/video3_kp/index.json --init ../models/pitch_kp/best.pt --out ../models/pitch_kp_ft --epochs 15 --lr 5e-4
```

6. トラッキング（位置合わせを自動で実行してから追跡する。リポジトリ直下で実行）
```
cd ..
uv run python src/track.py video3.mov --slowdown 4 --kp-model models/pitch_kp/best.pt
```

## 使い方（従来: 片側ゴールのPAだけで位置合わせ）

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
