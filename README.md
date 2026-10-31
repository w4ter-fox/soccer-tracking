# Soccer Tracking & Pitch Registration

Player, referee and ball tracking for amateur soccer footage, with automatic pitch registration that maps every player onto a metric 2D pitch — even when the camera pans from the center circle to the goal.

<p align="center">
  <img src="docs/media/demo.gif" alt="Input video (left) and tracking result (right)" width="100%">
</p>

<p align="center">
  <img src="docs/media/demo_2d.gif" alt="Players projected onto a 2D pitch" width="70%">
</p>

> 日本語の説明は [README.ja.md](README.ja.md) を参照してください。

## Highlights

- **Tracking that survives occlusion** — YOLO11 + ByteTrack, then offline re-linking of broken tracklets by extrapolating each player's last velocity (Hungarian assignment, team-constrained).
- **Team assignment per track, not per frame** — jersey colors are voted over the whole track (ignoring occluded frames), so a player's color never flickers.
- **Ball tracking as a global path problem** — all candidates (full frame, tiled, and ROI re-detection) are collected, then a dynamic-programming search picks the most plausible trajectory; ballistic (quadratic) fits fill the gaps. Penalty and center spots are rejected in pitch coordinates.
- **Pitch registration without manual clicks** — a 4 M-parameter MobileNetV3 U-Net with Coordinate Attention, trained from scratch on SoccerNet-Calibration, detects 37 pitch landmarks every few frames. Camera motion (ORB) bridges the frames in between, and white lines plus grass mowing stripes refine the homography.

## Pipeline

```mermaid
flowchart LR
    V[Video] --> D[YOLO11 detection]
    D --> T[ByteTrack]
    T --> L[Tracklet re-linking<br/>velocity extrapolation]
    L --> C[Team vote per track]
    D --> B[Ball candidates<br/>full / tiled / ROI]
    B --> P[DP path selection<br/>+ ballistic fit]
    V --> K[Keypoint U-Net<br/>every N frames]
    K --> R[Homography<br/>RANSAC]
    V --> M[Camera motion<br/>ORB]
    R --> G[Registration<br/>lines + stripes refine, smoothing]
    M --> G
    C --> O[Tracking video<br/>+ 2D positions in meters]
    P --> O
    G --> O
```

## Results

| Component | Metric | Value |
|---|---|---|
| Pitch keypoint detector (SoccerNet-Calibration valid, 1,629 images) | PCK@5px (640×352 input) | 0.891 |
| | Images with ≥5 correct keypoints (homography possible) | 95.1% |
| | False detections of off-screen keypoints | 0.3% |
| Registration (synthetic pan from center circle to goal, 300 frames) | Median ground error | 0.17 m |
| Tracklet re-linking (synthetic, 17 breaks of 0.3–2 s) | Correct links / wrong links | 16 / 0 |

The detector is 4.07 M parameters and trains in about one hour on an RTX 3080 (80 epochs).

## Setup

Tested on Ubuntu (WSL2) with an RTX 3080.

```bash
sudo apt install -y ffmpeg
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Put videos in `data/` (copy them into the Linux filesystem; reading from `/mnt/c` is slow).

## Usage

Track players and the ball, register the pitch automatically, and export 2D positions:

```bash
uv run python src/track.py video3.mov --slowdown 4 --kp-model models/pitch_kp/best.pt
uv run python src/visualize.py video3.mov
```

Outputs in `data/`:

| File | Content |
|---|---|
| `<name>_tracking.mp4` | Input video with color-coded boxes (team, referee, goalkeeper) and the ball trail |
| `<name>_positions.csv` | Per-frame positions: image coordinates and pitch coordinates in meters |
| `<name>_tracks.csv` | Per-track team and how it was decided |
| `<name>_homography.npz` | Per-frame image-to-pitch homography |
| `<name>_2d.mp4` | Players on a 2D pitch |

Useful options:

- `--slowdown 4` — the input is slowed down 4× (e.g. iPhone slow motion). Time-based parameters are scaled accordingly.
- `JERSEY_COLORS` at the top of `src/track.py` — HSV ranges for each team, the referee and goalkeepers.
- `uv run python src/register.py video3.mov --kp-model models/pitch_kp/best.pt --debug-video` — writes `<name>_registration.mp4` with the projected pitch lines, to check the registration.
- `uv run python src/pick_keyframes.py video3.mov` — click landmarks manually on a few frames if no model is available.

## Training the keypoint detector

```bash
cd src
uv run --with SoccerNet python -c "from SoccerNet.Downloader import SoccerNetDownloader as D; D(LocalDirectory='$HOME/data/soccernet').downloadDataTask(task='calibration-2023', split=['train','valid','test'])"
uv run python -m kp.convert_soccernet --soccernet ~/data/soccernet/calibration-2023 --out ~/data/sn_keypoints
uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/pitch_kp --workers 4
```

SoccerNet-Calibration annotates line points, not keypoints. `kp/convert_soccernet.py` fits a homography per image from line intersections and line correspondences (normalized DLT), recovers center-circle landmarks geometrically when few intersections are visible, refines against every annotated point, and projects the 37 landmarks. About half of the images (non-main-camera views) are rejected.

To fine-tune on your own footage, `kp/annotate.py` pre-fills landmarks from the registration so you only correct them, and `kp/train.py --extra ... --init ...` mixes them into training.

## Project structure

```
src/
├── track.py              detection, tracking, re-linking, team vote, ball path, 2D export
├── register.py           per-frame pitch registration
├── visualize.py          2D pitch rendering
├── pitch.py              pitch model in meters (landmarks and lines)
├── pick_keyframes.py     manual keyframes (optional)
├── video_io.py           frame-accurate video reader
├── kp/                   pitch keypoint detector
│   ├── convert_soccernet.py
│   ├── model.py          MobileNetV3 U-Net + Coordinate Attention
│   ├── dataset.py        augmentation (flip with left/right swap, blur, JPEG, noise)
│   ├── train.py
│   ├── detect.py
│   └── annotate.py
└── legacy/               earlier single-goal registration
scripts/make_demo.py      builds the GIFs in docs/media
```

## Limitations

- The ball is not shown on the 2D pitch: a ground-plane homography cannot place an airborne ball, and frames where the ball is not seen cannot be reconstructed. The ball is shown in the tracking video instead.
- Jersey colors are configured with HSV ranges per match.
- Player identities (names) are not assigned yet; track IDs are anonymous.
- Slow-motion `.mov` files from phones are variable frame rate, so `time_s` in the CSV is approximate.

## Acknowledgements

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [supervision](https://github.com/roboflow/supervision), [trackers](https://github.com/roboflow/trackers), [mplsoccer](https://github.com/andrewRowlinson/mplsoccer), and the [SoccerNet](https://www.soccer-net.org/) Camera Calibration dataset. Model weights trained on SoccerNet are not distributed in this repository; check the SoccerNet terms before sharing them.
