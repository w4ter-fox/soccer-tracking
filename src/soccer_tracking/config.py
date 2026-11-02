"""設定値とクラスID

調整する値はすべて Config にまとめる。コマンドライン引数で一部を上書きでき、
関数には Config をそのまま渡す（グローバル変数を書き換えない）。
"""

from dataclasses import dataclass, field

# 内部のクラスID（ラベルには出さない。ボックスの色の決定にだけ使う）
TEAM_WHITE, TEAM_BLUE, REFEREE, GOALKEEPER, BALL, UNKNOWN = 0, 1, 2, 3, 4, 5
TEAM_NAMES = {"white": TEAM_WHITE, "blue": TEAM_BLUE, "referee": REFEREE, "gk": GOALKEEPER}

# クラスID順のボックスの色（RGB）
BOX_COLORS_RGB = [
    (255, 255, 255),  # 0: 白チーム
    (30, 90, 255),  # 1: 青チーム
    (255, 215, 0),  # 2: 審判（黄）
    (230, 30, 30),  # 3: GK（赤）
    (255, 140, 0),  # 4: ボール（オレンジ）
    (150, 150, 150),  # 5: 判定できない（灰）
]
BALL_BGR = (0, 140, 255)

# ユニフォームの色（HSV の範囲。OpenCV の H は 0〜180）。1つのクラスに複数の範囲を書ける
DEFAULT_JERSEY_COLORS = {
    TEAM_WHITE: [((0, 0, 150), (180, 45, 255))],
    TEAM_BLUE: [((95, 60, 40), (130, 255, 255))],
    REFEREE: [((18, 80, 80), (34, 255, 255))],
    GOALKEEPER: [((0, 110, 60), (10, 255, 255)), ((165, 110, 60), (180, 255, 255))],
}


@dataclass
class Config:
    # ---- 検出 ----
    model: str = "yolo11x.pt"
    imgsz: int = 1280
    person_conf: float = 0.1  # 低信頼の検出は ByteTrack が隠れた選手の追跡継続にだけ使う
    nms_iou: float = 0.8  # 高めにすると重なった選手の箱が NMS で消されにくい
    occlusion_ratio: float = 0.3  # 他の選手とこれ以上重なったフレームの色判定は数えない

    # ---- 時間 ----
    slowdown: int = 4  # 入力が実時間の何倍にスロー化されているか
    output_speed: str = "input"  # "input": 入力と同じ速度 / "realtime": slowdown フレームごとに書き出す
    output_crf: int = 18  # 出力動画（H.264）の画質

    # ---- ユニフォームの色 ----
    jersey_colors: dict = field(default_factory=lambda: dict(DEFAULT_JERSEY_COLORS))
    jersey_min_ratio: float = 0.08  # どの色もこの割合に届かなければ、そのフレームの色判定はしない

    # ---- チームの決定 ----
    team_min_votes: int = 8  # 自分の色判定だけで決めるのに必要な判定数（実時間換算のフレーム数）
    team_min_share: float = 0.6  # 最多の色の割合がこれ以上なら確定
    team_inherit_gap_s: float = 1.5  # 判定が足りないトラックは、この秒数以内に近くで途切れたトラックから引き継ぐ
    team_inherit_dist_m: float = 5.0
    team_inherit_dist_box: float = 1.5

    # ---- 途切れたトラックのつなぎ直し ----
    link_max_gap_s: float = 3.0
    link_vel_window_s: float = 0.5
    link_r0_m: float = 1.5
    link_r_per_s_m: float = 2.0
    link_r0_box: float = 1.0
    link_r_per_s_box: float = 1.0

    # ---- ボールの検出とオンライン追跡（切り出し探索の位置決め用）----
    ball_conf: float = 0.05
    ball_init_conf: float = 0.25
    ball_gate_px: float = 80
    ball_max_misses: int = 12  # 実時間換算のフレーム数
    ball_roi: int = 320
    ball_tiles: int = 2  # 画面を tiles×tiles に分けて拡大推論（0で無効）
    ball_tile_overlap: float = 0.2
    ball_trail: int = 15

    # ---- ボールの経路選択（全フレームの候補から動的計画法で選ぶ）----
    ball_keep_topk: int = 5
    ball_max_speed_px: float = 60  # 実時間1フレームあたり
    ball_link_gap: int = 8  # 実時間換算のフレーム数
    ball_link_bonus: float = 0.15
    ball_conf_offset: float = 0.15
    ball_jump_penalty: float = 0.5
    ball_static_px: float = 2.0
    ball_static_m: float = 0.05
    ball_spot_radius_m: float = 1.0  # ペナルティマーク・センターマークからこの距離以内の候補を捨てる
    ball_spot_keep_conf: float = 0.6

    # ---- ボールの弾道フィット ----
    ball_fit_tol_px: float = 6.0
    ball_fit_window: int = 12
    ball_seg_max_gap: int = 8  # 実時間換算のフレーム数
