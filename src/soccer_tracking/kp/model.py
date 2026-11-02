"""ピッチのキーポイント検出用の軽量 U-Net（キーポイントごとのヒートマップを出す）

エンコーダは torchvision の MobileNetV3（Large / Small）。重みは初期化から学習する（フルスクラッチ）。
デコーダの各段に Coordinate Attention を入れる。ピッチの目印は「画面のどの高さ・横位置にあるか」が
手がかりになるので、位置情報を保ったまま注意をかけられる CA と相性がよい。
出力は入力の 1/4 解像度のヒートマップ（キーポイントの数だけチャンネル）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

# エンコーダごとの、取り出す段の番号とチャンネル数（stride 4, 8, 16, 32）
ENCODERS = {
    "mnv3_large": (mobilenet_v3_large, [3, 6, 12, 16], [24, 40, 112, 960]),
    "mnv3_small": (mobilenet_v3_small, [1, 3, 8, 12], [16, 24, 48, 576]),
}


class CoordAtt(nn.Module):
    """Coordinate Attention（Hou et al., CVPR 2021）"""

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(8, ch // reduction)
        self.conv1 = nn.Conv2d(ch, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, 1)
        self.conv_w = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(dim=3, keepdim=True)  # (n, c, h, 1)
        xw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (n, c, w, 1)
        y = self.act(self.bn(self.conv1(torch.cat([xh, xw], dim=2))))
        yh, yw = torch.split(y, [h, w], dim=2)
        ah = torch.sigmoid(self.conv_h(yh))
        aw = torch.sigmoid(self.conv_w(yw.permute(0, 1, 3, 2)))
        return x * ah * aw


def conv_bn(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, use_ca=True):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(cin + cskip, cout), conv_bn(cout, cout))
        self.ca = CoordAtt(cout) if use_ca else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.ca(self.conv(torch.cat([x, skip], dim=1)))


class PitchKeypointNet(nn.Module):
    def __init__(self, num_kp, encoder="mnv3_large", width=64, use_ca=True):
        super().__init__()
        fn, self.taps, chs = ENCODERS[encoder]
        self.encoder = fn(weights=None).features
        self.reduce = nn.Sequential(conv_bn(chs[3], width * 4, 1), CoordAtt(width * 4) if use_ca else nn.Identity())
        self.up3 = UpBlock(width * 4, chs[2], width * 2, use_ca)  # → stride 16
        self.up2 = UpBlock(width * 2, chs[1], width, use_ca)  # → stride 8
        self.up1 = UpBlock(width, chs[0], width, use_ca)  # → stride 4
        self.head = nn.Sequential(conv_bn(width, width), nn.Conv2d(width, num_kp, 1))
        # 出力の初期値を「ほぼ0」（sigmoid で 0.01）にして、学習の最初が安定するようにする
        nn.init.constant_(self.head[-1].bias, -4.6)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.taps:
                feats.append(x)
        s4, s8, s16, s32 = feats
        y = self.reduce(s32)
        y = self.up3(y, s16)
        y = self.up2(y, s8)
        y = self.up1(y, s4)
        return self.head(y)  # ロジット（sigmoid 前）


def focal_loss(logits, target, alpha=2, beta=4):
    """CenterNet のヒートマップ用 focal loss（ガウスの山の中心が正例、周りは減点を弱めた負例）"""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred**alpha * (1 - target) ** beta * neg
    n_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos.clamp(min=1)
