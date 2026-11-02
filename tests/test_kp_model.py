import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from soccer_tracking.kp.dataset import KEYS  # noqa: E402
from soccer_tracking.kp.model import PitchKeypointNet, focal_loss  # noqa: E402


@pytest.mark.parametrize("encoder", ["mnv3_large", "mnv3_small"])
def test_forward_shape(encoder):
    model = PitchKeypointNet(len(KEYS), encoder).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 352, 640))
    assert out.shape == (1, len(KEYS), 88, 160)


def test_focal_loss_prefers_correct_peak():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    good = torch.full((1, 1, 8, 8), -5.0)
    good[0, 0, 4, 4] = 5.0
    bad = torch.full((1, 1, 8, 8), -5.0)
    bad[0, 0, 1, 1] = 5.0
    assert focal_loss(good, target) < focal_loss(bad, target)
