import numpy as np

from gen.messages_pb2 import FelzenszwalbInput, Image
from nodes.segment_felzenszwalb import segment_felzenszwalb


def _half_red_half_blue():
    arr = np.zeros((60, 120, 3), dtype=np.uint8)
    arr[:, :60] = (255, 0, 0)
    arr[:, 60:] = (0, 0, 255)
    return arr


def test_segment_felzenszwalb_is_deterministic(ax, image_msg):
    """Felzenszwalb has no randomness at all — two calls with identical
    input/params must be byte-for-byte identical, no seed needed."""
    img = image_msg(_half_red_half_blue())
    r1 = segment_felzenszwalb(ax, FelzenszwalbInput(image=img, scale=100, sigma=0.5, min_size=20))
    r2 = segment_felzenszwalb(ax, FelzenszwalbInput(image=img, scale=100, sigma=0.5, min_size=20))
    assert r1.error == ""
    assert [(s.label, s.area) for s in r1.segments] == [(s.label, s.area) for s in r2.segments]


def test_segment_felzenszwalb_recovers_the_red_blue_split(ax, image_msg):
    result = segment_felzenszwalb(
        ax, FelzenszwalbInput(image=image_msg(_half_red_half_blue()), scale=200, sigma=0.5, min_size=50)
    )
    assert result.error == ""
    assert result.segment_count >= 1
    assert any(s.mean_r > 150 and s.mean_b < 50 for s in result.segments)
    assert any(s.mean_b > 150 and s.mean_r < 50 for s in result.segments)


def test_segment_felzenszwalb_negative_min_size_returns_structured_error(ax, image_msg):
    result = segment_felzenszwalb(
        ax, FelzenszwalbInput(image=image_msg(_half_red_half_blue()), min_size=-1)
    )
    assert result.error != ""


def test_segment_felzenszwalb_malformed_image_returns_structured_error(ax):
    result = segment_felzenszwalb(ax, FelzenszwalbInput(image=Image(data=b"garbage")))
    assert result.error != ""
