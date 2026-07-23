import numpy as np

from gen.messages_pb2 import SlicInput, Image
from nodes.segment_slic import segment_slic


def _half_red_half_blue():
    arr = np.zeros((60, 120, 3), dtype=np.uint8)
    arr[:, :60] = (255, 0, 0)
    arr[:, 60:] = (0, 0, 255)
    return arr


def test_segment_slic_is_deterministic_for_a_fixed_seed(ax, image_msg):
    """Determinism is the contract this node explicitly promises for a
    caller-supplied seed: two identical calls must return identical
    segments, in the same order, with identical stats."""
    img = image_msg(_half_red_half_blue())
    r1 = segment_slic(ax, SlicInput(image=img, n_segments=20, compactness=10, seed=7))
    r2 = segment_slic(ax, SlicInput(image=img, n_segments=20, compactness=10, seed=7))
    assert r1.error == ""
    assert r1.segment_count == r2.segment_count
    assert [(s.label, s.area, s.mean_r) for s in r1.segments] == [
        (s.label, s.area, s.mean_r) for s in r2.segments
    ]


def test_segment_slic_recovers_the_red_blue_split(ax, image_msg):
    """Correctness oracle: on a hard left/right red|blue split, SLIC's
    segments must reflect that split — some segment's mean color must be
    strongly red-dominant and another strongly blue-dominant."""
    result = segment_slic(ax, SlicInput(image=image_msg(_half_red_half_blue()), n_segments=8, seed=1))
    assert result.error == ""
    assert result.segment_count > 0
    assert any(s.mean_r > 150 and s.mean_b < 50 for s in result.segments)
    assert any(s.mean_b > 150 and s.mean_r < 50 for s in result.segments)


def test_segment_slic_returns_every_segment_no_truncation(ax, image_msg):
    result = segment_slic(ax, SlicInput(image=image_msg(_half_red_half_blue()), n_segments=20))
    assert result.error == ""
    assert result.segment_count > 0
    assert len(result.segments) == result.segment_count


def test_segment_slic_rejects_out_of_range_n_segments(ax, image_msg):
    result = segment_slic(ax, SlicInput(image=image_msg(_half_red_half_blue()), n_segments=1))
    assert result.error != ""


def test_segment_slic_malformed_image_returns_structured_error(ax):
    result = segment_slic(ax, SlicInput(image=Image(data=b"garbage")))
    assert result.error != ""


def test_segment_slic_negative_compactness_returns_structured_error(ax, image_msg):
    result = segment_slic(ax, SlicInput(image=image_msg(_half_red_half_blue()), compactness=-1))
    assert result.error != ""
