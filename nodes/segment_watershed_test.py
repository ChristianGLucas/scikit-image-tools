import numpy as np

from gen.messages_pb2 import WatershedInput, Image
from nodes.segment_watershed import segment_watershed


def test_segment_watershed_splits_two_touching_circles(ax, two_touching_circles, image_msg):
    """The correctness oracle: two_touching_circles is ONE connected
    component under plain connected-components labeling (the disks touch),
    but topologically has two distance-transform maxima — watershed must
    report 2 segments, proving it actually separated them rather than just
    relabeling the single blob.
    """
    result = segment_watershed(
        ax, WatershedInput(mask=image_msg(two_touching_circles), min_distance=15)
    )
    assert result.error == ""
    assert result.segment_count == 2
    assert len(result.segments) == 2
    # Sanity: both segments have a plausible area (not a 1-pixel sliver).
    for seg in result.segments:
        assert seg.area > 100


def test_segment_watershed_is_deterministic(ax, two_touching_circles, image_msg):
    img = image_msg(two_touching_circles)
    r1 = segment_watershed(ax, WatershedInput(mask=img, min_distance=15))
    r2 = segment_watershed(ax, WatershedInput(mask=img, min_distance=15))
    assert [(s.label, s.area, s.centroid_row, s.centroid_col) for s in r1.segments] == [
        (s.label, s.area, s.centroid_row, s.centroid_col) for s in r2.segments
    ]


def test_segment_watershed_empty_mask_returns_zero_segments(ax, image_msg):
    mask = np.zeros((20, 20), dtype=np.uint8)
    result = segment_watershed(ax, WatershedInput(mask=image_msg(mask)))
    assert result.error == ""
    assert result.segment_count == 0


def test_segment_watershed_single_blob_is_one_segment(ax, image_msg):
    from skimage.draw import disk

    mask = np.zeros((60, 60), dtype=np.uint8)
    rr, cc = disk((30, 30), 15, shape=mask.shape)
    mask[rr, cc] = 255
    result = segment_watershed(ax, WatershedInput(mask=image_msg(mask), min_distance=15))
    assert result.error == ""
    assert result.segment_count == 1


def test_segment_watershed_negative_min_distance_returns_structured_error(ax, image_msg):
    # 0 is the documented "use the default" sentinel (proto3 can't
    # distinguish unset from 0); a genuinely invalid value is negative.
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = segment_watershed(ax, WatershedInput(mask=image_msg(mask), min_distance=-1))
    assert result.error != ""


def test_segment_watershed_malformed_mask_returns_structured_error(ax):
    result = segment_watershed(ax, WatershedInput(mask=Image(data=b"garbage")))
    assert result.error != ""
