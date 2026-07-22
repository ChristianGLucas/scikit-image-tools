import numpy as np

from gen.messages_pb2 import RegionPropsInput, Image
from nodes.region_props import region_props


def test_region_props_finds_three_regions_with_correct_area_and_centroid(ax, three_squares, image_msg):
    """The flagship correctness oracle: three_squares' areas and centroids
    are HAND-COMPUTED from the exact rectangle coordinates (independent of
    skimage.measure) — regionprops must reproduce them exactly for
    axis-aligned rectangles.
    """
    result = region_props(ax, RegionPropsInput(mask=image_msg(three_squares["arr"])))
    assert result.error == ""
    assert result.count == 3
    assert not result.truncated
    assert len(result.regions) == 3

    got = sorted(result.regions, key=lambda r: r.area)
    expected = sorted(zip(three_squares["areas"], three_squares["centroids"]), key=lambda x: x[0])
    for region, (exp_area, (exp_row, exp_col)) in zip(got, expected):
        assert region.area == exp_area
        assert abs(region.centroid_row - exp_row) < 1e-6
        assert abs(region.centroid_col - exp_col) < 1e-6
        # Axis-aligned rectangles are not circles: solidity is exactly 1
        # (a rectangle's convex hull is itself).
        assert abs(region.solidity - 1.0) < 1e-6


def test_region_props_bbox_matches_hand_computed_box(ax, three_squares, image_msg):
    result = region_props(ax, RegionPropsInput(mask=image_msg(three_squares["arr"])))
    boxes = sorted(three_squares["boxes"], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    regions = sorted(result.regions, key=lambda r: r.area)
    for region, (r0, c0, r1, c1) in zip(regions, boxes):
        assert region.bbox_min_row == r0
        assert region.bbox_min_col == c0
        assert region.bbox_max_row == r1
        assert region.bbox_max_col == c1


def test_region_props_min_area_filters_small_regions(ax, three_squares, image_msg):
    # three_squares' areas are [100, 2400, 256]; min_area=150 excludes only
    # the 100px square, keeping the 2400px and 256px ones.
    result = region_props(ax, RegionPropsInput(mask=image_msg(three_squares["arr"]), min_area=150))
    assert result.error == ""
    assert result.count == 2
    areas = sorted(r.area for r in result.regions)
    assert areas == [256.0, 2400.0]


def test_region_props_intensity_image_mean(ax, image_msg):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    intensity = np.zeros((30, 30), dtype=np.uint8)
    intensity[5:15, 5:15] = 200
    result = region_props(
        ax, RegionPropsInput(mask=image_msg(mask), intensity_image=image_msg(intensity))
    )
    assert result.error == ""
    assert len(result.regions) == 1
    assert abs(result.regions[0].mean_intensity - 200.0) < 1e-6


def test_region_props_intensity_shape_mismatch_returns_structured_error(ax, image_msg):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    intensity = np.zeros((10, 10), dtype=np.uint8)
    result = region_props(
        ax, RegionPropsInput(mask=image_msg(mask), intensity_image=image_msg(intensity))
    )
    assert result.error != ""


def test_region_props_empty_mask_returns_zero_regions(ax, image_msg):
    mask = np.zeros((20, 20), dtype=np.uint8)
    result = region_props(ax, RegionPropsInput(mask=image_msg(mask)))
    assert result.error == ""
    assert result.count == 0
    assert list(result.regions) == []


def test_region_props_max_regions_truncates(ax, image_msg):
    # Ten separated 2x2 dots.
    mask = np.zeros((20, 200), dtype=np.uint8)
    for i in range(10):
        c = 5 + i * 20
        mask[5:7, c : c + 2] = 255
    result = region_props(ax, RegionPropsInput(mask=image_msg(mask), max_regions=3))
    assert result.error == ""
    assert result.count == 10
    assert result.truncated is True
    assert len(result.regions) == 3


def test_region_props_malformed_mask_returns_structured_error(ax):
    result = region_props(ax, RegionPropsInput(mask=Image(data=b"garbage")))
    assert result.error != ""
