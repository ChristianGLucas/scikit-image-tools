import numpy as np

from gen.messages_pb2 import CornersInput, Image
from nodes.detect_corners import detect_corners


def test_detect_corners_harris_finds_the_four_square_corners(ax, white_square_corners, image_msg):
    """Oracle: white_square_corners has four corners at exactly known pixel
    coordinates (20,20)/(20,79)/(79,20)/(79,79) — Harris must find all four,
    each within a couple pixels of the true location."""
    result = detect_corners(
        ax, CornersInput(image=image_msg(white_square_corners["arr"]), method="harris", min_distance=5)
    )
    assert result.error == ""
    assert result.count >= 4
    found = [(c.row, c.col) for c in result.corners]
    for exp_row, exp_col in white_square_corners["corners"]:
        assert any(abs(r - exp_row) <= 3 and abs(c - exp_col) <= 3 for r, c in found), (
            f"no detected corner near ({exp_row}, {exp_col}); found={found[:10]}"
        )


def test_detect_corners_shi_tomasi(ax, white_square_corners, image_msg):
    result = detect_corners(
        ax,
        CornersInput(image=image_msg(white_square_corners["arr"]), method="shi_tomasi", min_distance=5),
    )
    assert result.error == ""
    assert result.count >= 4


def test_detect_corners_max_corners_truncates(ax, white_square_corners, image_msg):
    result = detect_corners(
        ax,
        CornersInput(
            image=image_msg(white_square_corners["arr"]), method="harris", min_distance=1, max_corners=2
        ),
    )
    assert result.error == ""
    assert result.count > 2
    assert result.truncated is True
    assert len(result.corners) == 2


def test_detect_corners_flat_image_finds_none(ax, uniform_gray, image_msg):
    """Regression oracle: a perfectly flat/textureless region (e.g. sky, a
    wall) has a zero corner response everywhere. Without a relative-
    threshold, skimage's corner_peaks treats that all-tied plateau as a
    grid of "local maxima" and reports a spurious evenly-spaced set of fake
    corners — this asserts that artifact is suppressed.
    """
    result = detect_corners(ax, CornersInput(image=image_msg(uniform_gray)))
    assert result.error == ""
    assert result.count == 0
    assert list(result.corners) == []


def test_detect_corners_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = detect_corners(ax, CornersInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""


def test_detect_corners_malformed_image_returns_structured_error(ax):
    result = detect_corners(ax, CornersInput(image=Image(data=b"garbage")))
    assert result.error != ""
