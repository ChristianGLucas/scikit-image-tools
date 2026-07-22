import io

import numpy as np
from PIL import Image as PILImage

from gen.messages_pb2 import ThresholdInput, Image
from nodes.threshold import threshold


def _decode_mask(image_msg):
    return np.asarray(PILImage.open(io.BytesIO(image_msg.data)))


def test_threshold_otsu_separates_squares_from_background(ax, three_squares, image_msg):
    """Oracle: three_squares is a well-separated bimodal image (background=0,
    foreground=255) built with hand-computed geometry — Otsu must land
    strictly between the two levels, and the resulting mask must be
    foreground inside every known square and background outside all of them.
    """
    result = threshold(ax, ThresholdInput(image=image_msg(three_squares["arr"]), method="otsu"))
    assert result.error == ""
    assert 0 < result.threshold < 255
    assert result.method == "otsu"
    mask = _decode_mask(result.mask)
    for r0, c0, r1, c1 in three_squares["boxes"]:
        assert mask[r0 + 1, c0 + 1] > 0
    assert mask[0, 0] == 0


def test_threshold_li(ax, image_msg):
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[10:40, 10:40] = 200
    result = threshold(ax, ThresholdInput(image=image_msg(arr), method="li"))
    assert result.error == ""
    mask = _decode_mask(result.mask)
    assert mask[25, 25] > 0
    assert mask[2, 2] == 0


def test_threshold_yen(ax, image_msg):
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[10:40, 10:40] = 220
    result = threshold(ax, ThresholdInput(image=image_msg(arr), method="yen"))
    assert result.error == ""
    mask = _decode_mask(result.mask)
    assert mask[25, 25] > 0


def test_threshold_adaptive(ax, image_msg):
    arr = np.zeros((60, 60), dtype=np.uint8)
    arr[20:40, 20:40] = 200
    result = threshold(ax, ThresholdInput(image=image_msg(arr), method="adaptive", adaptive_block_size=15))
    assert result.error == ""
    assert result.method == "adaptive"
    mask = _decode_mask(result.mask)
    assert mask[30, 30] > 0


def test_threshold_defaults_to_otsu_when_method_empty(ax, image_msg):
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    result = threshold(ax, ThresholdInput(image=image_msg(arr)))
    assert result.error == ""
    assert result.method == "otsu"


def test_threshold_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((20, 20), dtype=np.uint8)
    result = threshold(ax, ThresholdInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""
    assert result.mask.data == b""


def test_threshold_malformed_image_returns_structured_error(ax):
    result = threshold(ax, ThresholdInput(image=Image(data=b"not an image")))
    assert result.error != ""


def test_threshold_url_rejected_no_network(ax):
    """This package is offline-only: a caller relying on `url` (as they
    might coming from image-tools/opencv-tools) must get a clear structured
    error, never a network call."""
    result = threshold(ax, ThresholdInput(image=Image(url="http://example.com/x.png")))
    assert result.error != ""
    assert "network" in result.error.lower()
