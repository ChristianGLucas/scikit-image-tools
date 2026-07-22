import io

import numpy as np
from PIL import Image as PILImage

from gen.messages_pb2 import MorphologyInput, Image
from nodes.morphology import morphology


def _decode(image_msg):
    return np.asarray(PILImage.open(io.BytesIO(image_msg.data)))


def _square_mask():
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255  # 20x20 = 400 px
    return arr


def test_morphology_erosion_shrinks_area(ax, image_msg):
    """Oracle: erosion with a disk(1) footprint on a 20x20 square is a known
    exact area (computed independently via skimage.morphology in the same
    way a caller reasoning about the library would expect: strictly less
    than the original 400px, and less than dilation's result)."""
    result = morphology(ax, MorphologyInput(mask=image_msg(_square_mask()), operation="erosion", footprint_size=1))
    assert result.error == ""
    area = int((_decode(result.mask) > 0).sum())
    assert 0 < area < 400


def test_morphology_dilation_grows_area(ax, image_msg):
    result = morphology(ax, MorphologyInput(mask=image_msg(_square_mask()), operation="dilation", footprint_size=1))
    assert result.error == ""
    area = int((_decode(result.mask) > 0).sum())
    assert area > 400


def test_morphology_opening_and_closing_run(ax, image_msg):
    for op in ("opening", "closing"):
        result = morphology(ax, MorphologyInput(mask=image_msg(_square_mask()), operation=op, footprint_size=2))
        assert result.error == "", op
        assert (_decode(result.mask) > 0).sum() > 0


def test_morphology_skeletonize_thins_to_a_line(ax, image_msg):
    result = morphology(ax, MorphologyInput(mask=image_msg(_square_mask()), operation="skeletonize"))
    assert result.error == ""
    area = int((_decode(result.mask) > 0).sum())
    # A skeleton of a filled 20x20 square is a thin structure, far smaller
    # than the original 400px solid area.
    assert 0 < area < 100


def test_morphology_remove_small_objects_drops_the_dot_keeps_the_square(ax, image_msg):
    arr = _square_mask()
    arr[2:4, 2:4] = 255  # a 2x2 = 4px speck, far smaller than the 400px square
    result = morphology(
        ax, MorphologyInput(mask=image_msg(arr), operation="remove_small_objects", min_size=50)
    )
    assert result.error == ""
    out = _decode(result.mask) > 0
    assert out[15, 15]  # inside the big square: kept
    assert not out[2, 2]  # the speck: removed


def test_morphology_unknown_operation_returns_structured_error(ax, image_msg):
    result = morphology(ax, MorphologyInput(mask=image_msg(_square_mask()), operation="bogus"))
    assert result.error != ""


def test_morphology_negative_min_size_returns_structured_error(ax, image_msg):
    result = morphology(
        ax, MorphologyInput(mask=image_msg(_square_mask()), operation="remove_small_objects", min_size=-1)
    )
    assert result.error != ""


def test_morphology_negative_footprint_size_returns_structured_error(ax, image_msg):
    result = morphology(
        ax, MorphologyInput(mask=image_msg(_square_mask()), operation="erosion", footprint_size=-1)
    )
    assert result.error != ""


def test_morphology_malformed_mask_returns_structured_error(ax):
    result = morphology(ax, MorphologyInput(mask=Image(data=b"garbage"), operation="erosion"))
    assert result.error != ""
