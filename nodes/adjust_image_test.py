import io

import numpy as np
from PIL import Image as PILImage

from gen.messages_pb2 import AdjustInput, Image
from nodes.adjust_image import adjust_image


def _decode(image_msg):
    return np.asarray(PILImage.open(io.BytesIO(image_msg.data)))


def test_adjust_image_gamma_matches_hand_computed_formula(ax, image_msg):
    """Oracle: gamma correction is output = (input/255)^gamma * 255 — a
    closed-form, independent-of-skimage formula. For value=128, gamma=2.0:
    (128/255)^2 * 255 = 64.25.
    """
    arr = np.full((20, 20), 128, dtype=np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="gamma", gamma=2.0))
    assert result.error == ""
    out = _decode(result.image)
    assert abs(int(out[0, 0]) - 64) <= 1


def test_adjust_image_gamma_one_is_near_identity(ax, image_msg):
    arr = np.full((20, 20), 128, dtype=np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="gamma", gamma=1.0))
    assert result.error == ""
    out = _decode(result.image)
    assert abs(int(out[0, 0]) - 128) <= 1


def test_adjust_image_equalize_runs(ax, image_msg):
    rng = np.random.default_rng(3)
    arr = (rng.random((40, 40)) * 255).astype(np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="equalize"))
    assert result.error == ""
    out = _decode(result.image)
    assert out.shape == arr.shape


def test_adjust_image_clahe_runs(ax, image_msg):
    rng = np.random.default_rng(4)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="clahe", clip_limit=0.02))
    assert result.error == ""


def test_adjust_image_color_preserves_channels(ax, solid_red, image_msg):
    result = adjust_image(ax, AdjustInput(image=image_msg(solid_red), method="gamma", gamma=1.5))
    assert result.error == ""
    out = _decode(result.image)
    assert out.shape == solid_red.shape


def test_adjust_image_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""


def test_adjust_image_out_of_range_gamma_returns_structured_error(ax, image_msg):
    # gamma=-1 is non-positive, and (like the rest of this package's optional
    # numeric params) is treated as "unset" and replaced by the default —
    # 15 is explicitly out-of-range (cap is 10) so it cannot be defaulted away.
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = adjust_image(ax, AdjustInput(image=image_msg(arr), method="gamma", gamma=15))
    assert result.error != ""


def test_adjust_image_malformed_image_returns_structured_error(ax):
    result = adjust_image(ax, AdjustInput(image=Image(data=b"garbage")))
    assert result.error != ""
