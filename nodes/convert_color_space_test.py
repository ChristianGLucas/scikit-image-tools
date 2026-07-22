import io

import numpy as np
from PIL import Image as PILImage

from gen.messages_pb2 import ColorConvertInput, Image
from nodes.convert_color_space import convert_color_space


def test_convert_color_space_gray_matches_standard_luma_formula(ax, solid_red, image_msg):
    """Oracle: the ITU-R BT.601 luma formula for pure red (255,0,0) gives
    0.2125*255 = 54.19 — an exact, independent-of-skimage expected value."""
    result = convert_color_space(ax, ColorConvertInput(image=image_msg(solid_red), target="gray"))
    assert result.error == ""
    assert result.target == "gray"
    out = np.asarray(PILImage.open(io.BytesIO(result.image.data)))
    assert out.ndim == 2
    assert abs(int(out[0, 0]) - 54) <= 1


def test_convert_color_space_hsv_uniform_image_is_uniform_output(ax, solid_red, image_msg):
    result = convert_color_space(ax, ColorConvertInput(image=image_msg(solid_red), target="hsv"))
    assert result.error == ""
    assert result.target == "hsv"
    out = np.asarray(PILImage.open(io.BytesIO(result.image.data)))
    assert out.ndim == 3 and out.shape[-1] == 3
    # A perfectly uniform input color has a constant value on every HSV
    # channel; the documented rescale-constant-to-zero behavior applies.
    assert (out == out[0, 0]).all()


def test_convert_color_space_lab_runs_and_returns_three_channels(ax, solid_red, image_msg):
    result = convert_color_space(ax, ColorConvertInput(image=image_msg(solid_red), target="lab"))
    assert result.error == ""
    out = np.asarray(PILImage.open(io.BytesIO(result.image.data)))
    assert out.shape == (40, 40, 3)


def test_convert_color_space_defaults_to_gray(ax, solid_red, image_msg):
    result = convert_color_space(ax, ColorConvertInput(image=image_msg(solid_red)))
    assert result.error == ""
    assert result.target == "gray"


def test_convert_color_space_unknown_target_returns_structured_error(ax, solid_red, image_msg):
    result = convert_color_space(ax, ColorConvertInput(image=image_msg(solid_red), target="bogus"))
    assert result.error != ""


def test_convert_color_space_malformed_image_returns_structured_error(ax):
    result = convert_color_space(ax, ColorConvertInput(image=Image(data=b"garbage")))
    assert result.error != ""
