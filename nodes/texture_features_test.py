import numpy as np

from gen.messages_pb2 import TextureInput, Image
from nodes.texture_features import texture_features


def test_texture_features_uniform_image_matches_textbook_glcm_values(ax, uniform_gray, image_msg):
    """Oracle: on a perfectly uniform image, the GLCM has all its mass on
    the diagonal at one gray level — textbook Haralick values follow
    directly: contrast=0, homogeneity=1, energy=1, ASM=1, dissimilarity=0.
    scikit-image's graycoprops defines correlation as exactly 1.0 (not NaN)
    in this zero-variance limit.
    """
    result = texture_features(ax, TextureInput(image=image_msg(uniform_gray)))
    assert result.error == ""
    assert abs(result.contrast - 0.0) < 1e-9
    assert abs(result.dissimilarity - 0.0) < 1e-9
    assert abs(result.homogeneity - 1.0) < 1e-9
    assert abs(result.energy - 1.0) < 1e-9
    assert abs(result.asm - 1.0) < 1e-9
    assert abs(result.correlation - 1.0) < 1e-9


def test_texture_features_checkerboard_has_high_contrast(ax, image_msg):
    """A fine black/white checkerboard has neighboring pixels at opposite
    extremes -> GLCM contrast must be large and homogeneity/energy small,
    the opposite of the uniform-image case above -- a real differential
    oracle."""
    arr = np.indices((32, 32)).sum(axis=0) % 2
    arr = (arr * 255).astype(np.uint8)
    result = texture_features(ax, TextureInput(image=image_msg(arr), distances=[1], angles=[0.0]))
    assert result.error == ""
    assert result.contrast > 100
    assert result.homogeneity < 0.5


def test_texture_features_custom_distances_and_angles(ax, uniform_gray, image_msg):
    result = texture_features(
        ax, TextureInput(image=image_msg(uniform_gray), distances=[1, 2], angles=[0.0, 1.5708])
    )
    assert result.error == ""


def test_texture_features_invalid_levels_returns_structured_error(ax, uniform_gray, image_msg):
    result = texture_features(ax, TextureInput(image=image_msg(uniform_gray), levels=1))
    assert result.error != ""


def test_texture_features_malformed_image_returns_structured_error(ax):
    result = texture_features(ax, TextureInput(image=Image(data=b"garbage")))
    assert result.error != ""
