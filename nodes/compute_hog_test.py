import numpy as np

from gen.messages_pb2 import HogInput, Image
from nodes.compute_hog import compute_hog


def test_compute_hog_feature_count_matches_formula(ax, image_msg):
    """Oracle: HOG's output length follows a closed-form formula from the
    image size and params (independent of skimage's internals) —
    n_blocks_row * n_blocks_col * cells_per_block^2 * orientations."""
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    result = compute_hog(
        ax, HogInput(image=image_msg(arr), orientations=9, pixels_per_cell=8, cells_per_block=2)
    )
    assert result.error == ""
    n_cells = 64 // 8
    n_blocks = n_cells - 2 + 1
    expected = n_blocks * n_blocks * 2 * 2 * 9
    assert result.feature_count == expected
    assert len(result.features) == expected
    assert all(f >= 0.0 for f in result.features)  # HOG features are non-negative by construction


def test_compute_hog_visualize_returns_same_size_image(ax, image_msg):
    rng = np.random.default_rng(1)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    result = compute_hog(ax, HogInput(image=image_msg(arr), visualize=True))
    assert result.error == ""
    assert result.visualization.width == 64
    assert result.visualization.height == 64
    assert len(result.visualization.data) > 0


def test_compute_hog_no_visualize_leaves_visualization_empty(ax, image_msg):
    arr = np.zeros((32, 32), dtype=np.uint8)
    result = compute_hog(ax, HogInput(image=image_msg(arr), visualize=False))
    assert result.error == ""
    assert result.visualization.data == b""


def test_compute_hog_too_small_for_cell_size_returns_structured_error(ax, image_msg):
    arr = np.zeros((4, 4), dtype=np.uint8)
    result = compute_hog(ax, HogInput(image=image_msg(arr), pixels_per_cell=8, cells_per_block=2))
    assert result.error != ""


def test_compute_hog_malformed_image_returns_structured_error(ax):
    result = compute_hog(ax, HogInput(image=Image(data=b"garbage")))
    assert result.error != ""


def test_compute_hog_negative_orientations_returns_structured_error(ax, image_msg):
    rng = np.random.default_rng(2)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    result = compute_hog(ax, HogInput(image=image_msg(arr), orientations=-5))
    assert result.error != ""
