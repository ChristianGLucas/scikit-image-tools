import numpy as np

from gen.messages_pb2 import EdgeInput, Image
from nodes.detect_edges import detect_edges


def test_detect_edges_canny_finds_the_step_column(ax, vertical_step_edge, image_msg):
    """Oracle: vertical_step_edge is a hard black|white step at column 40 —
    Canny must mark that column as edge and a column far from it as not."""
    result = detect_edges(ax, EdgeInput(image=image_msg(vertical_step_edge), method="canny"))
    assert result.error == ""
    assert result.method == "canny"
    assert 0.0 < result.edge_density < 0.5

    import io
    from PIL import Image as PILImage

    edge_arr = np.asarray(PILImage.open(io.BytesIO(result.edge_map.data)))
    assert edge_arr[30, 40] > 0 or edge_arr[30, 39] > 0 or edge_arr[30, 41] > 0
    assert edge_arr[30, 5] == 0
    assert edge_arr[30, 75] == 0


def test_detect_edges_sobel_default_method(ax, vertical_step_edge, image_msg):
    result = detect_edges(ax, EdgeInput(image=image_msg(vertical_step_edge)))
    assert result.error == ""
    assert result.method == "sobel"
    assert result.edge_map.width == vertical_step_edge.shape[1]
    assert result.edge_map.height == vertical_step_edge.shape[0]


def test_detect_edges_scharr(ax, vertical_step_edge, image_msg):
    result = detect_edges(ax, EdgeInput(image=image_msg(vertical_step_edge), method="scharr"))
    assert result.error == ""
    assert result.method == "scharr"
    assert result.edge_density > 0.0


def test_detect_edges_uniform_image_has_zero_gradient_edges(ax, uniform_gray, image_msg):
    result = detect_edges(ax, EdgeInput(image=image_msg(uniform_gray), method="sobel"))
    assert result.error == ""
    # A perfectly flat image has zero gradient everywhere.
    assert result.edge_density == 0.0


def test_detect_edges_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = detect_edges(ax, EdgeInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""


def test_detect_edges_malformed_image_returns_structured_error(ax):
    result = detect_edges(ax, EdgeInput(image=Image(data=b"garbage")))
    assert result.error != ""
