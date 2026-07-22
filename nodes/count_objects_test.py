from gen.messages_pb2 import CountObjectsInput, Image
from nodes.count_objects import count_objects


def test_count_objects_counts_three_squares(ax, three_squares, image_msg):
    result = count_objects(ax, CountObjectsInput(image=image_msg(three_squares["arr"])))
    assert result.error == ""
    assert result.count == 3
    assert abs(result.total_area - sum(three_squares["areas"])) < 1e-6
    assert abs(result.mean_area - sum(three_squares["areas"]) / 3) < 1e-6
    assert result.min_area_found == min(three_squares["areas"])
    assert result.max_area_found == max(three_squares["areas"])


def test_count_objects_min_area_excludes_small_square(ax, three_squares, image_msg):
    # three_squares' areas are [100, 2400, 256]; min_area=150 excludes only
    # the 100px square, keeping the 2400px and 256px ones.
    result = count_objects(
        ax, CountObjectsInput(image=image_msg(three_squares["arr"]), min_area=150)
    )
    assert result.error == ""
    assert result.count == 2


def test_count_objects_empty_image_counts_zero(ax, image_msg):
    import numpy as np

    arr = np.zeros((20, 20), dtype=np.uint8)
    result = count_objects(ax, CountObjectsInput(image=image_msg(arr)))
    assert result.error == ""
    assert result.count == 0
    assert result.total_area == 0.0


def test_count_objects_unknown_method_returns_structured_error(ax, image_msg):
    import numpy as np

    arr = np.zeros((10, 10), dtype=np.uint8)
    result = count_objects(ax, CountObjectsInput(image=image_msg(arr), threshold_method="bogus"))
    assert result.error != ""


def test_count_objects_malformed_image_returns_structured_error(ax):
    result = count_objects(ax, CountObjectsInput(image=Image(data=b"garbage")))
    assert result.error != ""
