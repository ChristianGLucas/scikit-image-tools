import numpy as np

from gen.messages_pb2 import ContoursInput, Image
from nodes.find_contours import find_contours


def test_find_contours_square_perimeter_close_to_hand_computed(ax, image_msg):
    """Oracle: a filled 30x30 square has a hand-computed perimeter of 4*30 =
    120px; marching squares traces the mid-level boundary, which is close to
    but not exactly that (sub-pixel corner cutting) -- assert within a
    generous but real tolerance, and exactly one contour."""
    arr = np.zeros((50, 50), dtype=np.uint8)
    arr[10:40, 10:40] = 255
    result = find_contours(ax, ContoursInput(image=image_msg(arr), level=128))
    assert result.error == ""
    assert result.count == 1
    contour = result.contours[0]
    assert abs(contour.length - 120) < 10
    assert contour.point_count > 3


def test_find_contours_two_separate_squares_gives_two_contours(ax, image_msg):
    arr = np.zeros((50, 100), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    arr[10:30, 60:80] = 255
    result = find_contours(ax, ContoursInput(image=image_msg(arr), level=128))
    assert result.error == ""
    assert result.count == 2


def test_find_contours_returns_every_point_no_subsampling(ax, image_msg):
    arr = np.zeros((200, 200), dtype=np.uint8)
    arr[20:180, 20:180] = 255
    result = find_contours(ax, ContoursInput(image=image_msg(arr), level=128))
    assert result.error == ""
    contour = result.contours[0]
    assert contour.point_count > 10
    assert len(contour.points) == contour.point_count


def test_find_contours_blank_image_finds_none(ax, image_msg):
    arr = np.zeros((30, 30), dtype=np.uint8)
    result = find_contours(ax, ContoursInput(image=image_msg(arr), level=128))
    assert result.error == ""
    assert result.count == 0


def test_find_contours_returns_every_contour_no_truncation(ax, image_msg):
    arr = np.zeros((20, 400), dtype=np.uint8)
    for i in range(10):
        c = 5 + i * 40
        arr[5:15, c : c + 10] = 255
    result = find_contours(ax, ContoursInput(image=image_msg(arr), level=128))
    assert result.error == ""
    assert result.count == 10
    assert len(result.contours) == 10


def test_find_contours_malformed_image_returns_structured_error(ax):
    result = find_contours(ax, ContoursInput(image=Image(data=b"garbage")))
    assert result.error != ""
