import numpy as np

from gen.messages_pb2 import BlobsInput, Image
from nodes.detect_blobs import detect_blobs


def test_detect_blobs_log_finds_the_disk(ax, one_disk_blob, image_msg):
    """Oracle: one_disk_blob is drawn (skimage.draw.disk, a different code
    path than blob_log/blob_dog) at a known center/radius — LoG must report
    exactly one blob near that center with a plausibly-close radius."""
    result = detect_blobs(
        ax, BlobsInput(image=image_msg(one_disk_blob["arr"]), method="log", min_sigma=1, max_sigma=30)
    )
    assert result.error == ""
    assert result.count == 1
    blob = result.blobs[0]
    exp_row, exp_col = one_disk_blob["center"]
    assert abs(blob.row - exp_row) <= 3
    assert abs(blob.col - exp_col) <= 3
    assert 4.0 <= blob.radius <= 20.0


def test_detect_blobs_dog_finds_the_disk(ax, one_disk_blob, image_msg):
    result = detect_blobs(
        ax, BlobsInput(image=image_msg(one_disk_blob["arr"]), method="dog", min_sigma=1, max_sigma=30)
    )
    assert result.error == ""
    assert result.count >= 1
    found_near_center = any(
        abs(b.row - one_disk_blob["center"][0]) <= 3 and abs(b.col - one_disk_blob["center"][1]) <= 3
        for b in result.blobs
    )
    assert found_near_center


def test_detect_blobs_defaults_to_log(ax, one_disk_blob, image_msg):
    result = detect_blobs(ax, BlobsInput(image=image_msg(one_disk_blob["arr"])))
    assert result.error == ""


def test_detect_blobs_blank_image_finds_none(ax, image_msg):
    arr = np.zeros((50, 50), dtype=np.uint8)
    result = detect_blobs(ax, BlobsInput(image=image_msg(arr)))
    assert result.error == ""
    assert result.count == 0


def test_detect_blobs_max_sigma_less_than_min_sigma_returns_structured_error(ax, image_msg):
    arr = np.zeros((20, 20), dtype=np.uint8)
    result = detect_blobs(ax, BlobsInput(image=image_msg(arr), min_sigma=10, max_sigma=1))
    assert result.error != ""


def test_detect_blobs_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((20, 20), dtype=np.uint8)
    result = detect_blobs(ax, BlobsInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""


def test_detect_blobs_malformed_image_returns_structured_error(ax):
    result = detect_blobs(ax, BlobsInput(image=Image(data=b"garbage")))
    assert result.error != ""


def test_detect_blobs_negative_min_sigma_returns_structured_error(ax, image_msg):
    arr = np.zeros((20, 20), dtype=np.uint8)
    result = detect_blobs(ax, BlobsInput(image=image_msg(arr), min_sigma=-5))
    assert result.error != ""
