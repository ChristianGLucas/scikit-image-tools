import numpy as np

from gen.messages_pb2 import StatsInput, Image
from nodes.image_statistics import image_statistics


def test_image_statistics_uniform_gray_has_exact_stats(ax, uniform_gray, image_msg):
    """Oracle: a perfectly uniform image (value=100 everywhere) has exact,
    hand-computable stats: mean=100, std=0, min=max=100, and Shannon entropy
    of a single-valued distribution is exactly 0 bits."""
    result = image_statistics(ax, StatsInput(image=image_msg(uniform_gray)))
    assert result.error == ""
    assert len(result.channels) == 1
    ch = result.channels[0]
    assert ch.channel == "gray"
    assert ch.mean == 100.0
    assert ch.std == 0.0
    assert ch.min == 100.0
    assert ch.max == 100.0
    assert result.entropy == 0.0


def test_image_statistics_color_image_has_per_channel_stats(ax, solid_red, image_msg):
    result = image_statistics(ax, StatsInput(image=image_msg(solid_red)))
    assert result.error == ""
    names = [c.channel for c in result.channels]
    assert names == ["r", "g", "b"]
    by_name = {c.channel: c for c in result.channels}
    assert by_name["r"].mean == 255.0
    assert by_name["g"].mean == 0.0
    assert by_name["b"].mean == 0.0


def test_image_statistics_histogram_sums_to_pixel_count(ax, uniform_gray, image_msg):
    result = image_statistics(ax, StatsInput(image=image_msg(uniform_gray), histogram_bins=16))
    assert result.error == ""
    ch = result.channels[0]
    assert len(ch.histogram) == 16
    assert sum(ch.histogram) == uniform_gray.size


def test_image_statistics_negative_bins_returns_structured_error(ax, uniform_gray, image_msg):
    result = image_statistics(ax, StatsInput(image=image_msg(uniform_gray), histogram_bins=-1))
    assert result.error != ""


def test_image_statistics_malformed_image_returns_structured_error(ax):
    result = image_statistics(ax, StatsInput(image=Image(data=b"garbage")))
    assert result.error != ""
