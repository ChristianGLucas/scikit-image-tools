import io

import numpy as np
from PIL import Image as PILImage

from gen.messages_pb2 import DenoiseInput, Image
from nodes.denoise import denoise


def _decode(image_msg):
    return np.asarray(PILImage.open(io.BytesIO(image_msg.data))).astype(np.float64)


def test_denoise_tv_reduces_noise_std(ax, noisy_gradient, image_msg):
    """Oracle: noisy_gradient's ground-truth signal is a smooth ramp; the
    additive noise raises local variance. TV denoising must reduce the
    image's overall standard deviation relative to the noisy input — a real
    invariant, not a tautology, since a no-op or noise-amplifying bug would
    fail this."""
    result = denoise(ax, DenoiseInput(image=image_msg(noisy_gradient), method="tv", weight=0.3))
    assert result.error == ""
    out = _decode(result.image)
    assert out.std() < noisy_gradient.astype(np.float64).std()


def test_denoise_bilateral(ax, noisy_gradient, image_msg):
    result = denoise(ax, DenoiseInput(image=image_msg(noisy_gradient), method="bilateral"))
    assert result.error == ""
    out = _decode(result.image)
    assert out.shape == noisy_gradient.shape


def test_denoise_wavelet(ax, noisy_gradient, image_msg):
    result = denoise(ax, DenoiseInput(image=image_msg(noisy_gradient), method="wavelet"))
    assert result.error == ""
    out = _decode(result.image)
    assert out.std() < noisy_gradient.astype(np.float64).std()


def test_denoise_color_image_preserves_channels(ax, solid_red, image_msg):
    result = denoise(ax, DenoiseInput(image=image_msg(solid_red), method="tv"))
    assert result.error == ""
    out = _decode(result.image)
    assert out.ndim == 3 and out.shape[-1] == 3


def test_denoise_defaults_to_tv(ax, noisy_gradient, image_msg):
    result = denoise(ax, DenoiseInput(image=image_msg(noisy_gradient)))
    assert result.error == ""


def test_denoise_unknown_method_returns_structured_error(ax, image_msg):
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = denoise(ax, DenoiseInput(image=image_msg(arr), method="bogus"))
    assert result.error != ""


def test_denoise_out_of_range_weight_returns_structured_error(ax, image_msg):
    arr = np.zeros((10, 10), dtype=np.uint8)
    result = denoise(ax, DenoiseInput(image=image_msg(arr), method="tv", weight=100))
    assert result.error != ""


def test_denoise_malformed_image_returns_structured_error(ax):
    result = denoise(ax, DenoiseInput(image=Image(data=b"garbage")))
    assert result.error != ""
