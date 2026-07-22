from gen.messages_pb2 import DenoiseInput, DenoiseResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, encode_gray_u8, encode_rgb_u8

_METHODS = {"tv", "bilateral", "wavelet"}


def denoise(ax: AxiomContext, input: DenoiseInput) -> DenoiseResult:
    """Denoises an image via total-variation (skimage.restoration.
    denoise_tv_chambolle, default — edge-preserving smoothing, higher
    `weight` = smoother), bilateral (denoise_bilateral — edge-preserving,
    tunable `sigma_color`/`sigma_spatial`), or wavelet (denoise_wavelet —
    BayesShrink-thresholded) filtering. Works on grayscale or color input;
    returns the denoised image in the same channel layout.
    """
    try:
        arr = load_array(input.image)
        method = (input.method or "tv").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")

        import numpy as np
        from skimage.util import img_as_float
        from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet

        is_color = arr.ndim == 3
        work = arr[..., :3] if (is_color and arr.shape[-1] == 4) else arr
        work_f = img_as_float(work)
        channel_axis = -1 if is_color else None

        if method == "tv":
            weight = input.weight if input.weight > 0 else 0.1
            if weight <= 0 or weight > 10:
                raise SkimgError("weight must be in (0, 10]")
            result = denoise_tv_chambolle(work_f, weight=weight, channel_axis=channel_axis)
        elif method == "bilateral":
            sigma_spatial = input.sigma_spatial if input.sigma_spatial > 0 else 1.0
            sigma_color = input.sigma_color if input.sigma_color > 0 else None
            result = denoise_bilateral(
                work_f, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=channel_axis
            )
        else:
            result = denoise_wavelet(work_f, channel_axis=channel_axis)

        result = np.clip(result, 0.0, 1.0)
        out = Image(**encode_rgb_u8(result)) if is_color else Image(**encode_gray_u8(result))

        return DenoiseResult(image=out)
    except SkimgError as exc:
        return DenoiseResult(error=str(exc))
