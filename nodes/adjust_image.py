from gen.messages_pb2 import AdjustInput, AdjustResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, encode_gray_u8, encode_rgb_u8

_METHODS = {"equalize", "clahe", "gamma"}


def adjust_image(ax: AxiomContext, input: AdjustInput) -> AdjustResult:
    """Adjusts image exposure via global histogram equalization
    (skimage.exposure.equalize_hist, default), CLAHE — contrast-limited
    adaptive histogram equalization (equalize_adapthist, tunable
    `clip_limit`), or gamma correction (adjust_gamma, `gamma` exponent).
    Works on grayscale or color input (each channel adjusted independently
    for color); returns the adjusted image in the same channel layout.
    """
    try:
        arr = load_array(input.image)
        method = (input.method or "equalize").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")

        import numpy as np
        from skimage.util import img_as_float
        from skimage.exposure import equalize_hist, equalize_adapthist, adjust_gamma

        is_color = arr.ndim == 3
        work = arr[..., :3] if (is_color and arr.shape[-1] == 4) else arr
        work_f = img_as_float(work)

        def apply_one(plane: "np.ndarray") -> "np.ndarray":
            if method == "equalize":
                return equalize_hist(plane)
            if method == "gamma":
                gamma = input.gamma if input.gamma != 0 else 1.0
                if gamma <= 0 or gamma > 10:
                    raise SkimgError("gamma must be in (0, 10]")
                return adjust_gamma(plane, gamma=gamma)
            clip_limit = input.clip_limit if input.clip_limit != 0 else 0.01
            if clip_limit <= 0 or clip_limit > 1:
                raise SkimgError("clip_limit must be in (0, 1]")
            kernel_size = input.kernel_size if input.kernel_size > 0 else None
            return equalize_adapthist(plane, clip_limit=clip_limit, kernel_size=kernel_size)

        if is_color:
            channels = [apply_one(work_f[..., i]) for i in range(work_f.shape[-1])]
            result = np.stack(channels, axis=-1)
        else:
            result = apply_one(work_f)

        result = np.clip(result, 0.0, 1.0)
        out = Image(**encode_rgb_u8(result)) if is_color else Image(**encode_gray_u8(result))

        return AdjustResult(image=out)
    except SkimgError as exc:
        return AdjustResult(error=str(exc))
