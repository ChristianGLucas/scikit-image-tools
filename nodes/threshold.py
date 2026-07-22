from gen.messages_pb2 import ThresholdInput, ThresholdResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray, encode_mask

_METHODS = {"otsu", "adaptive", "li", "yen"}


def threshold(ax: AxiomContext, input: ThresholdInput) -> ThresholdResult:
    """Computes a binarization threshold for a grayscale image and returns
    the binary mask (255 = foreground/above-threshold, 0 = background) plus
    the numeric threshold value used. `method` is one of "otsu" (global,
    default), "adaptive" (local mean, tunable block size/offset), "li"
    (minimum cross-entropy), or "yen" (maximum correlation). Color images are
    converted to grayscale (standard luminosity weights) first. Wraps
    scikit-image's skimage.filters threshold functions.
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)  # float64 in [0, 1]
        method = (input.method or "otsu").strip().lower()
        if method not in _METHODS:
            raise SkimgError(
                f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
            )

        from skimage.filters import threshold_otsu, threshold_local, threshold_li, threshold_yen

        if method == "otsu":
            t = threshold_otsu(gray)
            mask = gray > t
            threshold_value = float(t)
        elif method == "li":
            t = threshold_li(gray)
            mask = gray > t
            threshold_value = float(t)
        elif method == "yen":
            t = threshold_yen(gray)
            mask = gray > t
            threshold_value = float(t)
        else:  # adaptive
            block_size = input.adaptive_block_size or 11
            if block_size < 3:
                raise SkimgError("adaptive_block_size must be >= 3")
            if block_size % 2 == 0:
                block_size += 1  # skimage requires odd block sizes
            local_t = threshold_local(gray, block_size=block_size, offset=input.adaptive_offset)
            mask = gray > local_t
            threshold_value = float(local_t.mean())

        return ThresholdResult(
            mask=Image(**encode_mask(mask)),
            threshold=threshold_value,
            method=method,
        )
    except SkimgError as exc:
        return ThresholdResult(error=str(exc))
