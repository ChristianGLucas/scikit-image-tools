from gen.messages_pb2 import EdgeInput, EdgeResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray, encode_mask, encode_gray_u8, rescale_to_u8

_METHODS = {"canny", "sobel", "scharr"}


def detect_edges(ax: AxiomContext, input: EdgeInput) -> EdgeResult:
    """Detects edges via the Canny detector (skimage.feature.canny, returns a
    binary edge mask) or the Sobel/Scharr gradient-magnitude operators
    (skimage.filters.sobel / scharr, returns a normalized gradient map).
    `method` is one of "canny", "sobel" (default), "scharr". For "canny",
    `sigma` controls pre-smoothing and `low_threshold`/`high_threshold`
    (both 0 = library auto-selected) set the hysteresis thresholds. Returns
    the edge/gradient map as a single-channel image plus the overall edge
    density in [0, 1].
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)
        method = (input.method or "sobel").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")

        if method == "canny":
            from skimage.feature import canny

            sigma = input.sigma if input.sigma > 0 else 1.0
            low = input.low_threshold if input.low_threshold > 0 else None
            high = input.high_threshold if input.high_threshold > 0 else None
            edges = canny(gray, sigma=sigma, low_threshold=low, high_threshold=high)
            density = float(edges.mean())
            out = Image(**encode_mask(edges))
        else:
            from skimage.filters import sobel, scharr

            fn = sobel if method == "sobel" else scharr
            grad = fn(gray)
            density = float((grad > grad.mean()).mean())
            out = Image(**encode_gray_u8(rescale_to_u8(grad)))

        return EdgeResult(edge_map=out, edge_density=density, method=method)
    except SkimgError as exc:
        return EdgeResult(error=str(exc))
