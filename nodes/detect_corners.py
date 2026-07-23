from gen.messages_pb2 import CornersInput, CornersResult, Point
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

_METHODS = {"harris", "shi_tomasi"}


def detect_corners(ax: AxiomContext, input: CornersInput) -> CornersResult:
    """Detects corner keypoints via the Harris (default) or Shi-Tomasi
    corner-response operators (skimage.feature.corner_harris /
    corner_shi_tomasi, peaks extracted with corner_peaks). Returns every
    corner (row, col) pixel coordinate found, ranked strongest-first, at
    least `min_distance` pixels apart.
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)
        method = (input.method or "harris").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")
        min_distance = input.min_distance or 5
        if min_distance < 1:
            raise SkimgError("min_distance must be >= 1")

        from skimage.feature import corner_harris, corner_shi_tomasi, corner_peaks

        response = corner_harris(gray) if method == "harris" else corner_shi_tomasi(gray)
        # threshold_rel matters: without it, corner_peaks' underlying local-
        # max search treats a flat/textureless region (sky, a wall, any
        # uniform patch) as a plateau tied at the same response value and
        # reports a spurious evenly-spaced grid of "corners" across it, even
        # though the true corner response there is exactly zero. Requiring
        # each peak to be within 1% of the strongest response in the image
        # suppresses that artifact while still finding genuine corners.
        coords = corner_peaks(response, min_distance=min_distance, threshold_rel=0.01)

        total = len(coords)
        corners = [Point(row=float(r), col=float(c)) for r, c in coords]

        return CornersResult(corners=corners, count=total)
    except SkimgError as exc:
        return CornersResult(error=str(exc))
