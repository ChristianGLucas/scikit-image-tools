from gen.messages_pb2 import CornersInput, CornersResult, Point
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

_METHODS = {"harris", "shi_tomasi"}
DEFAULT_MAX_CORNERS = 500
HARD_MAX_CORNERS = 5000


def detect_corners(ax: AxiomContext, input: CornersInput) -> CornersResult:
    """Detects corner keypoints via the Harris (default) or Shi-Tomasi
    corner-response operators (skimage.feature.corner_harris /
    corner_shi_tomasi, peaks extracted with corner_peaks). Returns corner
    (row, col) pixel coordinates ranked strongest-first, at least
    `min_distance` pixels apart, capped at `max_corners`.
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
        max_corners = input.max_corners or DEFAULT_MAX_CORNERS
        max_corners = min(max(max_corners, 1), HARD_MAX_CORNERS)

        from skimage.feature import corner_harris, corner_shi_tomasi, corner_peaks

        response = corner_harris(gray) if method == "harris" else corner_shi_tomasi(gray)
        # Detect against a fixed internal hard cap (independent of the
        # caller's max_corners) so `total`/`truncated` are accurate for any
        # realistic image, while still bounding worst-case cost.
        # threshold_rel matters: without it, corner_peaks' underlying local-
        # max search treats a flat/textureless region (sky, a wall, any
        # uniform patch) as a plateau tied at the same response value and
        # reports a spurious evenly-spaced grid of "corners" across it, even
        # though the true corner response there is exactly zero. Requiring
        # each peak to be within 1% of the strongest response in the image
        # suppresses that artifact while still finding genuine corners.
        coords = corner_peaks(
            response, min_distance=min_distance, threshold_rel=0.01, num_peaks=HARD_MAX_CORNERS
        )

        total = len(coords)
        truncated = total > max_corners
        coords = coords[:max_corners]
        corners = [Point(row=float(r), col=float(c)) for r, c in coords]

        return CornersResult(corners=corners, count=total, truncated=truncated)
    except SkimgError as exc:
        return CornersResult(error=str(exc))
