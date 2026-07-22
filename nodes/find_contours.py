from gen.messages_pb2 import ContoursInput, ContoursResult, Contour, Point
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

DEFAULT_MAX_CONTOURS = 100
HARD_MAX_CONTOURS = 2000
DEFAULT_MAX_POINTS = 500
HARD_MAX_POINTS = 5000


def find_contours(ax: AxiomContext, input: ContoursInput) -> ContoursResult:
    """Finds iso-valued contour lines at a given intensity `level` (0-255
    grayscale units, default 128) via marching squares (skimage.measure.
    find_contours). Returns each contour's (row, col) point path and arc
    length, longest first, capped at `max_contours` contours and
    `max_points_per_contour` points per contour (longer contours are evenly
    subsampled, never head-truncated, so the shape is preserved).
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)  # [0, 1]

        level = input.level if input.level != 0 else 128.0
        level01 = level / 255.0
        max_contours = input.max_contours or DEFAULT_MAX_CONTOURS
        max_contours = min(max(max_contours, 1), HARD_MAX_CONTOURS)
        max_points = input.max_points_per_contour or DEFAULT_MAX_POINTS
        max_points = min(max(max_points, 2), HARD_MAX_POINTS)

        import numpy as np
        from skimage.measure import find_contours as _find_contours

        raw_contours = _find_contours(gray, level=level01)

        def length(pts: "np.ndarray") -> float:
            if len(pts) < 2:
                return 0.0
            diffs = np.diff(pts, axis=0)
            return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())

        scored = sorted(raw_contours, key=length, reverse=True)
        total = len(scored)
        truncated = total > max_contours
        scored = scored[:max_contours]

        contours = []
        for pts in scored:
            n = len(pts)
            if n > max_points:
                idx = np.linspace(0, n - 1, max_points).round().astype(int)
                sampled = pts[idx]
            else:
                sampled = pts
            contours.append(
                Contour(
                    points=[Point(row=float(r), col=float(c)) for r, c in sampled],
                    point_count=n,
                    length=length(pts),
                )
            )

        return ContoursResult(contours=contours, count=total, truncated=truncated)
    except SkimgError as exc:
        return ContoursResult(error=str(exc))
