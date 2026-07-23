from gen.messages_pb2 import ContoursInput, ContoursResult, Contour, Point
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

def find_contours(ax: AxiomContext, input: ContoursInput) -> ContoursResult:
    """Finds iso-valued contour lines at a given intensity `level` (0-255
    grayscale units, default 128) via marching squares (skimage.measure.
    find_contours). Returns every contour's (row, col) point path and arc
    length, longest first.
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)  # [0, 1]

        level = input.level if input.level != 0 else 128.0
        level01 = level / 255.0

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

        contours = []
        for pts in scored:
            contours.append(
                Contour(
                    points=[Point(row=float(r), col=float(c)) for r, c in pts],
                    point_count=len(pts),
                    length=length(pts),
                )
            )

        return ContoursResult(contours=contours, count=total)
    except SkimgError as exc:
        return ContoursResult(error=str(exc))
