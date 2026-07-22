from gen.messages_pb2 import CountObjectsInput, CountObjectsResult
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

_METHODS = {"otsu", "li", "yen"}


def count_objects(ax: AxiomContext, input: CountObjectsInput) -> CountObjectsResult:
    """Thresholds a grayscale image (Otsu/Li/Yen) and counts the resulting
    connected components in a single call — a lightweight convenience over
    RegionProps for when only the count and aggregate area statistics are
    needed, not the full per-region table. `min_area` excludes small
    components (e.g. noise) from the count. Wraps skimage.filters +
    skimage.measure.label.
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)
        method = (input.threshold_method or "otsu").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown threshold_method {method!r}; expected one of {sorted(_METHODS)}")

        from skimage.filters import threshold_otsu, threshold_li, threshold_yen
        from skimage.measure import label, regionprops

        if method == "otsu":
            t = threshold_otsu(gray)
        elif method == "li":
            t = threshold_li(gray)
        else:
            t = threshold_yen(gray)
        mask = gray > t

        connectivity = input.connectivity or 2
        if connectivity not in (1, 2):
            raise SkimgError("connectivity must be 1 or 2")
        min_area = input.min_area or 0
        if min_area < 0:
            raise SkimgError("min_area must be >= 0")

        labeled = label(mask, connectivity=connectivity)
        props = regionprops(labeled)
        if min_area:
            props = [p for p in props if p.area >= min_area]

        areas = [float(p.area) for p in props]
        count = len(areas)
        total_area = sum(areas)
        mean_area = (total_area / count) if count else 0.0
        min_area_found = min(areas) if areas else 0.0
        max_area_found = max(areas) if areas else 0.0

        return CountObjectsResult(
            count=count,
            threshold=float(t),
            total_area=total_area,
            mean_area=mean_area,
            min_area_found=min_area_found,
            max_area_found=max_area_found,
        )
    except SkimgError as exc:
        return CountObjectsResult(error=str(exc))
