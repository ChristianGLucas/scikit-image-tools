from gen.messages_pb2 import WatershedInput, WatershedResult, SegmentStat
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, segment_stats

DEFAULT_MAX_SEGMENTS_RETURNED = 2000
HARD_MAX_SEGMENTS_RETURNED = 5000


def segment_watershed(ax: AxiomContext, input: WatershedInput) -> WatershedResult:
    """Separates touching objects in a binary mask via marker-based
    watershed: computes the Euclidean distance transform, seeds markers at
    its local maxima (at least `min_distance` pixels apart), and floods from
    those markers within the mask (skimage.segmentation.watershed). This is
    the standard technique for splitting adjoining blobs (e.g. touching
    cells) that a plain connected-components label would merge into one
    region — feed it the `mask` output of Threshold. Fully deterministic for
    a fixed input and `min_distance`. Returns the segment count and
    per-segment statistics, capped at `max_segments_returned`.
    """
    try:
        mask_arr = load_array(input.mask, allow_color=False)
        mask = mask_arr > 0

        min_distance = input.min_distance or 10
        if min_distance < 1:
            raise SkimgError("min_distance must be >= 1")
        max_segments_returned = input.max_segments_returned or DEFAULT_MAX_SEGMENTS_RETURNED
        max_segments_returned = min(max(max_segments_returned, 1), HARD_MAX_SEGMENTS_RETURNED)

        import numpy as np
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed

        if not mask.any():
            return WatershedResult(segment_count=0, segments=[], truncated=False)

        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
        markers_mask = np.zeros(distance.shape, dtype=bool)
        if coords.size:
            markers_mask[tuple(coords.T)] = True
        else:
            # No distinct peaks found (e.g. a thin/uniform blob) — treat the
            # whole mask as one marker so the result is still one segment,
            # not zero.
            markers_mask[mask] = True
        markers, _ = ndi.label(markers_mask)
        labels = watershed(-distance, markers, mask=mask)

        rows, total, truncated = segment_stats(labels, mask_arr, max_segments_returned)
        segments = [SegmentStat(**r) for r in rows]
        return WatershedResult(segment_count=total, segments=segments, truncated=truncated)
    except SkimgError as exc:
        return WatershedResult(error=str(exc))
