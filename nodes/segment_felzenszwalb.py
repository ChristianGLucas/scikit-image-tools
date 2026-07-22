from gen.messages_pb2 import FelzenszwalbInput, FelzenszwalbResult, SegmentStat
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, segment_stats

DEFAULT_MAX_SEGMENTS_RETURNED = 2000
HARD_MAX_SEGMENTS_RETURNED = 5000


def segment_felzenszwalb(ax: AxiomContext, input: FelzenszwalbInput) -> FelzenszwalbResult:
    """Segments an image via the deterministic graph-based Felzenszwalb-
    Huttenlocher algorithm (skimage.segmentation.felzenszwalb) — no
    randomness involved, so output depends only on the image and params.
    Higher `scale` favors larger/fewer segments; `min_size` merges away
    segments smaller than it. Returns the segment count and per-segment
    area/centroid/mean-color statistics, capped at `max_segments_returned`.
    """
    try:
        arr = load_array(input.image)

        scale = input.scale if input.scale != 0 else 100.0
        if scale <= 0:
            raise SkimgError("scale must be > 0")
        sigma = input.sigma if input.sigma != 0 else 0.8
        if sigma < 0:
            raise SkimgError("sigma must be >= 0")
        min_size = input.min_size if input.min_size != 0 else 20
        if min_size < 0:
            raise SkimgError("min_size must be >= 0")
        max_segments_returned = input.max_segments_returned or DEFAULT_MAX_SEGMENTS_RETURNED
        max_segments_returned = min(max(max_segments_returned, 1), HARD_MAX_SEGMENTS_RETURNED)

        from skimage.segmentation import felzenszwalb

        channel_axis = -1 if arr.ndim == 3 else None
        img = arr[..., :3] if (arr.ndim == 3 and arr.shape[-1] == 4) else arr
        labels = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size, channel_axis=channel_axis)
        # felzenszwalb labels start at 0; regionprops ignores label 0 as
        # background, so shift to start at 1 to include every segment.
        labels = labels + 1

        rows, total, truncated = segment_stats(labels, arr, max_segments_returned)
        segments = [SegmentStat(**r) for r in rows]
        return FelzenszwalbResult(segment_count=total, segments=segments, truncated=truncated)
    except SkimgError as exc:
        return FelzenszwalbResult(error=str(exc))
