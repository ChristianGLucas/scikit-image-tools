from gen.messages_pb2 import SlicInput, SlicResult, SegmentStat
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, segment_stats


def segment_slic(ax: AxiomContext, input: SlicInput) -> SlicResult:
    """Segments an image into superpixels via SLIC (Simple Linear Iterative
    Clustering, skimage.segmentation.slic). `seed` seeds numpy's global RNG
    before segmenting, so output is reproducible for a fixed input+params
    (scikit-image's current SLIC implementation is itself already
    grid-initialized and deterministic; `seed` is accepted so callers can
    rely on determinism as a stated contract rather than an implementation
    detail). Returns the segment count and per-segment area/centroid/
    mean-color statistics for every segment found.
    """
    try:
        arr = load_array(input.image)

        n_segments = input.n_segments or 100
        if n_segments < 2:
            raise SkimgError("n_segments must be >= 2")
        compactness = input.compactness if input.compactness != 0 else 10.0
        if compactness <= 0:
            raise SkimgError("compactness must be > 0")
        sigma = max(0.0, input.sigma)

        import numpy as np
        from skimage.segmentation import slic

        np.random.seed(input.seed or 0)

        channel_axis = -1 if arr.ndim == 3 else None
        img_for_slic = arr[..., :3] if (arr.ndim == 3 and arr.shape[-1] == 4) else arr
        labels = slic(
            img_for_slic,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            channel_axis=channel_axis,
            start_label=1,
        )

        rows, total = segment_stats(labels, arr)
        segments = [SegmentStat(**r) for r in rows]
        return SlicResult(segment_count=total, segments=segments)
    except SkimgError as exc:
        return SlicResult(error=str(exc))
