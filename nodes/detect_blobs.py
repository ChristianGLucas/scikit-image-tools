from gen.messages_pb2 import BlobsInput, BlobsResult, Blob
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

_METHODS = {"log", "dog"}
_SQRT2 = 1.4142135623730951


def detect_blobs(ax: AxiomContext, input: BlobsInput) -> BlobsResult:
    """Detects blob-like structures via scale-space Laplacian-of-Gaussian
    (skimage.feature.blob_log, default — more accurate) or Difference-of-
    Gaussian (blob_dog — faster) detectors. `min_sigma`/`max_sigma` bound the
    blob size range searched; lower `threshold` finds more/fainter blobs.
    Returns every blob's (row, col) center and approximate radius
    (sigma * sqrt(2)) found.
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)
        method = (input.method or "log").strip().lower()
        if method not in _METHODS:
            raise SkimgError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")

        min_sigma = input.min_sigma if input.min_sigma != 0 else 1.0
        max_sigma = input.max_sigma if input.max_sigma != 0 else 30.0
        if min_sigma <= 0:
            raise SkimgError("min_sigma must be > 0")
        if max_sigma <= 0:
            raise SkimgError("max_sigma must be > 0")
        if max_sigma < min_sigma:
            raise SkimgError("max_sigma must be >= min_sigma")
        threshold = input.threshold if input.threshold != 0 else 0.1

        from skimage.feature import blob_log, blob_dog

        if method == "log":
            found = blob_log(gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        else:
            found = blob_dog(gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)

        total = len(found)
        blobs = [Blob(row=float(r), col=float(c), radius=float(s) * _SQRT2) for r, c, s in found]

        return BlobsResult(blobs=blobs, count=total)
    except SkimgError as exc:
        return BlobsResult(error=str(exc))
