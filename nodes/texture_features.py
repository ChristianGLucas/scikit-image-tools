import math

from gen.messages_pb2 import TextureInput, TextureResult
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray

MAX_LEVELS = 256  # hard dtype limit: quantized gray levels are packed into np.uint8


def texture_features(ax: AxiomContext, input: TextureInput) -> TextureResult:
    """Computes Gray-Level Co-occurrence Matrix (GLCM) texture descriptors —
    contrast, dissimilarity, homogeneity, energy, correlation, ASM
    (skimage.feature.graycomatrix / graycoprops) — the classic Haralick
    texture features used in scientific/microscopy image analysis. The image
    is grayscale-converted and quantized to `levels` gray levels (default 32)
    before computing the co-occurrence matrix; results are averaged over the
    given `distances` (default [1]) and `angles` in radians (default
    [0, pi/4, pi/2, 3pi/4]).
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)  # [0, 1]

        levels = input.levels if input.levels > 0 else 32
        if levels < 2 or levels > MAX_LEVELS:
            raise SkimgError(f"levels must be in [2, {MAX_LEVELS}]")

        distances = list(input.distances) if len(input.distances) else [1]
        angles = list(input.angles) if len(input.angles) else [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        if not distances or not angles:
            raise SkimgError("distances and angles must be non-empty")
        if any(d < 1 for d in distances):
            raise SkimgError("each distance must be >= 1")

        import numpy as np
        from skimage.feature import graycomatrix, graycoprops

        quantized = (gray * (levels - 1)).round().astype(np.uint8)
        glcm = graycomatrix(
            quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True
        )

        def avg(prop: str) -> float:
            # A uniform region (e.g. flat background) makes "correlation"
            # divide by a zero standard deviation -> NaN, which is not valid
            # JSON; a uniform patch is a realistic input, not a pathological
            # one, so guard it rather than let it leak into the response.
            val = float(np.nan_to_num(graycoprops(glcm, prop).mean(), nan=0.0, posinf=0.0, neginf=0.0))
            return val

        return TextureResult(
            contrast=avg("contrast"),
            dissimilarity=avg("dissimilarity"),
            homogeneity=avg("homogeneity"),
            energy=avg("energy"),
            correlation=avg("correlation"),
            asm=avg("ASM"),
        )
    except SkimgError as exc:
        return TextureResult(error=str(exc))
