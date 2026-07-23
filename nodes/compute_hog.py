from gen.messages_pb2 import HogInput, HogResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray, encode_gray_u8, rescale_to_u8

def compute_hog(ax: AxiomContext, input: HogInput) -> HogResult:
    """Computes the Histogram of Oriented Gradients (HOG) feature descriptor
    (skimage.feature.hog) — the classic gradient-orientation feature vector
    used for object/shape detection and classification. `orientations` sets
    the number of gradient-direction bins; `pixels_per_cell`/
    `cells_per_block` control the spatial pooling. Returns the flattened
    feature vector and, when `visualize` is true, a normalized gradient-
    orientation visualization image (same dimensions as the input).
    """
    try:
        arr = load_array(input.image)
        gray = to_gray(arr)

        orientations = input.orientations if input.orientations != 0 else 9
        ppc = input.pixels_per_cell if input.pixels_per_cell != 0 else 8
        cpb = input.cells_per_block if input.cells_per_block != 0 else 2
        if orientations < 1:
            raise SkimgError("orientations must be >= 1")
        if ppc < 2:
            raise SkimgError("pixels_per_cell must be >= 2")
        if cpb < 1:
            raise SkimgError("cells_per_block must be >= 1")

        h, w = gray.shape
        n_cells_row, n_cells_col = h // ppc, w // ppc
        n_blocks_row = max(n_cells_row - cpb + 1, 0)
        n_blocks_col = max(n_cells_col - cpb + 1, 0)
        if n_blocks_row <= 0 or n_blocks_col <= 0:
            raise SkimgError(
                "image too small for the given pixels_per_cell/cells_per_block "
                "(no complete blocks fit)"
            )

        from skimage.feature import hog

        if input.visualize:
            features, viz = hog(
                gray,
                orientations=orientations,
                pixels_per_cell=(ppc, ppc),
                cells_per_block=(cpb, cpb),
                visualize=True,
                feature_vector=True,
            )
            viz_image = Image(**encode_gray_u8(rescale_to_u8(viz)))
        else:
            features = hog(
                gray,
                orientations=orientations,
                pixels_per_cell=(ppc, ppc),
                cells_per_block=(cpb, cpb),
                visualize=False,
                feature_vector=True,
            )
            viz_image = Image()

        feat_list = [float(x) for x in features]
        return HogResult(features=feat_list, feature_count=len(feat_list), visualization=viz_image)
    except SkimgError as exc:
        return HogResult(error=str(exc))
