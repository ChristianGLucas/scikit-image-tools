from gen.messages_pb2 import MorphologyInput, MorphologyResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, encode_mask

_OPS = {"erosion", "dilation", "opening", "closing", "skeletonize", "remove_small_objects"}


def morphology(ax: AxiomContext, input: MorphologyInput) -> MorphologyResult:
    """Applies a binary morphological operation to a mask: "erosion",
    "dilation", "opening", "closing" (disk structuring element of radius
    `footprint_size`), "skeletonize" (thin to a 1-pixel-wide skeleton), or
    "remove_small_objects" (drop connected components smaller than
    `min_size` pixels). Wraps skimage.morphology.
    """
    try:
        mask_arr = load_array(input.mask, allow_color=False)
        mask = mask_arr > 0
        op = (input.operation or "").strip().lower()
        if op not in _OPS:
            raise SkimgError(f"unknown operation {op!r}; expected one of {sorted(_OPS)}")

        from skimage.morphology import erosion, dilation, opening, closing, disk, skeletonize, remove_small_objects

        if op == "skeletonize":
            result = skeletonize(mask)
        elif op == "remove_small_objects":
            min_size = input.min_size if input.min_size != 0 else 64
            if min_size < 0:
                raise SkimgError("min_size must be >= 0")
            result = remove_small_objects(mask, max_size=max(min_size - 1, 0), connectivity=2)
        else:
            footprint_size = input.footprint_size if input.footprint_size != 0 else 1
            if footprint_size < 1:
                raise SkimgError("footprint_size must be >= 1")
            selem = disk(footprint_size)
            fn = {"erosion": erosion, "dilation": dilation, "opening": opening, "closing": closing}[op]
            result = fn(mask, footprint=selem)

        return MorphologyResult(mask=Image(**encode_mask(result)))
    except SkimgError as exc:
        return MorphologyResult(error=str(exc))
