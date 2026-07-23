from gen.messages_pb2 import RegionPropsInput, RegionPropsResult, Region
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array


def region_props(ax: AxiomContext, input: RegionPropsInput) -> RegionPropsResult:
    """Labels connected components of a binary mask (any nonzero pixel is
    foreground) and computes scikit-image `regionprops` measurements for
    each: area, centroid, bounding box, eccentricity, orientation, perimeter,
    solidity, mean intensity, major/minor axis length, extent, and Euler
    number. This is the flagship node for counting and measuring discrete
    objects (cells, particles, blobs) in a scientific image — feed it the
    `mask` output of Threshold. Supply `intensity_image` to compute
    `mean_intensity` against a different (e.g. original grayscale) image;
    otherwise the mask itself is used. `min_area` filters out tiny regions
    (e.g. noise specks). Wraps skimage.measure.label + regionprops.
    """
    try:
        mask_arr = load_array(input.mask, allow_color=False)
        mask = mask_arr > 0

        if len(input.intensity_image.data) > 0:
            intensity = load_array(input.intensity_image, allow_color=False)
            if intensity.shape != mask.shape:
                raise SkimgError(
                    f"intensity_image shape {intensity.shape} does not match "
                    f"mask shape {mask.shape}"
                )
        else:
            # Docstring contract: "otherwise the mask itself is used" — pass
            # it explicitly rather than None, which would make
            # `intensity_mean` raise (regionprops only computes it when an
            # intensity image was actually supplied).
            intensity = mask_arr

        connectivity = input.connectivity or 2
        if connectivity not in (1, 2):
            raise SkimgError("connectivity must be 1 or 2")

        min_area = input.min_area
        if min_area < 0:
            raise SkimgError("min_area must be >= 0")

        from skimage.measure import label, regionprops

        labeled = label(mask, connectivity=connectivity)
        props = regionprops(labeled, intensity_image=intensity)
        if min_area:
            props = [p for p in props if p.area >= min_area]

        total = len(props)

        regions = []
        for p in props:
            min_row, min_col, max_row, max_col = p.bbox
            regions.append(
                Region(
                    label=int(p.label),
                    area=float(p.area),
                    centroid_row=float(p.centroid[0]),
                    centroid_col=float(p.centroid[1]),
                    bbox_min_row=int(min_row),
                    bbox_min_col=int(min_col),
                    bbox_max_row=int(max_row),
                    bbox_max_col=int(max_col),
                    eccentricity=float(p.eccentricity),
                    orientation=float(p.orientation),
                    perimeter=float(p.perimeter),
                    solidity=float(p.solidity),
                    mean_intensity=float(p.intensity_mean),
                    major_axis_length=float(p.axis_major_length),
                    minor_axis_length=float(p.axis_minor_length),
                    extent=float(p.extent),
                    euler_number=int(p.euler_number),
                    equivalent_diameter=float(p.equivalent_diameter_area),
                )
            )

        return RegionPropsResult(regions=regions, count=total)
    except SkimgError as exc:
        return RegionPropsResult(error=str(exc))
