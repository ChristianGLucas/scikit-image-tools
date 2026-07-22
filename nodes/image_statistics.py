from gen.messages_pb2 import StatsInput, StatsResult, ChannelStats
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray, shannon_entropy


def image_statistics(ax: AxiomContext, input: StatsInput) -> StatsResult:
    """Computes per-channel image statistics — mean, standard deviation,
    min, max, and a histogram — for grayscale images ("gray") or each of
    R/G/B(/A) for color images, plus the Shannon entropy (bits) of the
    grayscale-converted image.
    """
    try:
        arr = load_array(input.image)
        bins = input.histogram_bins if input.histogram_bins > 0 else 256
        if bins < 2 or bins > 256:
            raise SkimgError("histogram_bins must be in [2, 256]")

        import numpy as np

        channels = []
        if arr.ndim == 2:
            names = ["gray"]
            planes = [arr]
        else:
            names = ["r", "g", "b", "a"][: arr.shape[-1]]
            planes = [arr[..., i] for i in range(arr.shape[-1])]

        for name, plane in zip(names, planes):
            plane_f = plane.astype(np.float64)
            hist, _ = np.histogram(plane, bins=bins, range=(0, 255))
            channels.append(
                ChannelStats(
                    channel=name,
                    mean=float(plane_f.mean()),
                    std=float(plane_f.std()),
                    min=float(plane_f.min()),
                    max=float(plane_f.max()),
                    histogram=[int(x) for x in hist],
                )
            )

        gray = to_gray(arr)
        entropy = shannon_entropy(gray, bins=256)

        return StatsResult(channels=channels, entropy=entropy)
    except SkimgError as exc:
        return StatsResult(error=str(exc))
