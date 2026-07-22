from gen.messages_pb2 import ColorConvertInput, ColorConvertResult, Image
from gen.axiom_context import AxiomContext
from nodes._imaging import SkimgError, load_array, to_gray, encode_gray_u8, encode_rgb_u8, rescale_to_u8

_TARGETS = {"gray", "hsv", "lab"}


def convert_color_space(ax: AxiomContext, input: ColorConvertInput) -> ColorConvertResult:
    """Converts an image to grayscale, HSV, or CIE Lab (skimage.color).
    "gray" returns a standard single-channel grayscale PNG. "hsv"/"lab" have
    native channel ranges that don't fit 0-255 directly (e.g. Lab's L is
    [0,100], a/b are roughly [-128,127]); each channel is therefore
    independently min-max rescaled to 0-255 for PNG storage, in H,S,V or
    L,a,b channel order — this is a lossy-but-deterministic encoding, not a
    standard color-managed transform.
    """
    try:
        arr = load_array(input.image)
        target = (input.target or "gray").strip().lower()
        if target not in _TARGETS:
            raise SkimgError(f"unknown target {target!r}; expected one of {sorted(_TARGETS)}")

        if target == "gray":
            gray = to_gray(arr)
            out = Image(**encode_gray_u8(gray))
        else:
            rgb = arr[..., :3] if (arr.ndim == 3 and arr.shape[-1] == 4) else arr
            if rgb.ndim == 2:
                import numpy as np

                rgb = np.stack([rgb, rgb, rgb], axis=-1)

            from skimage.color import rgb2hsv, rgb2lab

            converted = rgb2hsv(rgb) if target == "hsv" else rgb2lab(rgb)
            u8 = converted.copy()
            for ch in range(3):
                u8[..., ch] = rescale_to_u8(converted[..., ch])
            out = Image(**encode_rgb_u8(u8.astype("uint8")))

        return ColorConvertResult(image=out, target=target)
    except SkimgError as exc:
        return ColorConvertResult(error=str(exc))
