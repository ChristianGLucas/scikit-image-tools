# scikit-image-tools

Composable Axiom nodes for deterministic **scientific image analysis** —
thresholding, region measurement (regionprops), segmentation, morphology,
feature detection, texture analysis, and denoising — wrapping
[scikit-image](https://scikit-image.org/) (BSD-3-Clause) and numpy.

Built for the [Axiom](https://axiomide.com) marketplace, handle
`christiangeorgelucas`.

This package is distinct from `image-tools` (basic Pillow operations — resize,
crop, rotate, format conversion) and `opencv-tools` (OpenCV computer-vision
ops). `scikit-image-tools` covers the higher-level *scientific*-imaging
surface: counting and measuring discrete objects, segmenting an image into
regions, and extracting quantitative texture/shape features — the operations
a microscopy, document-analysis, or computer-vision agent needs.

All nodes are pure, deterministic, and offline (no network calls — the `url`
field on the shared `Image` envelope exists only for flow-edge shape
compatibility with `image-tools`/`opencv-tools` and is never fetched).
Payload size, pixel-count, and memory/CPU/time limits are enforced by the
Axiom platform itself, not by this package.

## Nodes

- **Threshold** — Otsu / adaptive / Li / Yen binarization → mask + threshold value.
- **RegionProps** — label connected components + full regionprops measurement
  (area, centroid, bbox, eccentricity, orientation, perimeter, solidity, mean
  intensity, axis lengths, extent, Euler number). The flagship node for
  counting/measuring objects (cells, particles, blobs).
- **CountObjects** — lightweight threshold+label object count with aggregate
  area stats.
- **SegmentSlic** — SLIC superpixel segmentation (caller-seeded for
  determinism) → segment count + per-segment stats.
- **SegmentFelzenszwalb** — graph-based Felzenszwalb-Huttenlocher segmentation.
- **SegmentWatershed** — marker-based watershed segmentation for splitting
  touching objects in a binary mask.
- **DetectEdges** — Canny / Sobel / Scharr edge detection → edge map + density.
- **DetectCorners** — Harris / Shi-Tomasi corner detection → coordinates.
- **DetectBlobs** — Laplacian-of-Gaussian / Difference-of-Gaussian blob
  detection → coordinates + radii.
- **Morphology** — erosion / dilation / opening / closing / skeletonize /
  remove_small_objects on a binary mask.
- **ComputeHOG** — Histogram of Oriented Gradients feature descriptor.
- **ConvertColorSpace** — RGB ↔ grayscale / HSV / Lab.
- **TextureFeatures** — GLCM (Haralick) texture descriptors: contrast,
  homogeneity, energy, correlation, dissimilarity, ASM.
- **Denoise** — total-variation / bilateral / wavelet denoising.
- **FindContours** — marching-squares iso-contour extraction.
- **ImageStatistics** — per-channel mean/std/min/max/histogram + entropy.
- **AdjustImage** — histogram equalization / CLAHE / gamma correction.

## License

MIT — see [LICENSE](./LICENSE). scikit-image and numpy are both BSD-3-Clause;
the full runtime dependency tree is permissively licensed (BSD/MIT/Apache-2.0).
