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

## Use it from your agent or app

Every node in this package is a **live, auto-scaling API endpoint** on the
[Axiom](https://axiomide.com) marketplace — call it from an AI agent or your own
code, with nothing to self-host.

**📦 See it on the marketplace:**
https://dev.axiomide.com/marketplace/christiangeorgelucas/scikit-image-tools@0.1.1

**Hook it up to an AI agent (MCP).** Add Axiom's hosted MCP server to any MCP
client and every node becomes a typed tool your agent can call — search the
catalog, inspect a schema, and invoke it directly.

```bash
# Claude Code
claude mcp add --transport http axiom https://api.axiomide.com/mcp \
  --header "Authorization: Bearer $AXIOM_API_KEY"
```

Claude Desktop, Cursor, or any config-based client:

```json
{
  "mcpServers": {
    "axiom": {
      "type": "http",
      "url": "https://api.axiomide.com/mcp",
      "headers": { "Authorization": "Bearer YOUR_AXIOM_API_KEY" }
    }
  }
}
```

**Call it from the CLI.**

```bash
axiom invoke christiangeorgelucas/scikit-image-tools/Threshold --input '{ ... }'
```

**Call it over HTTP.**

```bash
curl -X POST https://api.axiomide.com/invocations/v1/nodes/christiangeorgelucas/scikit-image-tools/0.1.1/Threshold \
  -H "Authorization: Bearer $AXIOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{ ... }'
```

> Input/output schema for each node is on the marketplace page above, or via
> `axiom inspect node christiangeorgelucas/scikit-image-tools/Threshold`.

### Get started free

Install the CLI:

```bash
# macOS / Linux — Homebrew
brew install axiomide/tap/axiom

# macOS / Linux — install script
curl -fsSL https://raw.githubusercontent.com/AxiomIDE/axiom-releases/main/install.sh | sh
```

**Windows:** download the `windows/amd64` `.zip` from the
[releases page](https://github.com/AxiomIDE/axiom-releases/releases), unzip it,
and put `axiom.exe` on your `PATH`.

Then `axiom version` to verify, `axiom login` (GitHub or Google) to authenticate,
and create an API key under **Console → API Keys**. Docs and sign-up at
**[axiomide.com](https://axiomide.com)**.

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
