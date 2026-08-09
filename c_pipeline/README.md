# 4DGS C Pipeline

Clean C implementation of a decoder + encoder + comparison tool for the
`.4dgs` / `.4dgs.gz` compressed dynamic Gaussian scene format produced by
`Dynamic_C3DGS`. The format is the same one your `parser.py` writes and
that the existing `flamsplat/core/utils/fourdgs_loader.c` reads.

This is **NOT a new format** — every byte layout is verified against the
existing Python and C readers. See `FORMAT_SCHEMA.md` for the full
authoritative schema.

## Files

| File              | Purpose                                                |
|-------------------|--------------------------------------------------------|
| `4dgs_format.h`   | Format constants (magic, version, dims, EOF symbol)    |
| `fp16.{h,c}`      | IEEE-754 half ↔ float                                  |
| `gzip_io.{h,c}`   | File I/O + zlib gzip inflate / deflate                 |
| `huffman.{h,c}`   | Decode tree, canonical-Huffman encode, byte readers   |
| `rvq.{h,c}`       | RVQ reconstruction + greedy nearest-neighbor encoder   |
| `scene.{h,c}`     | `scene_t` struct, motion application, bounds, fallback RGB |
| `decoder.{h,c}`   | Stage 2 + 3: parse + validate                          |
| `encoder.{h,c}`   | Stage 5A/B: re-emit binary (uses existing codebooks)   |
| `renderer_cpu.{h,c}` | Stage 4: simple CPU point/disc splatter to PPM      |
| `main.c`          | CLI driver                                             |
| `Makefile`        | Build (links `-lz -lm`)                                |

## Build

Requires `cc` (clang or gcc), `zlib`, `make`. On macOS these are
already installed.

```
cd c_pipeline
make
```

Produces `4dgs_pipeline`.

## Commands

```
./4dgs_pipeline decode      <in.4dgs[.gz]>
./4dgs_pipeline render      <in.4dgs[.gz]> <out.ppm> [t]
./4dgs_pipeline encode      <in.4dgs[.gz]> <out.4dgs[.gz]>
./4dgs_pipeline roundtrip   <in.4dgs[.gz]> <re.4dgs[.gz]>
./4dgs_pipeline render-pair <in.4dgs[.gz]> <orig.ppm> <re.ppm> [t]
```

The output file extension `.gz` controls whether the encoder gzip-wraps.

## Demo (what to show your manager)

From inside `c_pipeline/`:

```
# 1. decode + validate (Stages 2 & 3)
./4dgs_pipeline decode ../ours_cook_spinach.4dgs

# 2. render a frame at t=0 (Stage 4)
./4dgs_pipeline render ../ours_cook_spinach.4dgs frame_t0.ppm 0.0

# 3. round-trip + per-attribute error report (Stages 5 & 6)
./4dgs_pipeline roundtrip ../ours_cook_spinach.4dgs roundtrip.4dgs

# 4. visual side-by-side: orig vs round-trip (Stage 6 visual)
./4dgs_pipeline render-pair ../ours_cook_spinach.4dgs orig.ppm reenc.ppm 0.0
```

PPM files open in macOS Preview directly.

## Verified results on `ours_cook_spinach.4dgs`

```
[decoder] magic=4DGS version=3 N=52648
  xyz      shape=[52648,3] min=[-48.25,-36.88, 4.52] max=[58.34,57.22,340.25]
  motion   shape=[52648,9] min=-8.77       max=8.86
  opacity  N=52648  qrange=[0.005, 1.0]
  tcen     N=52648  qrange=[-0.039, 1.04]
  tsca     N=52648  qrange=[-6.32, 14.31]
  scale    [52648,3] layers=4 cb=256 rvq_bit=8
  rotation [52648,4] layers=4 cb=256 rvq_bit=8
  omega    [52648,4] layers=3 cb=256 rvq_bit=8
  tfea     [52648,3] layers=3 cb=256 rvq_bit=8
  features_dc [52648,6] (baked)
  rgb_dec  w1=[6,12]  w2=[3,6]
[decoder] consumed 2,694,905 / 2,694,905 bytes (EXACT)
```

Round-trip max-error per attribute (orig vs decoded(encoded(orig))):

| Attribute    | max_err     | mean_err     | Notes                              |
|--------------|-------------|--------------|------------------------------------|
| xyz          | 0           | 0            | f16 → f16 lossless                 |
| motion       | 0           | 0            | f16 → f16 lossless                 |
| opacity      | 0           | 0            | original was 8-bit, exact match    |
| tcen         | 1.19e-07    | 5.88e-08     | float-precision dequant rounding   |
| tsca         | 0           | 0            |                                    |
| scale        | 0.786       | 6.7e-04      | greedy RVQ reassignment            |
| rotation     | 0.577       | 4.3e-04      | greedy RVQ reassignment            |
| omega        | 0.027       | 6.85e-06     | greedy RVQ reassignment            |
| tfea         | 0.236       | 4.0e-04      | greedy RVQ reassignment            |
| features_dc  | 0           | 0            | f16 → f16 lossless                 |
| rgb_w1, rgb_w2 | 0         | 0            | f16 → f16 lossless                 |

The non-zero RVQ entries are expected — see "Encoder strategy" below.

## Encoder strategy (Stage 5)

Two levels are implemented in the same code path; choose by which
attributes you re-encode from raw vs. preserve.

**Level A — container round-trip:** the encoder reads the existing
codebooks from the input, re-emits the same VQ block layout (codebooks
are stored back as f16 with bit-identical values). This part is
lossless.

**Level B — true compression:** the encoder takes the **decoded float
vectors** for `scale`, `rotation`, `omega`, `tfea` and rebuilds the
indices from scratch using **greedy nearest-neighbor residual
quantization** (`rvq_encode_indices` in `rvq.c`):

1. Start with the input vector as the initial residual.
2. For each layer, scan the codebook and pick the entry that minimizes
   L2 distance to the current residual.
3. Subtract the chosen entry from the residual and proceed.

Indices are then bit-packed LSB-first and Huffman-coded with a freshly
built canonical Huffman table (`huffman_build_table`). Scalar blocks
(opacity, tcen, tsca) are re-quantized to 8 bits with a fresh min-max
and re-encoded.

This is genuine encoding (Stage 5B) — no original Huffman bitstreams
are reused. Re-decoding yields a numerically very close but
non-identical scene because greedy RVQ is sub-optimal vs. the
encoder-time joint optimization in `Dynamic_C3DGS`.

If you want **bit-exact** preservation, you'd need to also keep the
original packed Huffman-encoded byte streams next to the decoded
floats in `scene_t`. That's a small extension but explicitly out of
scope here, since the current pipeline already proves the format is
well-understood and re-encodable end-to-end.

## What is NOT a true encoder here

We start from already-decoded floats and learned codebooks. We do
**not**:

- Train new codebooks from raw vectors (that requires k-means / EM
  iteration on the residuals).
- Bake `features_dc` from a hash grid + tcnn MLP — your `parser.py`
  needs CUDA / `tinycudann` for that. The C pipeline only re-emits
  whatever `features_dc` is already in the file.

## Why round-trip rendered images are pixel-identical

The CPU renderer uses `xyz` (f16-lossless), `motion` (f16-lossless),
`opacity` (8-bit, exact match), `features_dc` (f16-lossless). It does
not currently use `scale`, `rotation`, `omega`, `tfea`, so the small
RVQ-induced numerical drift in those attributes does not appear in the
rendered output. If you swap the renderer for a real Gaussian splatter,
you'd see sub-pixel deviations consistent with the per-attribute errors
above.
