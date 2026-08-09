# 4D Gaussian Scene — Encoder / Decoder Pipeline (Manager Brief)

## What we were given

| File                          | Format             | Role                                                              |
|-------------------------------|--------------------|-------------------------------------------------------------------|
| `point_cloud_pp.npz`          | numpy archive      | Original `Dynamic_C3DGS` preprocessed checkpoint (Python-side)    |
| `ours_cook_spinach.4dgs`      | binary (gzip-able) | Manager's converted, C-friendly compressed scene                  |
| `flamsplat/` (C scene engine) | C source           | Reference C decoder for `.4dgs` (`fourdgs_loader.c`)              |
| Paper: arxiv 2408.03822       | PDF                | "Compact 3D Gaussian Splatting…" — describes the codec we're building |

The deliverable from the task spec was: **encode → decode → render → compare**, and prove the C pipeline reads / writes the binary correctly with a measurable round-trip.

## What was built

```
┌─────────────────────┐    parser.py (manager)    ┌───────────────────────────────────┐
│ point_cloud_pp.npz  │ ─────────────────────────► │ ours_cook_spinach.4dgs(.gz)       │
│  (numpy floats,     │                           │  - 4DGS header + version + N      │
│   raw codebooks,    │                           │  - f16 xyz, motion                │
│   Huffman tables)   │                           │  - Huffman scalar (opacity, …)    │
└─────────────────────┘                           │  - RVQ + Huffman (scale, rot, …)  │
                                                  │  - features_dc, rgb_dec MLP       │
                                                  └───────┬───────────────────────────┘
                                                          │
                                  ┌───────────────────────┴────────────────────────────┐
                                  │                                                    │
                                  ▼                                                    ▼
                  ┌─────────────────────────────────┐                   ┌─────────────────────────────┐
                  │ c_pipeline/4dgs_pipeline (C)    │                   │ Python renderer             │
                  │  - decoder.c (parse + validate) │                   │  - scene_io.py (load .4dgs) │
                  │  - encoder.c (Huffman + RVQ)    │                   │  - renderer.py (EWA splat)  │
                  │  - round-trip + pixel compare   │                   │  - compare_render.py        │
                  └─────────────────────────────────┘                   └─────────────────────────────┘
                                  │                                                    │
                                  ▼                                                    ▼
                       Stage 5/6 numerical proof                          Stage 4/6 visual proof
                       (per-attribute max_err, mean_err)                  (compare_demo.mp4 — labeled)
```

## What runs end-to-end

### 1. C encoder + decoder (`c_pipeline/4dgs_pipeline`)

Built with `make` from 9 small modules:

```
4dgs_format.h   gzip_io.{h,c}    huffman.{h,c}  rvq.{h,c}    fp16.{h,c}
scene.{h,c}     decoder.{h,c}    encoder.{h,c}  renderer_cpu.{h,c}    main.c
```

Implements all six stages of the spec:

| Stage | Capability |
|------:|------------|
| 1 | Format schema in `c_pipeline/FORMAT_SCHEMA.md` (cross-checked vs. `fourdgs_loader.c` and `scene_io.py`) |
| 2 | Decoder: gzip auto-detect + parse + Huffman + RVQ + optional features/rgb_dec |
| 3 | Validation: prints N, attribute shapes, per-attribute min/max/mean, byte accounting |
| 4 | CPU PPM debug renderer (sanity-check visual; not a Gaussian splatter) |
| 5A | Container round-trip writer |
| 5B | True compression encoder: greedy nearest-neighbour RVQ residual assignment + canonical Huffman + 8-bit min-max requantization |
| 6 | Per-attribute round-trip comparison (max_err, mean_err) |

### 2. Python renderer (`compare_render.py`)

Real EWA Gaussian-splat renderer with proper SH DC color reconstruction
(`color = clamp(0.5 + SH_C0 * features_dc[:, :3])` — same formula as the
flamsplat shader). Loads both the original `.4dgs` and the C-round-tripped
`.4dgs` and produces a single MP4 with each frame split:

- **Left half:** ORIGINAL — labeled with file name + size
- **Right half:** C ROUND-TRIPPED — labeled with file name, size, compression ratio, per-frame PSNR

This is what your manager actually watches.

## Numbers on `ours_cook_spinach.4dgs` (paper-style)

| Metric | Value |
|---|---|
| Gaussians N | 52,648 |
| Equivalent uncompressed PLY (paper baseline) | **≈ 15.46 MB** (full float32, SH degree 3, motion + temporal) |
| Manager input `.4dgs` (already compressed) | **≈ 2.57 MB** → **6.02× compression** vs raw |
| My C-encoder output `.4dgs.gz` | **≈ 2.37 MB** → **6.52× compression** vs raw |
| Bytes consumed by C decoder | 2,694,905 / 2,694,905 (**EXACT**) |
| Mean PSNR original vs round-trip (30-frame video) | **53.19 dB** (visually indistinguishable) |
| Per-attribute round-trip (`./4dgs_pipeline roundtrip`) | xyz / motion / opacity / tsca / features_dc / rgb_w1 / rgb_w2 → max_err = 0; scale / rot / omega / tfea → small RVQ-reassignment drift |

### Why the input and re-encoded sizes are close

Both files use the **same codec** (Huffman + RVQ + f16). The big compression
ratio comes from comparing to the **uncompressed PLY-equivalent** (15.46 MB),
not from comparing the input `.4dgs` to the re-encoded `.4dgs`. My re-encoded
file is slightly smaller because (a) canonical Huffman is a touch more
efficient and (b) we gzip-wrap the body for an extra deflate pass.

The *meaningful* compression number to quote is **6× over uncompressed
float32**. That's what is overlaid on every frame of `compare_demo.mp4`.

## Files produced for the demo

| File | Purpose |
|---|---|
| `c_pipeline/4dgs_pipeline` | C encoder/decoder/renderer binary |
| `c_pipeline/FORMAT_SCHEMA.md` | Authoritative byte-layout doc |
| `c_pipeline/README.md` | Build / run / what-each-file-does |
| `c_pipeline/roundtrip.4dgs` | Output of: decode → re-encode (proof of working encoder) |
| `compare_render.py` | Python script that produces side-by-side video |
| `compare_demo.mp4` | **Side-by-side video with file sizes & PSNR labels** |
| `compare_frames_demo/frame_*.png` | Same content as PNG frames for slides |

## Demo script (what to type in Terminal in front of your manager)

```bash
cd /Users/radhikarungta/Documents/gaussian_encoder_decoder

# A) Build the C pipeline
cd c_pipeline && make && cd ..

# B) Show decoder + validation (Stage 2/3): "every byte is understood"
./c_pipeline/4dgs_pipeline decode ours_cook_spinach.4dgs

# C) Show numerical round-trip table (Stage 5/6): "encoder is correct"
./c_pipeline/4dgs_pipeline roundtrip ours_cook_spinach.4dgs c_pipeline/roundtrip.4dgs

# D) Render the side-by-side comparison video with on-screen file sizes
/opt/homebrew/bin/python3.12 compare_render.py \
    --reference  ours_cook_spinach.4dgs \
    --compressed c_pipeline/roundtrip.4dgs \
    --output     compare_demo.mp4 \
    --num-frames 30 --framerate 15

# E) Open the video
open compare_demo.mp4
```

## How to talk through the demo (suggested narrative)

> *"My manager gave me two files: `point_cloud_pp.npz`, the Dynamic-C3DGS preprocessed checkpoint, and `ours_cook_spinach.4dgs`, the manager's compressed C-friendly binary. The paper at arxiv 2408.03822 describes the codec used: scalar attributes are Huffman-coded after 8-bit min-max quantization, and per-Gaussian vector attributes (scale, rotation, omega, tfea) are residual-vector-quantized with shared codebooks then Huffman-coded.*
>
> *I implemented the full encoder + decoder in C. The decoder reads the gzip wrapper, the 4DGS header, all three Huffman scalar blocks, all four RVQ Huffman blocks, the optional baked features_dc and the rgb_dec MLP weights — and confirms it consumed all 2,694,905 bytes of the input file with no leftovers.*
>
> *The encoder takes the decoded scene, re-quantizes the scalars, performs greedy nearest-neighbour RVQ index assignment using the file's own codebooks, and re-emits a fresh `.4dgs.gz`. Decoding that file again gives me xyz, motion, opacity, tsca, features_dc, and the MLP weights bit-for-bit identical to the original; the four RVQ-encoded attributes have small expected drift (mean error ≤ 4×10⁻⁴).*
>
> *Visually — here's the side-by-side video. Left half is the original 2.69 MB scene rendered with my Python EWA Gaussian-splat renderer; right half is exactly the same renderer but on the file my C encoder produced. Each frame shows the file size and the per-frame PSNR. Mean PSNR over 30 frames is 53 dB, which is well above the visually-lossless threshold — you can see the two halves are pixel-identical to the eye."*

## What was deliberately kept out of scope

- **Training new RVQ codebooks from scratch** (k-means iterations on residuals). The paper does this once during model fitting; we re-use the file's own codebooks and only do the assignment step.
- **Baking `features_dc` from a hash grid + tcnn MLP**. That requires CUDA and `tinycudann`; the C tool reuses whatever `features_dc` the file already contains.
- **GPU-accelerated rendering**. `renderer.py` is a CPU EWA splatter (good enough for verification; ~2s per 512×512 frame).

These are all clearly listed in `c_pipeline/README.md`.

## Where each item from the task spec lives

| Task spec item | File / command |
|---|---|
| "encode .4dgs.gz" | `c_pipeline/encoder.c` → `./4dgs_pipeline encode` |
| "decode .4dgs.gz" | `c_pipeline/decoder.c` → `./4dgs_pipeline decode` |
| "render" | `compare_render.py` + `renderer.py` (proper EWA splat) |
| "comparison logic" | `./4dgs_pipeline roundtrip` (numerical) and `compare_demo.mp4` (visual + size + PSNR overlay) |
| Paper-style size visible on frame | Each output frame has a translucent footer showing file size, ratio, PSNR |
