# File Inventory — what each file does in this repo

This is the hand-off list. Every file is grouped by **what role it plays in
the encode → decode → render → compare pipeline**, so you can point at
exactly the file when explaining each step.

---

## 1. INPUT DATA (provided by manager)

| File | Format | Role |
|------|--------|------|
| `point_cloud_pp.npz`      | numpy archive | Original Dynamic_C3DGS preprocessed checkpoint (alternate input path). |
| `ours_cook_spinach.4dgs`  | C-friendly binary | The compressed dynamic Gaussian scene we operate on end-to-end. |

---

## 2. ENCODER  (Stage 5 of the spec)

The encoder turns decoded scene attributes back into a `.4dgs` / `.4dgs.gz`
binary using Huffman + RVQ + f16, exactly like the format reader expects.

| File | What it does |
|------|--------------|
| `c_pipeline/encoder.h` / `encoder.c`     | Top-level: takes a `scene_t` and writes the binary; gzip-wraps if filename ends with `.gz`. |
| `c_pipeline/huffman.h` / `huffman.c`     | Builds canonical Huffman tables from byte-frequency, encodes byte streams + the EOF marker. |
| `c_pipeline/rvq.h` / `rvq.c`             | Greedy nearest-neighbour residual VQ index assignment + LSB-first bit packing. |
| `c_pipeline/fp16.h` / `fp16.c`           | float ↔ IEEE-754 half conversion (used for xyz, motion, codebooks, features_dc, rgb_dec). |
| `c_pipeline/gzip_io.h` / `gzip_io.c`     | zlib gzip deflate (and the read side, for the decoder). |
| `c_pipeline/scene.h` / `scene.c`         | In-memory scene struct shared between encoder and decoder. |
| `c_pipeline/4dgs_format.h`               | Format constants (magic `4DGS`, version, dims, EOF symbol). |

**Python-side encoder (a simpler npz-based codec):**
- `encoder.py` — saves `SceneProperties` into a min-max-quantized `.npz`.
- `quantize.py` — `quantize_scalar`, `quantize_vector`, `decode_rvq` helpers.
- `bitstream.py` — bit packing utilities.

---

## 3. DECODER  (Stage 2 of the spec)

The decoder reads a `.4dgs` (or `.4dgs.gz`) file and produces an in-memory
scene with all attributes dequantized.

| File | What it does |
|------|--------------|
| `c_pipeline/decoder.h` / `decoder.c`     | Parses gzip wrapper + 4DGS header + raw f16 blocks + scalar Huffman blocks + RVQ Huffman blocks + optional features_dc / rgb_dec; prints validation. |
| `c_pipeline/huffman.c`                   | Used here for the Huffman-block decode side. |
| `c_pipeline/rvq.c`                       | Reconstructs vectors from packed indices + codebooks. |
| `c_pipeline/fp16.c`                      | f16 → f32 conversion for all f16 blocks. |
| `c_pipeline/gzip_io.c`                   | gzip auto-detect + inflate. |

**Python-side decoder (used by the renderer):**
- `decoder.py` — loads the `.npz` checkpoint into `SceneProperties`.
- `scene_io.py` — `load_scene_properties_from_4dgs()` reads the binary the same way the C decoder does, plus reconstructs RGB through the baked rgb_dec MLP.

---

## 4. COMPRESSION LOGIC  (the actual codec, paper Section)

| Step in paper | C source line | Python source line |
|---------------|---------------|--------------------|
| 8-bit min-max scalar quantization (opacity/tcen/tsca) | `encoder.c::write_scalar_block` | `quantize.py::quantize_scalar` |
| Canonical Huffman over byte frequencies              | `huffman.c::huffman_build_table`, `huffman_encode_block` | `dahuffman` library + `scene_io.py::_decode_huffman_entries` |
| Residual VQ (greedy NN) with shared codebooks        | `rvq.c::rvq_encode_indices`, `rvq_pack_indices` | (decode-only in Python) |
| f16 storage of xyz / motion / codebooks / features   | `fp16.c::float_to_fp16` | numpy `.astype(np.float16)` |
| gzip wrapper                                          | `gzip_io.c::gzip_deflate` | gzip module in `scene_io.py` |

So when your manager asks *"where is the compression?"*, point at:
- `c_pipeline/encoder.c` (the orchestrator) and
- `c_pipeline/huffman.c` + `c_pipeline/rvq.c` (the two compression cores).

---

## 5. RENDER

| File | What it does |
|------|--------------|
| `renderer.py`           | CPU EWA Gaussian-splat renderer. Builds 3D covariance from rotation+scale, projects to 2D via Jacobian, alpha-composites disks. (Same algorithm as `flamsplat/core/rendering/splat.glsl`, just on CPU in numpy.) |
| `scene.py`              | Time-walks the scene: `xyz(t) = xyz0 + v*t + a*t² + j*t³` from the `motion[N,9]` polynomial; passes static color/scale/rotation to the renderer. |
| `camera.py`             | Pinhole camera with `set_pose(position, target, up)` look-at matrix. |
| `render_pipeline.py`    | Wires `Scene + Camera + renderer.render_scene_cpu` and exports per-frame PNGs + MP4. |
| `export_video.py`       | PNG-frames → MP4 via `imageio-ffmpeg`. |
| `main.py`               | CLI: `python main.py <input.4dgs|input.npz>` produces a single video. |
| `compare_render.py`     | **The demo script.** Loads two scenes (reference + C-encoder output), renders them with the same orbit camera + time, overlays sizes / PSNR / file names, writes a side-by-side MP4. |

---

## 6. COMPARISON / VALIDATION  (Stage 6 of the spec)

| File | What it does |
|------|--------------|
| `c_pipeline/main.c::cmd_roundtrip`  | The numerical proof. Decodes the input, re-encodes, decodes again, prints per-attribute `max_err` and `mean_err`. |
| `compare_render.py`                 | The visual proof. Side-by-side video with PSNR per frame and on-screen file sizes. |

---

## 7. DOCUMENTATION

| File | What it covers |
|------|----------------|
| `c_pipeline/FORMAT_SCHEMA.md` | Authoritative byte-layout for the `.4dgs` format. Cross-checked against `flamsplat/core/utils/fourdgs_loader.c` and `scene_io.py`. |
| `c_pipeline/README.md`        | Build instructions, command summary, the actual numbers measured on `ours_cook_spinach.4dgs`. |
| `MANAGER_BRIEF.md`            | One-page narrative for stakeholders: what input we got, what was built, the compression numbers, the demo script. |
| `FILE_INVENTORY.md` (this)    | The file-by-file map you're reading. |

---

## 8. EXISTING REFERENCE CODE  (for context — we did not modify these)

| File | What it is |
|------|------------|
| `flamsplat/core/utils/fourdgs_loader.c` | Manager's reference C decoder. Our `c_pipeline/decoder.c` is the cleaner re-implementation. |
| `flamsplat/core/rendering/splat.glsl`   | GPU shader for proper Gaussian splatting. Our `renderer.py` is the CPU equivalent. |
| `flamsplat/core/splat.c`                | GPU-side sort + draw pipeline. |
| `flamsplat/tools/decode_4dgs.c`         | Manager's partial dumper (header + xyz + motion only). |

---

## How to build + run (one-liner reference)

```bash
# build the C encoder/decoder/renderer/compare tool
cd c_pipeline && make && cd ..

# Stage 2/3: decode + validate
./c_pipeline/4dgs_pipeline decode ours_cook_spinach.4dgs

# Stage 5/6: numerical round-trip proof
./c_pipeline/4dgs_pipeline roundtrip ours_cook_spinach.4dgs c_pipeline/roundtrip.4dgs

# Produce a gzip-wrapped output for size demo
./c_pipeline/4dgs_pipeline encode ours_cook_spinach.4dgs c_pipeline/roundtrip.4dgs.gz

# Render side-by-side comparison video with sizes overlaid
/opt/homebrew/bin/python3.12 compare_render.py \
    --reference  ours_cook_spinach.4dgs \
    --compressed c_pipeline/roundtrip.4dgs.gz \
    --output     compare_demo.mp4

open compare_demo.mp4
```

## Outputs to ship to your manager

| File | What it shows |
|------|---------------|
| `c_pipeline/roundtrip.4dgs(.gz)`  | Output of your encoder — proof the codec is implemented end-to-end. |
| `compare_demo.mp4`                 | Side-by-side render with size & PSNR overlay. |
| `MANAGER_BRIEF.md`                 | Talking points + numbers. |
| `c_pipeline/FORMAT_SCHEMA.md`      | The format spec. |
| `c_pipeline/README.md`             | How to rebuild the pipeline. |
| `FILE_INVENTORY.md`                | This file — what each source file does. |
