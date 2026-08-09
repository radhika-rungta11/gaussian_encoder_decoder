# gaussian_encoder_decoder

Encoders and decoders for 4D Gaussian-splatting scenes. The `main`/Huffman
pipeline lives at the repo root; the `feature/QUEEN` branch adds a
**QUEEN (QUantized Efficient ENcoding)** pipeline that targets stronger
compression than the existing Huffman baseline.

## QUEEN compression

QUEEN follows the quantization-sparsity-entropy structure proposed in
Girish et al., NeurIPS 2024 (see also the C3DGS paper:
<https://maincold2.github.io/c3dgs/>). The pipeline:

1. **Sparsity gating** — drop Gaussians whose effective alpha
   `sigmoid(opacity)` falls below `1/255` (they can't contribute to a
   rendered pixel anyway).
2. **Per-attribute quantization** — bit-budgets tuned per attribute:
   `xyz` 16-bit per axis, `motion` / `scale` (log) / `rotation` 10-bit,
   `opacity` / `tcen` / `tsca` / `omega` 8-bit, `tfea` is fed through a
   2-stage residual VQ with k-means codebooks (256 entries each).
3. **Entropy coding** — every quantized payload is run through both LZMA
   and zlib, whichever produces fewer bytes wins.

The output `.queen` file is fully self-describing: a JSON header records
every quantization parameter and codebook offset.

### How to run

```bash
# default: compress point_cloud_pp.npz -> point_cloud_pp.queen
python3.12 compress_queen.py

# explicit paths
python3.12 compress_queen.py point_cloud_pp.npz --out point_cloud_pp.queen
```

The runner prints a human-readable report to the console and also
writes `queen_compression_results.md` (Markdown table you can paste
into a doc or slide).

### Latest results on `point_cloud_pp.npz`

| Reference | Size | QUEEN size | Ratio |
|---|---|---|---|
| Source NPZ on disk (Huffman baseline) | 925.7 KB | 405.8 KB | **2.28× smaller** |
| Raw float32 dump | 1.90 MB | 405.8 KB | **4.79× smaller** |

Per-attribute PSNR ranges from **45 dB** on the high-dim color features
(RVQ, lossy by design) to **108 dB** on `xyz`; `opacity`, `tcen`, and
`tsca` round-trip losslessly at 8 bits.

### Files

- [queen_codec.py](queen_codec.py) — the encoder + decoder + entropy
  coding utilities.
- [compress_queen.py](compress_queen.py) — CLI runner and report
  generator.
- [queen_compression_results.md](queen_compression_results.md) —
  Markdown report written by the runner.

## Legacy Huffman pipeline

The original Huffman + RVQ pipeline (the baseline QUEEN improves on)
remains usable through `main.py` and `encoder.py` / `decoder.py`.
