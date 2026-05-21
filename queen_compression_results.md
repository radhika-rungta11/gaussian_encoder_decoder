# QUEEN compression report

**QUEEN (QUantized Efficient ENcoding)** is applied to the Gaussian splatting scene as a quantization-sparsity pipeline (per Girish et al., NeurIPS 2024). Pipeline stages:

1. **Sparsity gating** — Gaussians whose effective alpha (`sigmoid(opacity)`) falls below `1/255` are dropped because they cannot contribute to a rendered pixel.

2. **Per-attribute quantization** — `xyz` uses 16-bit per-axis min-max; `scale` uses log-domain quantization; `rotation` is sign-canonicalized then per-channel quantized; `tfea` is fed through a 2-stage residual VQ with k-means codebooks (256 entries each); the remaining attributes use 8–10-bit uniform quantization.

3. **Entropy coding** — each quantized payload is run through both LZMA and zlib; the smaller one is kept. The JSON header records every quantization parameter so the file is fully self-describing.


- Input file: `point_cloud_pp.npz`
- Output file: `point_cloud_pp.queen`
- Points in: **15,556**
- Points kept after sparsity gating: **15,556** (100.00 %)
- Encode time: **0.30 s** &nbsp;&nbsp; Decode time: **0.02 s**

## Overall compression

| Reference | Size | QUEEN size | Ratio |
|---|---|---|---|
| Source NPZ on disk (Huffman baseline) | 925.7 KB | 405.8 KB | **2.28x smaller** (43.84 %) |
| Raw float32 dump | 1.90 MB | 405.8 KB | **4.79x smaller** (20.87 %) |

## Per-attribute breakdown

| Attribute | Pipeline stage | Compressed size | Bits / point | PSNR (dB) |
|---|---|---|---|---|
| `xyz` | uniform | 75.2 KB | 39.60 | 107.97 |
| `motion` | uniform | 118.9 KB | 62.62 | 68.61 |
| `opacity` | uniform | 12.8 KB | 6.75 | lossless |
| `scale` | log_uniform | 4.6 KB | 2.42 | 76.21 |
| `rotation` | uniform | 80.3 KB | 42.28 | 18.13 |
| `rgb` | uniform | 38.3 KB | 20.19 | 59.34 |
| `tcen` | uniform | 14.9 KB | 7.83 | lossless |
| `tsca` | uniform | 13.8 KB | 7.27 | lossless |
| `omega` | uniform | 10.4 KB | 5.45 | 53.25 |
| `tfea` | rvq | 32.8 KB | 17.26 | 44.99 |
| _json header_ | metadata | 3.9 KB | – | – |

## How to reproduce

```bash
python3.12 compress_queen.py point_cloud_pp.npz --out point_cloud_pp.queen
```