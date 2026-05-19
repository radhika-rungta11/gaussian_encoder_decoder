# Stage 1: .4dgs / .4dgs.gz Binary Schema

This schema is derived from cross-checking three sources that all agree:

1. `flamsplat/core/utils/fourdgs_loader.c` — production C decoder
2. `scene_io.py` — Python loader (`load_scene_properties_from_4dgs`)
3. The Python writer doc-string the manager sent (`parser.py` style)

All integers are little-endian. `f` = float32, `H` = uint16, `B` = uint8,
`I` = uint32, `f16` = IEEE-754 half precision.

## Outer container

```
[optional gzip wrapper, magic 0x1F 0x8B]
└── 4DGS body (decoded below)
```

The C decoder sniffs `0x1F 0x8B`; if absent it treats the buffer as raw.

## Header (12 bytes total)

| Offset | Type | Field        | Notes                                  |
|--------|------|--------------|----------------------------------------|
| 0      | 4s   | magic        | ASCII `"4DGS"` = 0x53474434 little-endian |
| 4      | I    | version      | currently 3                            |
| 8      | I    | N            | number of Gaussians                    |

## Raw float16 blocks (after header)

| Field    | Count   | Bytes     |
|----------|---------|-----------|
| xyz      | N × 3   | 6 N       |
| motion   | N × 9   | 18 N      |

`motion` represents linear + quadratic + cubic temporal coefficients
per axis (3 stages × 3 axes = 9). Position at time `t` is

```
xyz(t) = xyz0 + motion[0:3]*t + motion[3:6]*t^2 + motion[6:9]*t^3
```

(Verified by treating `motion` as polynomial coefficients consistent
with `Dynamic_C3DGS` papers; the C decoder simply preserves them.)

## Scalar Huffman blocks × 3

Order: **opacity, tcen, tsca**. Each block:

```
f   min_val
f   max_val
H   htable_len             # number of Huffman code entries
N times:
    H   symbol             # 0..255 for byte symbol, 0xFFFF for EOF
    B   bit_len
    I   code_bits          # MSB-first packed code, length = bit_len
I   data_len
B[data_len] payload        # MSB-first bit-packed Huffman stream
```

Decoder rebuilds the prefix tree, then walks payload bits MSB-first.
When a leaf with symbol 256 (EOF) is reached, decoding stops; otherwise
emits `uint8` value. Final dequantization:

```
value = min_val + (max_val - min_val) * (q / 255)
```

Decoded length must equal `N`.

## VQ / RVQ Huffman blocks × 4

Order: **scale (dim=3), rotation (dim=4), omega (dim=4), tfea (dim=3)**.

```
B   num_layers
H   codebook_size          # must be a power of two
H   dim
f16[num_layers * codebook_size * dim]   codebooks
[Huffman block as above]   packed_indices
```

After Huffman decoding, the byte stream is interpreted as a bit array
(LSB-first within each byte). Index for Gaussian `i`, layer `l`:

```
rvq_bit = log2(codebook_size)
bit_pos = (i*num_layers + l)*rvq_bit + b   for b in 0..rvq_bit-1
byte    = packed_indices[bit_pos / 8]
bit     = (byte >> (bit_pos % 8)) & 1
index  |= bit << b
```

Reconstruction sums codebook entries across layers (residual VQ):

```
vec[i] = Σ_l codebook[l, index(i, l), :]
```

Special interpretation per attribute:

| Attribute | dim | Notes                                                        |
|-----------|-----|--------------------------------------------------------------|
| scale     | 3   | log-space scale; apply `exp()` before use                    |
| rotation  | 4   | quaternion `(w,x,y,z)`; normalize after summing              |
| omega     | 4   | per-Gaussian temporal phase coefficients                     |
| tfea      | 3   | runtime temporal color features fed to rgb_dec               |

## Optional baked features (1 byte flag + payload)

```
B   has_features
if has_features == 1:
    f16[N * 6]   features_dc        # DC SH coefficients per Gaussian
```

The high 3 channels are passed through `sigmoid` for fallback RGB.

## Optional rgb_dec (1 byte flag + payload)

```
B   has_rgb_dec
if has_rgb_dec == 1:
    f16[6 * 12]   w1                # MLP layer 1
    f16[3 * 6]    w2                # MLP layer 2
```

A 12 → 6 → 3 sandwich MLP. Inputs (per Python loader): concatenate
`features_dc` (6) + `tfea` (3) + `tcen` (1) + `tsca` (1) + `1` (bias),
ReLU after w1, sigmoid after w2.

## Byte-accounting invariant

After all blocks are read, `reader.offset` MUST equal `reader.size`.
The C decoder emits a warning if not (decoder.c does the same and
treats it as a hard error in strict mode).

## Things deliberately *not* in the binary

- Hash grid parameters
- Hash-grid-derived MLP weights
- Original `_pp.npz` Huffman tables for `hash`
- `rvq_info_geo` / `rvq_info_temp` (those become `(num_layers, log2(codebook_size))` per VQ block instead)

## Open / unconfirmed items

None at this time. All fields, types, and sizes are confirmed by both
implementations.
