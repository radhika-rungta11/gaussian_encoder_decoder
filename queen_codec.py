"""
QUEEN: QUantized Efficient ENcoding for 4D Gaussian splatting scenes.

Implements the QUEEN-style compression pipeline (Girish et al., NeurIPS 2024):

    raw float scene
        |
        v  (1) opacity-based sparsity gating
        v  (2) per-attribute quantization
        |       * xyz                 16-bit per-axis uniform
        |       * motion              10-bit per-axis uniform
        |       * scale               10-bit per-axis log-uniform
        |       * rotation            10-bit per-channel uniform (sign-canonicalized)
        |       * opacity / tcen      8-bit uniform
        |       * tsca                8-bit uniform
        |       * omega               8-bit per-axis uniform
        |       * tfea                Residual VQ, 2 stages, 256 entries
        v  (3) entropy coding of the index payloads
        v       (lzma + zlib; smaller wins per attribute)
        |
        v
    .queen file  =  [magic | json header | blob bytes]

The format is fully self-describing — the JSON header records every
quantization parameter so the decoder needs no extra metadata.
"""

from __future__ import annotations

import json
import lzma
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2


QUEEN_MAGIC = b"QUEEN001"
HEADER_LEN_FMT = "<I"


# ---------------------------------------------------------------------------
# Entropy coding helpers
# ---------------------------------------------------------------------------


def _entropy_compress(buf: bytes) -> Tuple[bytes, str]:
    """Pick the smaller of lzma / zlib for this payload."""
    z = zlib.compress(buf, 9)
    x = lzma.compress(buf, preset=9 | lzma.PRESET_EXTREME)
    if len(x) <= len(z):
        return x, "lzma"
    return z, "zlib"


def _entropy_decompress(buf: bytes, method: str) -> bytes:
    if method == "lzma":
        return lzma.decompress(buf)
    if method == "zlib":
        return zlib.decompress(buf)
    raise ValueError(f"Unknown entropy method {method!r}")


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def _q_dtype(bits: int) -> np.dtype:
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    return np.uint32


def _quantize_uniform(arr: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis (last-axis) min-max uniform quantization."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    levels = (1 << bits) - 1
    if arr.ndim == 1:
        mn = np.array([arr.min()], dtype=np.float32)
        mx = np.array([arr.max()], dtype=np.float32)
        span = max(float(mx[0] - mn[0]), 1e-12)
        normalized = (arr - mn[0]) / span
    else:
        mn = arr.min(axis=0).astype(np.float32)
        mx = arr.max(axis=0).astype(np.float32)
        span = np.maximum(mx - mn, 1e-12)
        normalized = (arr - mn[None, :]) / span[None, :]
    q = np.rint(normalized * levels).clip(0, levels).astype(_q_dtype(bits))
    return q, mn, mx


def _dequantize_uniform(q: np.ndarray, mn: np.ndarray, mx: np.ndarray, bits: int, ndim: int) -> np.ndarray:
    levels = (1 << bits) - 1
    normalized = q.astype(np.float32) / levels
    if ndim == 1:
        return mn[0] + normalized * (mx[0] - mn[0])
    return mn[None, :] + normalized * (mx - mn)[None, :]


def _quantize_log_uniform(arr: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-domain quantization for strictly-positive attributes (scale)."""
    arr = np.maximum(np.asarray(arr, dtype=np.float32), 1e-8)
    return _quantize_uniform(np.log(arr), bits)


def _dequantize_log_uniform(q, mn, mx, bits, ndim):
    return np.exp(_dequantize_uniform(q, mn, mx, bits, ndim)).astype(np.float32)


def _canonicalize_quaternion(rot: np.ndarray) -> np.ndarray:
    """Normalize and flip sign so the leading component is non-negative."""
    rot = np.asarray(rot, dtype=np.float32)
    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    rot = rot / norm
    sign = np.where(rot[:, :1] < 0, -1.0, 1.0).astype(np.float32)
    return rot * sign


# ---------------------------------------------------------------------------
# Residual vector quantization (used for tfea — the highest-dim attribute)
# ---------------------------------------------------------------------------


def _train_rvq(
    data: np.ndarray, stages: int = 2, entries: int = 256, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Train RVQ codebooks layer-by-layer using k-means on residuals."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    rng = np.random.default_rng(seed)
    codebooks: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    residual = data.copy()

    for stage in range(stages):
        n_pts = residual.shape[0]
        k = min(entries, n_pts)
        init = residual[rng.choice(n_pts, size=k, replace=False)]
        if k < entries:
            pad = np.tile(init[:1], (entries - k, 1))
            init = np.concatenate([init, pad], axis=0)
        codebook, labels = kmeans2(residual, init, iter=15, minit="matrix", seed=seed + stage)
        codebooks.append(codebook.astype(np.float32))
        labels = labels.astype(_q_dtype(int(np.ceil(np.log2(entries)))))
        indices.append(labels)
        residual = residual - codebook[labels]

    codebook_arr = np.stack(codebooks, axis=0).astype(np.float16)
    index_arr = np.stack(indices, axis=1)
    return codebook_arr, index_arr


def _decode_rvq(codebooks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    codebooks = np.asarray(codebooks, dtype=np.float32)
    stages = indices.shape[1]
    out = np.zeros((indices.shape[0], codebooks.shape[-1]), dtype=np.float32)
    for s in range(stages):
        out += codebooks[s, indices[:, s]]
    return out


# ---------------------------------------------------------------------------
# Sparsity gating (the "QUE" stage of QUEEN — bandwidth saved by pruning)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_sparsity_gate(scene, opacity_threshold: float = 1.0 / 255.0) -> Tuple[Any, np.ndarray]:
    """
    Drop Gaussians whose effective alpha (sigmoid of stored opacity logit) is
    below the renderer's quantization step. These contribute nothing visually
    yet cost bits — exactly what QUEEN's gating step removes.
    """
    if scene.opacity is None:
        return scene, np.ones((scene.xyz.shape[0],), dtype=bool)

    alpha = _sigmoid(scene.opacity.reshape(-1).astype(np.float32))
    keep_mask = alpha >= opacity_threshold
    if keep_mask.all():
        return scene, keep_mask

    filtered = _filter_scene(scene, keep_mask)
    return filtered, keep_mask


def _filter_scene(scene, mask: np.ndarray):
    out = type(scene)()
    n = int(mask.sum())
    for name in vars(scene):
        value = getattr(scene, name)
        if isinstance(value, np.ndarray) and value.shape[:1] == (mask.shape[0],) and value.shape[0] != 0:
            setattr(out, name, value[mask])
        else:
            setattr(out, name, value)
    return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


@dataclass
class _Blob:
    name: str
    raw_bytes: bytes
    bytes_compressed: bytes = b""
    method: str = "raw"

    @property
    def compressed_size(self) -> int:
        return len(self.bytes_compressed) if self.bytes_compressed else len(self.raw_bytes)


@dataclass
class QueenEncoderConfig:
    opacity_threshold: float = 1.0 / 255.0
    xyz_bits: int = 16
    motion_bits: int = 10
    scale_bits: int = 10
    rotation_bits: int = 10
    opacity_bits: int = 8
    tcen_bits: int = 8
    tsca_bits: int = 8
    omega_bits: int = 8
    tfea_rvq_stages: int = 2
    tfea_rvq_entries: int = 256
    rgb_bits: int = 8


@dataclass
class QueenEncodingReport:
    num_points_in: int
    num_points_out: int
    raw_float32_bytes: int
    output_file_bytes: int = 0
    header_bytes: int = 0
    per_attribute: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def encode_scene(scene, output_path: str, config: Optional[QueenEncoderConfig] = None) -> QueenEncodingReport:
    cfg = config or QueenEncoderConfig()
    num_in = int(scene.xyz.shape[0])
    raw_float32_bytes = _estimate_raw_float32_bytes(scene)

    gated_scene, keep_mask = apply_sparsity_gate(scene, cfg.opacity_threshold)
    num_out = int(gated_scene.xyz.shape[0])

    header: Dict[str, Any] = {
        "format": "queen-v1",
        "config": vars(cfg),
        "num_points_in": num_in,
        "num_points_out": num_out,
        "kept_fraction": num_out / max(num_in, 1),
        "attributes": {},
    }
    blobs: List[_Blob] = []

    def _add_uniform(name: str, arr: np.ndarray, bits: int, log: bool = False) -> None:
        if arr is None:
            return
        ndim = arr.ndim
        if log:
            q, mn, mx = _quantize_log_uniform(arr, bits)
        else:
            q, mn, mx = _quantize_uniform(arr, bits)
        payload = q.tobytes()
        compressed, method = _entropy_compress(payload)
        blob = _Blob(name=f"{name}_q", raw_bytes=payload, bytes_compressed=compressed, method=method)
        blobs.append(blob)
        header["attributes"][name] = {
            "kind": "log_uniform" if log else "uniform",
            "bits": bits,
            "dtype": str(q.dtype),
            "shape": list(arr.shape),
            "ndim": ndim,
            "min": mn.tolist(),
            "max": mx.tolist(),
            "indices_blob": blob.name,
            "indices_method": method,
            "indices_bytes": len(compressed),
            "indices_raw_bytes": len(payload),
        }

    def _add_rvq(name: str, arr: np.ndarray, stages: int, entries: int) -> None:
        if arr is None:
            return
        codebooks, indices = _train_rvq(arr.astype(np.float32), stages=stages, entries=entries)
        cb_bytes = codebooks.tobytes()
        idx_bytes = indices.tobytes()
        cb_comp, cb_method = _entropy_compress(cb_bytes)
        idx_comp, idx_method = _entropy_compress(idx_bytes)
        cb_blob = _Blob(f"{name}_codebook", cb_bytes, cb_comp, cb_method)
        idx_blob = _Blob(f"{name}_indices", idx_bytes, idx_comp, idx_method)
        blobs.append(cb_blob)
        blobs.append(idx_blob)
        header["attributes"][name] = {
            "kind": "rvq",
            "stages": int(stages),
            "entries": int(entries),
            "codebook_shape": list(codebooks.shape),
            "codebook_dtype": str(codebooks.dtype),
            "codebook_blob": cb_blob.name,
            "codebook_method": cb_method,
            "codebook_bytes": len(cb_comp),
            "codebook_raw_bytes": len(cb_bytes),
            "indices_shape": list(indices.shape),
            "indices_dtype": str(indices.dtype),
            "indices_blob": idx_blob.name,
            "indices_method": idx_method,
            "indices_bytes": len(idx_comp),
            "indices_raw_bytes": len(idx_bytes),
        }

    # ----- geometry & motion -----
    _add_uniform("xyz", gated_scene.xyz, cfg.xyz_bits)
    if gated_scene.motion is not None:
        _add_uniform("motion", gated_scene.motion, cfg.motion_bits)

    # ----- per-Gaussian appearance -----
    if gated_scene.opacity is not None:
        _add_uniform("opacity", gated_scene.opacity.reshape(-1), cfg.opacity_bits)

    if gated_scene.scale is not None:
        _add_uniform("scale", gated_scene.scale, cfg.scale_bits, log=True)

    if gated_scene.rotation is not None:
        rotation = _canonicalize_quaternion(gated_scene.rotation)
        _add_uniform("rotation", rotation, cfg.rotation_bits)

    if gated_scene.rgb is not None and not np.allclose(gated_scene.rgb, 1.0):
        _add_uniform("rgb", gated_scene.rgb, cfg.rgb_bits)

    # ----- temporal attributes -----
    if gated_scene.tcen is not None:
        _add_uniform("tcen", gated_scene.tcen, cfg.tcen_bits)
    if gated_scene.tsca is not None:
        _add_uniform("tsca", gated_scene.tsca, cfg.tsca_bits)
    if gated_scene.omega is not None:
        _add_uniform("omega", gated_scene.omega, cfg.omega_bits)
    if gated_scene.tfea is not None:
        _add_rvq("tfea", gated_scene.tfea, cfg.tfea_rvq_stages, cfg.tfea_rvq_entries)

    header_bytes = _serialize_blobs_into_file(output_path, header, blobs)

    report = QueenEncodingReport(
        num_points_in=num_in,
        num_points_out=num_out,
        raw_float32_bytes=raw_float32_bytes,
        header_bytes=header_bytes,
    )
    for name, info in header["attributes"].items():
        compressed = int(info.get("indices_bytes", 0)) + int(info.get("codebook_bytes", 0))
        raw = int(info.get("indices_raw_bytes", 0)) + int(info.get("codebook_raw_bytes", 0))
        report.per_attribute[name] = {
            "kind": info["kind"],
            "compressed_bytes": compressed,
            "raw_quantized_bytes": raw,
            "bits_per_point": (compressed * 8.0) / max(num_out, 1),
        }

    import os
    report.output_file_bytes = os.path.getsize(output_path)
    return report


def _serialize_blobs_into_file(path: str, header: Dict[str, Any], blobs: List[_Blob]) -> int:
    # Lay out blob offsets relative to the start of the blob section
    offset = 0
    blob_map = {}
    for blob in blobs:
        payload = blob.bytes_compressed or blob.raw_bytes
        blob_map[blob.name] = {
            "offset": offset,
            "length": len(payload),
            "method": blob.method,
        }
        offset += len(payload)
    header["blobs"] = blob_map

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(QUEEN_MAGIC)
        fh.write(struct.pack(HEADER_LEN_FMT, len(header_bytes)))
        fh.write(header_bytes)
        for blob in blobs:
            fh.write(blob.bytes_compressed or blob.raw_bytes)
    return len(header_bytes)


def _estimate_raw_float32_bytes(scene) -> int:
    """How many bytes a naive float32 dump of the same fields would take."""
    total = 0
    for name in ["xyz", "motion", "opacity", "scale", "rotation", "rgb", "tcen", "tsca", "omega", "tfea"]:
        value = getattr(scene, name, None)
        if isinstance(value, np.ndarray):
            total += int(np.prod(value.shape)) * 4
    return total


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def decode_scene(path: str):
    """Inverse of encode_scene. Returns a SceneProperties from decoder.py."""
    from decoder import SceneProperties

    with open(path, "rb") as fh:
        magic = fh.read(len(QUEEN_MAGIC))
        if magic != QUEEN_MAGIC:
            raise ValueError(f"Not a QUEEN file: {path} (magic {magic!r})")
        header_len = struct.unpack(HEADER_LEN_FMT, fh.read(struct.calcsize(HEADER_LEN_FMT)))[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        blob_bytes = fh.read()

    blob_table = header["blobs"]

    def _read_blob(blob_name: str) -> bytes:
        info = blob_table[blob_name]
        raw = blob_bytes[info["offset"]:info["offset"] + info["length"]]
        return _entropy_decompress(raw, info["method"])

    scene = SceneProperties()
    num_points = header["num_points_out"]

    for name, info in header["attributes"].items():
        kind = info["kind"]
        if kind in ("uniform", "log_uniform"):
            q_bytes = _read_blob(info["indices_blob"])
            q = np.frombuffer(q_bytes, dtype=np.dtype(info["dtype"])).reshape(info["shape"])
            mn = np.asarray(info["min"], dtype=np.float32)
            mx = np.asarray(info["max"], dtype=np.float32)
            if kind == "uniform":
                value = _dequantize_uniform(q, mn, mx, info["bits"], info["ndim"]).astype(np.float32)
            else:
                value = _dequantize_log_uniform(q, mn, mx, info["bits"], info["ndim"]).astype(np.float32)
        elif kind == "rvq":
            cb_bytes = _read_blob(info["codebook_blob"])
            idx_bytes = _read_blob(info["indices_blob"])
            codebooks = np.frombuffer(cb_bytes, dtype=np.dtype(info["codebook_dtype"])).reshape(info["codebook_shape"])
            indices = np.frombuffer(idx_bytes, dtype=np.dtype(info["indices_dtype"])).reshape(info["indices_shape"])
            value = _decode_rvq(codebooks, indices).astype(np.float32)
        else:
            raise ValueError(f"Unknown attribute kind {kind!r} for {name}")

        if name == "rotation":
            value = _canonicalize_quaternion(value)
        if name == "opacity":
            value = value.reshape(-1, 1)
        setattr(scene, name, value)

    if scene.xyz is not None and scene.xyz.shape[0] != num_points:
        raise ValueError(
            f"Decoded xyz has {scene.xyz.shape[0]} points but header says {num_points}"
        )
    if scene.rgb is None and scene.xyz is not None:
        scene.rgb = np.ones((scene.xyz.shape[0], 3), dtype=np.float32)

    return scene, header


# ---------------------------------------------------------------------------
# Quality metric
# ---------------------------------------------------------------------------


def attribute_psnr(original: np.ndarray, decoded: np.ndarray) -> float:
    if original is None or decoded is None:
        return float("nan")
    o = np.asarray(original, dtype=np.float32).reshape(-1)
    d = np.asarray(decoded, dtype=np.float32).reshape(-1)
    if o.shape != d.shape:
        # Sparsity gate may have dropped points, so trim original to match
        m = min(o.size, d.size)
        o = o[:m]
        d = d[:m]
    mse = float(np.mean((o - d) ** 2))
    if mse <= 1e-20:
        return float("inf")
    peak = float(np.max(np.abs(o))) or 1.0
    return 20.0 * np.log10(peak / np.sqrt(mse))
