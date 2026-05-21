"""
Run QUEEN compression on a Gaussian-splatting NPZ and print a report.

Usage:
    python3.12 compress_queen.py                       # uses point_cloud_pp.npz
    python3.12 compress_queen.py path/to/scene.npz
    python3.12 compress_queen.py scene.npz --out compressed.queen --markdown results.md
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import numpy as np

import queen_codec
from scene_io import load_scene_properties


ATTRIBUTE_ORDER = [
    "xyz", "motion", "opacity", "scale", "rotation",
    "rgb", "tcen", "tsca", "omega", "tfea",
]


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _row(label: str, *cols, width: int = 16) -> str:
    return "  " + label.ljust(22) + "".join(str(c).ljust(width) for c in cols)


def _build_text_report(
    input_path: str,
    output_path: str,
    report: queen_codec.QueenEncodingReport,
    psnr: Dict[str, float],
    encode_seconds: float,
    decode_seconds: float,
) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  QUEEN compression report")
    lines.append("=" * 78)
    lines.append(f"  input file       : {input_path}")
    lines.append(f"  output file      : {output_path}")
    lines.append(f"  points in        : {report.num_points_in:,}")
    lines.append(f"  points kept      : {report.num_points_out:,} "
                 f"({100.0 * report.num_points_out / max(report.num_points_in, 1):.2f} %)")
    lines.append(f"  encode time      : {encode_seconds:.2f} s")
    lines.append(f"  decode time      : {decode_seconds:.2f} s")
    lines.append("")

    input_size = os.path.getsize(input_path)
    output_size = report.output_file_bytes
    raw_size = report.raw_float32_bytes

    lines.append("-" * 78)
    lines.append("  Overall size")
    lines.append("-" * 78)
    lines.append(_row("source NPZ on disk",  _fmt_bytes(input_size)))
    lines.append(_row("raw float32 baseline", _fmt_bytes(raw_size)))
    lines.append(_row("QUEEN .queen file",    _fmt_bytes(output_size)))
    lines.append("")
    lines.append(_row("vs source NPZ",        f"{output_size / input_size * 100:.2f} %",
                      f"{input_size / max(output_size, 1):.2f}x smaller"))
    lines.append(_row("vs raw float32",       f"{output_size / raw_size * 100:.2f} %",
                      f"{raw_size / max(output_size, 1):.2f}x smaller"))
    lines.append("")

    lines.append("-" * 78)
    lines.append("  Per-attribute breakdown (compressed bytes)")
    lines.append("-" * 78)
    lines.append(_row("attribute", "kind", "comp size", "bits/point", "PSNR (dB)"))
    for name in ATTRIBUTE_ORDER:
        if name not in report.per_attribute:
            continue
        info = report.per_attribute[name]
        psnr_val = psnr.get(name)
        psnr_label = "n/a" if psnr_val is None else (
            "inf" if not np.isfinite(psnr_val) else f"{psnr_val:.2f}"
        )
        lines.append(_row(
            name,
            info["kind"],
            _fmt_bytes(info["compressed_bytes"]),
            f"{info['bits_per_point']:.2f}",
            psnr_label,
        ))
    lines.append("")
    lines.append(_row("json header", "metadata", _fmt_bytes(report.header_bytes), "-", "-"))
    lines.append("=" * 78)
    return "\n".join(lines)


def _build_markdown_report(
    input_path: str,
    output_path: str,
    report: queen_codec.QueenEncodingReport,
    psnr: Dict[str, float],
    encode_seconds: float,
    decode_seconds: float,
) -> str:
    input_size = os.path.getsize(input_path)
    output_size = report.output_file_bytes
    raw_size = report.raw_float32_bytes

    md = []
    md.append("# QUEEN compression report\n")
    md.append("**QUEEN (QUantized Efficient ENcoding)** is applied to the "
              "Gaussian splatting scene as a quantization-sparsity pipeline "
              "(per Girish et al., NeurIPS 2024). Pipeline stages:\n")
    md.append("1. **Sparsity gating** — Gaussians whose effective alpha "
              "(`sigmoid(opacity)`) falls below `1/255` are dropped because "
              "they cannot contribute to a rendered pixel.\n")
    md.append("2. **Per-attribute quantization** — `xyz` uses 16-bit per-axis "
              "min-max; `scale` uses log-domain quantization; `rotation` is "
              "sign-canonicalized then per-channel quantized; `tfea` is fed "
              "through a 2-stage residual VQ with k-means codebooks (256 "
              "entries each); the remaining attributes use 8–10-bit uniform "
              "quantization.\n")
    md.append("3. **Entropy coding** — each quantized payload is run through "
              "both LZMA and zlib; the smaller one is kept. The JSON header "
              "records every quantization parameter so the file is fully "
              "self-describing.\n")
    md.append("")
    md.append(f"- Input file: `{input_path}`")
    md.append(f"- Output file: `{output_path}`")
    md.append(f"- Points in: **{report.num_points_in:,}**")
    md.append(f"- Points kept after sparsity gating: **{report.num_points_out:,}** "
              f"({100.0 * report.num_points_out / max(report.num_points_in, 1):.2f} %)")
    md.append(f"- Encode time: **{encode_seconds:.2f} s** &nbsp;&nbsp; "
              f"Decode time: **{decode_seconds:.2f} s**\n")

    md.append("## Overall compression\n")
    md.append("| Reference | Size | QUEEN size | Ratio |")
    md.append("|---|---|---|---|")
    md.append(f"| Source NPZ on disk (Huffman baseline) | {_fmt_bytes(input_size)} | "
              f"{_fmt_bytes(output_size)} | "
              f"**{input_size / max(output_size, 1):.2f}x smaller** "
              f"({output_size / input_size * 100:.2f} %) |")
    md.append(f"| Raw float32 dump | {_fmt_bytes(raw_size)} | {_fmt_bytes(output_size)} | "
              f"**{raw_size / max(output_size, 1):.2f}x smaller** "
              f"({output_size / raw_size * 100:.2f} %) |\n")

    md.append("## Per-attribute breakdown\n")
    md.append("| Attribute | Pipeline stage | Compressed size | Bits / point | PSNR (dB) |")
    md.append("|---|---|---|---|---|")
    for name in ATTRIBUTE_ORDER:
        if name not in report.per_attribute:
            continue
        info = report.per_attribute[name]
        psnr_val = psnr.get(name)
        psnr_label = "n/a" if psnr_val is None else (
            "lossless" if not np.isfinite(psnr_val) else f"{psnr_val:.2f}"
        )
        md.append(f"| `{name}` | {info['kind']} | {_fmt_bytes(info['compressed_bytes'])} | "
                  f"{info['bits_per_point']:.2f} | {psnr_label} |")
    md.append(f"| _json header_ | metadata | {_fmt_bytes(report.header_bytes)} | – | – |\n")

    md.append("## How to reproduce\n")
    md.append("```bash")
    md.append(f"python3.12 compress_queen.py {input_path} --out {output_path}")
    md.append("```")
    return "\n".join(md)


def _gather_psnr(original_scene, decoded_scene, kept_count: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in ATTRIBUTE_ORDER:
        orig = getattr(original_scene, name, None)
        decoded = getattr(decoded_scene, name, None)
        if orig is None or decoded is None:
            continue
        orig_match = orig[:kept_count] if orig.shape[0] >= kept_count else orig
        if name == "rotation":
            # q and -q represent the same rotation; canonicalize both sides
            # so PSNR reflects only quantization error, not sign flips.
            orig_match = queen_codec._canonicalize_quaternion(orig_match)
            decoded = queen_codec._canonicalize_quaternion(decoded)
        out[name] = queen_codec.attribute_psnr(orig_match, decoded)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="QUEEN compression for Gaussian splat scenes.")
    parser.add_argument("input_path", nargs="?", default="point_cloud_pp.npz",
                        help="Input .npz scene file (default: point_cloud_pp.npz).")
    parser.add_argument("--out", default="point_cloud_pp.queen",
                        help="Output .queen file path.")
    parser.add_argument("--markdown", default="queen_compression_results.md",
                        help="Path for the Markdown report (set empty to skip).")
    parser.add_argument("--opacity-threshold", type=float, default=1.0 / 255.0,
                        help="Sparsity gate threshold on sigmoid(opacity).")
    parser.add_argument("--no-aux", action="store_true",
                        help="Skip decoding auxiliary properties (scale/rotation/etc.).")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise SystemExit(f"Input file not found: {args.input_path}")

    print(f"Loading scene from {args.input_path}...")
    scene = load_scene_properties(args.input_path, decode_auxiliary_properties=not args.no_aux)

    cfg = queen_codec.QueenEncoderConfig(opacity_threshold=args.opacity_threshold)

    print("Running QUEEN encoder...")
    t0 = time.perf_counter()
    report = queen_codec.encode_scene(scene, args.out, config=cfg)
    encode_seconds = time.perf_counter() - t0

    print("Round-tripping through the QUEEN decoder for quality check...")
    t0 = time.perf_counter()
    decoded_scene, _header = queen_codec.decode_scene(args.out)
    decode_seconds = time.perf_counter() - t0

    psnr = _gather_psnr(scene, decoded_scene, report.num_points_out)

    text_report = _build_text_report(args.input_path, args.out, report, psnr, encode_seconds, decode_seconds)
    print()
    print(text_report)

    if args.markdown:
        md = _build_markdown_report(args.input_path, args.out, report, psnr, encode_seconds, decode_seconds)
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"\nMarkdown report written to {args.markdown}")


if __name__ == "__main__":
    main()
