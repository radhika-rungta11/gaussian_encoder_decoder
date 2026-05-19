import argparse
from dataclasses import dataclass

import numpy as np

from quantize import quantize_scalar, quantize_vector
from scene_io import load_scene_properties


ENCODER_FORMAT = "gaussian_encoder_decoder_v1"


@dataclass
class EncoderOutputs:
    blobs: dict


def _normalize_quaternion(rotation):
    rotation = np.asarray(rotation, dtype=np.float32)
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return rotation / norm


def encode_scene_properties(scene_properties):
    """
    Encode scene properties into a compact NPZ format that our decoder can load directly.
    This codec is intentionally simple and robust:
      - geometry and motion: uint16 min-max quantization
      - opacity / RGB: uint8 min-max quantization
      - scale: uint16 min-max quantization
      - rotation: float16 normalized quaternion
    """
    blobs = {
        "codec_format": np.array(ENCODER_FORMAT),
    }

    if scene_properties.xyz is None:
        raise ValueError("Cannot encode a scene without xyz positions.")

    xyz_q, xyz_min, xyz_max = quantize_vector(scene_properties.xyz, num_bits=16)
    blobs["xyz_q"] = xyz_q
    blobs["xyz_min"] = xyz_min
    blobs["xyz_max"] = xyz_max

    if scene_properties.motion is not None:
        motion_q, motion_min, motion_max = quantize_vector(scene_properties.motion, num_bits=16)
        blobs["motion_q"] = motion_q
        blobs["motion_min"] = motion_min
        blobs["motion_max"] = motion_max

    if scene_properties.opacity is not None:
        opacity_q, opacity_minmax = quantize_scalar(scene_properties.opacity.reshape(-1), num_bits=8)
        blobs["opacity_q"] = opacity_q
        blobs["opacity_minmax"] = opacity_minmax

    if scene_properties.scale is not None:
        scale_q, scale_min, scale_max = quantize_vector(scene_properties.scale, num_bits=16)
        blobs["scale_q"] = scale_q
        blobs["scale_min"] = scale_min
        blobs["scale_max"] = scale_max

    if scene_properties.rotation is not None:
        blobs["rotation_f16"] = _normalize_quaternion(scene_properties.rotation).astype(np.float16)

    if scene_properties.rgb is not None:
        rgb_q, rgb_min, rgb_max = quantize_vector(scene_properties.rgb, num_bits=8)
        blobs["rgb_q"] = rgb_q
        blobs["rgb_min"] = rgb_min
        blobs["rgb_max"] = rgb_max

    for name in ["tcen", "tsca", "omega", "tfea"]:
        value = getattr(scene_properties, name, None)
        if value is None:
            continue
        if value.ndim == 1:
            value_q, value_minmax = quantize_scalar(value, num_bits=16)
            blobs[f"{name}_q"] = value_q
            blobs[f"{name}_minmax"] = value_minmax
        else:
            value_q, value_min, value_max = quantize_vector(value, num_bits=16)
            blobs[f"{name}_q"] = value_q
            blobs[f"{name}_min"] = value_min
            blobs[f"{name}_max"] = value_max

    return EncoderOutputs(blobs=blobs)


def save_encoded_scene(scene_properties, output_path):
    encoded = encode_scene_properties(scene_properties)
    np.savez_compressed(output_path, **encoded.blobs)
    return output_path


_RAW_FLOAT32_ATTRIBUTES = (
    ("xyz", 3),
    ("motion", 9),
    ("opacity", 1),
    ("scale", 3),
    ("rotation", 4),
    ("rgb", 3),
    ("tcen", 1),
    ("tsca", 1),
    ("omega", 6),
    ("tfea", 3),
)


def compute_raw_float32_baseline_bytes(scene_properties):
    """
    Compute the size, in bytes, of storing every populated scene attribute
    as raw float32. This is the apples-to-apples baseline that the C3DGS
    paper compares its compression ratio against.
    """
    if scene_properties.xyz is None:
        return 0
    num_points = scene_properties.xyz.shape[0]

    total = 0
    for name, default_channels in _RAW_FLOAT32_ATTRIBUTES:
        value = getattr(scene_properties, name, None)
        if value is None:
            continue
        arr = np.asarray(value)
        channels = arr.shape[-1] if arr.ndim >= 2 else 1
        total += num_points * channels * 4
    return total


def _human_bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0


_PLY_EQUIVALENT_FLOATS_PER_GAUSSIAN = (
    3   # xyz
    + 3   # scale
    + 4   # rotation
    + 1   # opacity
    + 3   # features_dc (SH degree 0)
    + 45  # features_rest (SH degrees 1..3, 15 coeffs * 3 channels)
    + 9   # motion (4DGS extension)
    + 6   # omega (4DGS extension)
    + 1   # tcen
    + 1   # tsca
    + 3   # tfea
)


def compute_ply_equivalent_baseline_bytes(num_points):
    """
    Paper-style baseline: full float32 PLY with SH degree 3 plus the
    4DGS-specific motion / temporal channels.
    """
    return num_points * _PLY_EQUIVALENT_FLOATS_PER_GAUSSIAN * 4


def collect_size_report(scene_properties, raw_input_path, encoded_output_path):
    """
    Compute the headline metrics the manager will want to see. Returns a
    dict so callers can print, write to CSV, or aggregate across many runs.
    """
    import os

    raw_baseline = compute_raw_float32_baseline_bytes(scene_properties)
    raw_input_size = os.path.getsize(raw_input_path)
    encoded_size = os.path.getsize(encoded_output_path)
    num_points = scene_properties.xyz.shape[0] if scene_properties.xyz is not None else 0
    ply_baseline = compute_ply_equivalent_baseline_bytes(num_points)

    return {
        "input_path": raw_input_path,
        "encoded_path": encoded_output_path,
        "num_gaussians": num_points,
        "raw_float32_bytes": raw_baseline,
        "ply_equivalent_bytes": ply_baseline,
        "input_file_bytes": raw_input_size,
        "encoded_file_bytes": encoded_size,
        "ratio_vs_raw_float32": (raw_baseline / encoded_size) if encoded_size > 0 else 0.0,
        "ratio_vs_ply_equivalent": (ply_baseline / encoded_size) if encoded_size > 0 else 0.0,
        "ratio_vs_input_file": (raw_input_size / encoded_size) if encoded_size > 0 else 0.0,
    }


def format_size_report_text(report):
    lines = []
    lines.append("=" * 70)
    lines.append(" Compression report")
    lines.append("=" * 70)
    lines.append(f" Input file                             : {report['input_path']}")
    lines.append(f" Encoded output                         : {report['encoded_path']}")
    lines.append(f" Gaussians (N)                          : {report['num_gaussians']:,}")
    lines.append(f" Raw float32 attributes (loaded only)   : {_human_bytes(report['raw_float32_bytes'])}")
    lines.append(f" PLY-equivalent baseline (SH deg 3)     : {_human_bytes(report['ply_equivalent_bytes'])}")
    lines.append(f" Input file on disk                     : {_human_bytes(report['input_file_bytes'])}")
    lines.append(f" Encoder output on disk                 : {_human_bytes(report['encoded_file_bytes'])}")
    lines.append(f" Compression vs loaded float32          : {report['ratio_vs_raw_float32']:.2f}x")
    lines.append(f" Compression vs PLY-equivalent (paper)  : {report['ratio_vs_ply_equivalent']:.2f}x")
    lines.append(f" Compression vs input file              : {report['ratio_vs_input_file']:.2f}x")
    lines.append("=" * 70)
    return "\n".join(lines)


def write_size_report_csv(report, csv_path):
    """
    Two-column key/value CSV. Easy to import into Excel/Sheets for a bar chart.
    """
    import csv

    rows = [
        ("input_path", report["input_path"]),
        ("encoded_path", report["encoded_path"]),
        ("num_gaussians", report["num_gaussians"]),
        ("raw_float32_bytes", report["raw_float32_bytes"]),
        ("ply_equivalent_bytes", report["ply_equivalent_bytes"]),
        ("input_file_bytes", report["input_file_bytes"]),
        ("encoded_file_bytes", report["encoded_file_bytes"]),
        ("ratio_vs_raw_float32", f"{report['ratio_vs_raw_float32']:.4f}"),
        ("ratio_vs_ply_equivalent", f"{report['ratio_vs_ply_equivalent']:.4f}"),
        ("ratio_vs_input_file", f"{report['ratio_vs_input_file']:.4f}"),
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def write_sizes_for_bar_chart(report, csv_path):
    """
    Bar-chart-ready CSV: one row per bar.
    Open in Excel/Sheets -> Insert -> Bar Chart -> done.
    """
    import csv

    rows = [
        ("PLY equivalent (SH deg 3)", report["ply_equivalent_bytes"]),
        ("Raw float32 attributes", report["raw_float32_bytes"]),
        ("Input file (point_cloud_pp.npz)", report["input_file_bytes"]),
        ("Encoded output", report["encoded_file_bytes"]),
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(("label", "size_bytes", "size_kb", "size_mb"))
        for label, size in rows:
            writer.writerow((label, size, f"{size / 1024:.2f}", f"{size / (1024 * 1024):.4f}"))


def print_size_report(scene_properties, raw_input_path, encoded_output_path, results_dir=None):
    """
    Compute the size metrics, print them to stdout, and (if results_dir is set)
    write a human-readable .txt and two graph-ready .csv files into that dir.
    """
    import os

    report = collect_size_report(scene_properties, raw_input_path, encoded_output_path)
    text = format_size_report_text(report)
    print()
    print(text)
    print()

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        txt_path = os.path.join(results_dir, "compression_report.txt")
        kv_path = os.path.join(results_dir, "compression_report.csv")
        bar_path = os.path.join(results_dir, "compression_sizes_for_chart.csv")
        with open(txt_path, "w") as fh:
            fh.write(text + "\n")
        write_size_report_csv(report, kv_path)
        write_sizes_for_bar_chart(report, bar_path)
        print(f"Wrote {txt_path}")
        print(f"Wrote {kv_path}")
        print(f"Wrote {bar_path}  (use this one for the bar chart)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Encode a Gaussian scene into a compact NPZ archive.")
    parser.add_argument("input_path", help="Input .npz or .4dgs scene file")
    parser.add_argument("output_path", help="Output encoded .npz file")
    parser.add_argument(
        "--decode-aux",
        action="store_true",
        help="When the input is the original compressed NPZ, decode auxiliary Gaussian properties before re-encoding.",
    )
    args = parser.parse_args()

    scene = load_scene_properties(args.input_path, decode_auxiliary_properties=args.decode_aux)
    save_encoded_scene(scene, args.output_path)
    print(f"Saved encoded scene to {args.output_path}")


if __name__ == "__main__":
    main()
