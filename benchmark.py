"""
Run the encoder/decoder/compression pipeline on a batch of .npz checkpoints
and write one combined CSV row per file. Use this to prove the pipeline
generalises beyond `point_cloud_pp.npz`.

Examples:
  # Benchmark every .npz in a directory
  python3.12 benchmark.py --input-dir benchmark_inputs/

  # Or pass an explicit list
  python3.12 benchmark.py point_cloud_pp.npz another_scene.npz third.npz

  # With rendering (slower, produces a video per scene)
  python3.12 benchmark.py --input-dir benchmark_inputs/ --render
"""

import argparse
import csv
import glob
import html
import os
import time
import traceback

from encoder import (
    collect_size_report,
    format_size_report_text,
    save_encoded_scene,
)
from scene_io import load_scene_properties


def benchmark_one(input_path, output_dir, render=False, num_frames=10):
    """
    Run encode + reload + (optional) render for a single input .npz.
    Returns the report dict on success, or None on failure.
    """
    name = os.path.splitext(os.path.basename(input_path))[0]
    encoded_path = os.path.join(output_dir, f"{name}.encoded.npz")

    t_start = time.time()
    print(f"\n[BENCH] {input_path}")
    try:
        raw_scene = load_scene_properties(input_path, decode_auxiliary_properties=True)
    except Exception as exc:
        print(f"  load failed: {exc}")
        traceback.print_exc()
        return None
    t_load = time.time() - t_start

    t0 = time.time()
    save_encoded_scene(raw_scene, encoded_path)
    t_encode = time.time() - t0

    t0 = time.time()
    decoded_scene = load_scene_properties(encoded_path)
    t_decode = time.time() - t0

    report = collect_size_report(raw_scene, input_path, encoded_path)
    report["seconds_load_raw"] = round(t_load, 3)
    report["seconds_encode"] = round(t_encode, 3)
    report["seconds_decode_encoded"] = round(t_decode, 3)

    if render:
        from render_pipeline import render_scene_video

        frames_dir = os.path.join(output_dir, f"{name}_frames")
        video_path = os.path.join(output_dir, f"{name}.mp4")
        t0 = time.time()
        render_scene_video(
            scene_properties=decoded_scene,
            output_dir=frames_dir,
            output_mp4_path=video_path,
            num_frames=num_frames,
            framerate=15,
            width=256,
            height=256,
            orbit_radius=5.0,
            orbit_speed=0.5,
            time_step=1.0 / 30.0,
        )
        report["seconds_render"] = round(time.time() - t0, 3)
        report["rendered_video"] = video_path

    print(format_size_report_text(report))
    return report


def _human_bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024.0


def _svg_grouped_bars(reports, width=900, bar_height=22, group_gap=18, left_pad=260, right_pad=140):
    """
    Render a grouped horizontal bar chart as inline SVG.
    Three bars per scene: PLY equivalent (baseline), input file, encoded output.
    """
    series = [
        ("PLY equivalent (SH deg 3)", "ply_equivalent_bytes", "#94a3b8"),
        ("Input .npz on disk",        "input_file_bytes",     "#60a5fa"),
        ("Encoded output",            "encoded_file_bytes",   "#22c55e"),
    ]
    max_value = max(r[k] for r in reports for _, k, _ in series) or 1
    plot_width = width - left_pad - right_pad
    group_height = bar_height * len(series) + group_gap
    total_height = group_height * len(reports) + 60

    out = [f'<svg viewBox="0 0 {width} {total_height}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif, system-ui">']
    for i, report in enumerate(reports):
        y0 = 20 + i * group_height
        label = html.escape(os.path.basename(report["input_path"]))
        out.append(f'<text x="{left_pad - 12}" y="{y0 + bar_height * len(series) / 2}" text-anchor="end" font-size="13" fill="#0f172a" dominant-baseline="middle"><tspan font-weight="600">{label}</tspan>  <tspan fill="#64748b">({report["num_gaussians"]:,} pts)</tspan></text>')
        for j, (title, key, color) in enumerate(series):
            value = report[key]
            bar_w = max(1.0, plot_width * (value / max_value))
            y = y0 + j * bar_height
            out.append(f'<rect x="{left_pad}" y="{y + 2}" width="{bar_w:.1f}" height="{bar_height - 6}" fill="{color}" rx="3"/>')
            out.append(f'<text x="{left_pad + bar_w + 6}" y="{y + bar_height / 2}" font-size="11" fill="#0f172a" dominant-baseline="middle">{_human_bytes(value)}</text>')

    legend_y = total_height - 24
    legend_x = left_pad
    for title, _, color in series:
        out.append(f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" fill="{color}" rx="2"/>')
        out.append(f'<text x="{legend_x + 20}" y="{legend_y + 11}" font-size="12" fill="#334155">{html.escape(title)}</text>')
        legend_x += 240
    out.append('</svg>')
    return "\n".join(out)


def _svg_ratio_bars(reports, width=900, bar_height=26, gap=10, left_pad=260, right_pad=80):
    """
    Render a single bar per scene showing compression ratio vs PLY equivalent.
    """
    max_ratio = max(r["ratio_vs_ply_equivalent"] for r in reports) or 1.0
    plot_width = width - left_pad - right_pad
    total_height = (bar_height + gap) * len(reports) + 30
    out = [f'<svg viewBox="0 0 {width} {total_height}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif, system-ui">']
    for i, r in enumerate(reports):
        y = 12 + i * (bar_height + gap)
        ratio = r["ratio_vs_ply_equivalent"]
        bar_w = max(1.0, plot_width * (ratio / max_ratio))
        label = html.escape(os.path.basename(r["input_path"]))
        out.append(f'<text x="{left_pad - 12}" y="{y + bar_height / 2}" text-anchor="end" font-size="13" fill="#0f172a" dominant-baseline="middle">{label}</text>')
        out.append(f'<rect x="{left_pad}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" fill="#6366f1" rx="3"/>')
        out.append(f'<text x="{left_pad + bar_w + 6}" y="{y + bar_height / 2}" font-size="12" fill="#0f172a" dominant-baseline="middle" font-weight="600">{ratio:.2f}×</text>')
    out.append('</svg>')
    return "\n".join(out)


def _summary_table_rows(reports):
    rows = []
    for r in reports:
        rows.append(
            "<tr>"
            f"<td>{html.escape(os.path.basename(r['input_path']))}</td>"
            f"<td class='num'>{r['num_gaussians']:,}</td>"
            f"<td class='num'>{_human_bytes(r['ply_equivalent_bytes'])}</td>"
            f"<td class='num'>{_human_bytes(r['input_file_bytes'])}</td>"
            f"<td class='num'>{_human_bytes(r['encoded_file_bytes'])}</td>"
            f"<td class='num'><b>{r['ratio_vs_ply_equivalent']:.2f}×</b></td>"
            f"<td class='num'>{r['ratio_vs_input_file']:.2f}×</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html_report(reports, output_path):
    if not reports:
        return
    chart_sizes = _svg_grouped_bars(reports)
    chart_ratios = _svg_ratio_bars(reports)
    table_rows = _summary_table_rows(reports)
    total_files = len(reports)
    total_points = sum(r["num_gaussians"] for r in reports)
    total_input = sum(r["input_file_bytes"] for r in reports)
    total_encoded = sum(r["encoded_file_bytes"] for r in reports)
    total_ply = sum(r["ply_equivalent_bytes"] for r in reports)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gaussian Encoder/Decoder Benchmark</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 32px; color: #0f172a; background: #f8fafc; }}
  h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
  h2 {{ font-size: 16px; margin: 32px 0 12px; color: #334155; }}
  .sub {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; min-width: 180px; }}
  .card .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
  .card .value {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
  .panel {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
  th {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; background: #f1f5f9; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  svg {{ width: 100%; height: auto; }}
  footer {{ margin-top: 32px; font-size: 12px; color: #94a3b8; }}
</style>
</head>
<body>
  <h1>Gaussian Encoder / Decoder — Benchmark Report</h1>
  <div class="sub">Quantization-based encoder, decoder, and renderer pipeline applied to raw <code>.npz</code> checkpoints.</div>

  <div class="stats">
    <div class="card"><div class="label">Files benchmarked</div><div class="value">{total_files}</div></div>
    <div class="card"><div class="label">Total Gaussians</div><div class="value">{total_points:,}</div></div>
    <div class="card"><div class="label">PLY equiv. (paper)</div><div class="value">{_human_bytes(total_ply)}</div></div>
    <div class="card"><div class="label">Input on disk (total)</div><div class="value">{_human_bytes(total_input)}</div></div>
    <div class="card"><div class="label">Encoded on disk (total)</div><div class="value">{_human_bytes(total_encoded)}</div></div>
    <div class="card"><div class="label">Overall vs PLY</div><div class="value">{(total_ply / total_encoded if total_encoded else 0):.2f}×</div></div>
  </div>

  <h2>Sizes per scene</h2>
  <div class="panel">{chart_sizes}</div>

  <h2>Compression ratio vs PLY-equivalent baseline</h2>
  <div class="panel">{chart_ratios}</div>

  <h2>Per-file summary</h2>
  <div class="panel">
    <table>
      <thead><tr>
        <th>File</th><th class="num">Gaussians</th><th class="num">PLY equiv</th>
        <th class="num">Input .npz</th><th class="num">Encoded</th>
        <th class="num">Ratio vs PLY</th><th class="num">Ratio vs input</th>
      </tr></thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

  <footer>Generated by <code>benchmark.py</code>. Self-contained — no internet required.</footer>
</body>
</html>"""
    with open(output_path, "w") as fh:
        fh.write(html_doc)


def resolve_inputs(args):
    files = list(args.inputs)
    if args.input_dir:
        files.extend(sorted(glob.glob(os.path.join(args.input_dir, "*.npz"))))
    if not files:
        raise SystemExit(
            "No input files. Pass paths as positional args or use --input-dir."
        )
    files = [f for f in files if os.path.exists(f)]
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Explicit .npz files to benchmark.")
    parser.add_argument("--input-dir", help="Directory to scan for *.npz inputs.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to write encoded files and the summary CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path to write the combined results CSV (defaults to <output-dir>/benchmark_summary.csv).",
    )
    parser.add_argument("--render", action="store_true", help="Also render a short video per scene.")
    parser.add_argument("--num-frames", type=int, default=10, help="Frames per render when --render is set.")
    args = parser.parse_args()

    files = resolve_inputs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(args.output_dir, "benchmark_summary.csv")

    reports = []
    for path in files:
        report = benchmark_one(path, args.output_dir, render=args.render, num_frames=args.num_frames)
        if report is not None:
            reports.append(report)

    if not reports:
        raise SystemExit("No successful benchmarks. Nothing written.")

    fieldnames = list(reports[0].keys())
    for report in reports[1:]:
        for k in report:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(report)

    html_path = os.path.join(args.output_dir, "report.html")
    write_html_report(reports, html_path)

    print()
    print("=" * 70)
    print(f" CSV summary  : {summary_csv}")
    print(f" HTML report  : {html_path}    <-- open this in a browser")
    print(f" Files benchmarked            : {len(reports)} / {len(files)}")
    print("=" * 70)
    print()
    print(" file                          gaussians   ply_eq_kb   encoded_kb   ratio")
    for r in reports:
        print(
            f" {os.path.basename(r['input_path']):28s}  "
            f"{r['num_gaussians']:>9,}  "
            f"{r['ply_equivalent_bytes'] / 1024:>9.1f}  "
            f"{r['encoded_file_bytes'] / 1024:>10.1f}  "
            f"{r['ratio_vs_ply_equivalent']:>6.2f}x"
        )


if __name__ == "__main__":
    main()
