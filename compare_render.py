"""
Side-by-side comparison renderer with file-size overlay.

LEFT  panel: ORIGINAL .4dgs file (input to the C encoder)
RIGHT panel: C-encoder-output .4dgs file

Each frame shows:
  - raw float32 size baseline (what the same data would cost with no codec)
  - compressed file size
  - compression ratio vs raw
  - PSNR(original vs re-encoded) per frame

The two cameras share an orbit so the panels are pixel-aligned.

Usage:
  python compare_render.py \\
      --reference  ours_cook_spinach.4dgs \\
      --compressed c_pipeline/roundtrip.4dgs \\
      --output     compare.mp4
"""

import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera import Camera
from export_video import export_frames_to_mp4
from renderer import render_scene_cpu
from scene import Scene
from scene_io import load_scene_properties

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_size(num_bytes):
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.1f} KB"
    return f"{num_bytes} B"


def get_font(size):
    for p in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def annotate(img_np, header_lines, footer_lines, color=(255, 255, 255)):
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img, "RGBA")
    f_title = get_font(20)
    f_body  = get_font(15)

    # Top translucent bar
    top_h = 8 + 24 + 20 * (len(header_lines) - 1)
    draw.rectangle((0, 0, img.width, top_h), fill=(0, 0, 0, 170))
    y = 4
    for i, line in enumerate(header_lines):
        f = f_title if i == 0 else f_body
        draw.text((12, y), line, fill=color, font=f)
        y += 24 if i == 0 else 19

    # Bottom translucent bar
    bot_h = 8 + 20 * len(footer_lines)
    draw.rectangle((0, img.height - bot_h, img.width, img.height),
                   fill=(0, 0, 0, 170))
    y = img.height - bot_h + 4
    for line in footer_lines:
        draw.text((12, y), line, fill=color, font=f_body)
        y += 19
    return np.array(img)


def render_one(scene_props, camera, t):
    s = Scene(scene_props)
    pts      = s.get_points_at_time(t)
    colors   = s.get_color_at_time(t)
    # Use static opacity for the demo so both panels render the same Gaussians
    opacity  = scene_props.opacity
    scale    = s.get_scale_at_time(t)
    rotation = s.get_rotation_at_time(t)
    return render_scene_cpu(
        pts, colors, camera,
        opacity=opacity, scale=scale, rotation=rotation,
        background_color=(0.02, 0.02, 0.03),
    )


def compute_psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-9:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def raw_float32_size(num_points, has_features=True, full_sh=True):
    """Paper-style 'uncompressed' baseline — what the same scene would
    occupy as a plain float32 PLY (3DGS + dynamic extension).

    Per-Gaussian floats:
      xyz(3) + opacity(1) + scale(3) + rotation(4)               = 11
      features_dc(3)                                              + 3
      [optional] features_rest for SH degrees 1..3 (15 bands * 3) +45
      motion(9) + tcen(1) + tsca(1) + omega(4) + tfea(3)          +18
    """
    floats = 11 + 3 + 18
    if full_sh:
        floats += 45
    total = num_points * floats * 4
    total += (6 * 12 + 3 * 6) * 4   # rgb_dec MLP weights (negligible)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference",   default="ours_cook_spinach.4dgs")
    ap.add_argument("--compressed",  default="c_pipeline/roundtrip.4dgs.gz")
    ap.add_argument("--output",      default="compare_demo.mp4")
    ap.add_argument("--frames-dir",  default="compare_frames_demo")
    ap.add_argument("--num-frames",  type=int, default=30)
    ap.add_argument("--framerate",   type=int, default=15)
    ap.add_argument("--width",       type=int, default=512)
    ap.add_argument("--height",      type=int, default=512)
    ap.add_argument("--orbit-speed", type=float, default=0.5,
                    help="fraction of full circle over the whole video")
    ap.add_argument("--no-orbit", action="store_true",
                    help="freeze camera direction; only animate t")
    args = ap.parse_args()

    if not os.path.exists(args.reference):
        raise SystemExit(f"reference not found: {args.reference}")
    if not os.path.exists(args.compressed):
        raise SystemExit(f"compressed not found: {args.compressed}")

    ref_size = os.path.getsize(args.reference)
    cmp_size = os.path.getsize(args.compressed)

    print("Loading reference scene...")
    ref = load_scene_properties(args.reference)
    print("Loading compressed scene...")
    cmp = load_scene_properties(args.compressed)
    if ref.xyz.shape[0] != cmp.xyz.shape[0]:
        raise SystemExit(
            f"point-count mismatch: ref={ref.xyz.shape[0]} cmp={cmp.xyz.shape[0]}"
        )
    n_pts = ref.xyz.shape[0]

    raw_sz = raw_float32_size(n_pts, has_features=ref.features_dc is not None)
    ratio_raw_to_ref = raw_sz / ref_size
    ratio_raw_to_cmp = raw_sz / cmp_size

    print(f"  Gaussians:                {n_pts:,}")
    print(f"  Hypothetical raw f32:     {fmt_size(raw_sz)}  (no codec)")
    print(f"  Reference .4dgs:          {fmt_size(ref_size)}  ({ratio_raw_to_ref:.2f}x vs raw)")
    print(f"  C-encoder output:         {fmt_size(cmp_size)}  ({ratio_raw_to_cmp:.2f}x vs raw)")

    # ---- camera -----------------------------------------------------------
    bbox_min = ref.xyz.min(axis=0)
    bbox_max = ref.xyz.max(axis=0)
    center = ((bbox_min + bbox_max) * 0.5).astype(np.float32)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    radius = diag * 1.0
    print(f"  Scene bbox center:       {center}")
    print(f"  Scene diag:              {diag:.2f}  (camera radius {radius:.2f})")

    cam_a = Camera(width=args.width, height=args.height)
    cam_b = Camera(width=args.width, height=args.height)

    os.makedirs(args.frames_dir, exist_ok=True)
    out_w = args.width * 2 + 8
    out_h = args.height + 38   # extra strip for global title

    psnr_history = []
    for f in range(args.num_frames):
        t0 = time.time()
        # t walks [0, 1) over the whole video so motion polynomial stays sane
        t = f / args.num_frames

        if args.no_orbit:
            angle = 0.0
        else:
            angle = t * args.orbit_speed * 2.0 * np.pi
        # orbit around y axis
        cam_pos = center + np.array([
            np.sin(angle) * radius, 0.0, np.cos(angle) * radius,
        ], dtype=np.float32)
        cam_a.set_pose(position=cam_pos, target=center)
        cam_b.set_pose(position=cam_pos, target=center)

        img_ref = render_one(ref, cam_a, t)
        img_cmp = render_one(cmp, cam_b, t)
        psnr = compute_psnr(img_ref, img_cmp)
        psnr_history.append(psnr)

        img_ref_a = annotate(
            img_ref,
            header_lines=[
                "REFERENCE  (input .4dgs)",
                f"file: {os.path.basename(args.reference)}",
            ],
            footer_lines=[
                f"raw float32 baseline:  {fmt_size(raw_sz)}",
                f"compressed size:       {fmt_size(ref_size)}    "
                f"=>  {ratio_raw_to_ref:.2f}x compression",
                f"N = {n_pts:,}     t = {t:.3f}",
            ],
        )
        img_cmp_a = annotate(
            img_cmp,
            header_lines=[
                "C-ENCODER OUTPUT  (decode -> encode -> decode)",
                f"file: {os.path.basename(args.compressed)}",
            ],
            footer_lines=[
                f"raw float32 baseline:  {fmt_size(raw_sz)}",
                f"compressed size:       {fmt_size(cmp_size)}    "
                f"=>  {ratio_raw_to_cmp:.2f}x compression",
                f"PSNR vs reference:  {psnr:.2f} dB",
            ],
        )

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # global header strip
        title_img = Image.fromarray(canvas[:38])
        td = ImageDraw.Draw(title_img)
        td.rectangle((0, 0, out_w, 38), fill=(20, 20, 24))
        title_text = (
            f"4D Gaussian Splat Codec  |  "
            f"raw {fmt_size(raw_sz)}  ->  ref {fmt_size(ref_size)}  "
            f"({ratio_raw_to_ref:.2f}x)  ->  re-enc {fmt_size(cmp_size)}  "
            f"({ratio_raw_to_cmp:.2f}x)   |   PSNR {psnr:.1f} dB"
        )
        td.text((12, 9), title_text, fill=(230, 230, 230), font=get_font(16))
        canvas[:38] = np.array(title_img)

        canvas[38:, :args.width] = img_ref_a
        canvas[38:, args.width + 8:] = img_cmp_a
        canvas[38:, args.width:args.width + 8] = (40, 40, 44)

        out_path = os.path.join(args.frames_dir, f"frame_{f:04d}.png")
        Image.fromarray(canvas).save(out_path)
        elapsed = time.time() - t0
        print(f"  frame {f:04d}/{args.num_frames}  PSNR={psnr:.2f}  ({elapsed:.2f}s)")

    print(f"mean PSNR = {np.mean(psnr_history):.2f} dB")
    export_frames_to_mp4(args.frames_dir, args.output, framerate=args.framerate)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
