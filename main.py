import argparse
import os

from encoder import print_size_report, save_encoded_scene
from render_pipeline import render_scene_video
from scene_io import load_scene_properties


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end 4D Gaussian encoder / decoder / renderer demo. "
            "Loads the raw .npz checkpoint, compresses it, reloads the "
            "compressed file (round-trip), renders the reloaded scene, "
            "and prints a size report."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="point_cloud_pp.npz",
        help="Raw scene input. Defaults to point_cloud_pp.npz.",
    )
    parser.add_argument(
        "--encoded-output",
        default="encoded_scene.npz",
        help="Path to write the compressed scene to.",
    )
    parser.add_argument(
        "--output-dir",
        default="rendered_frames_decoded",
        help="Directory for rendered PNG frames.",
    )
    parser.add_argument(
        "--output-video",
        default="output_decoded_scene.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write compression_report.txt + .csv files for graphing.",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="Skip the encode step and render the input directly (for quick visual checks).",
    )
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--framerate", type=int, default=15)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--orbit-radius", type=float, default=5.0)
    parser.add_argument("--orbit-speed", type=float, default=0.5)
    parser.add_argument("--time-step", type=float, default=1.0 / 30.0)
    return parser


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Could not find input scene file: {args.input_path}")

    print(f"[1/4] Loading raw input scene: {args.input_path}")
    raw_scene = load_scene_properties(args.input_path, decode_auxiliary_properties=True)

    if args.skip_encode:
        scene_to_render = raw_scene
        encoded_path = None
    else:
        print(f"[2/4] Encoding to: {args.encoded_output}")
        encoded_path = save_encoded_scene(raw_scene, args.encoded_output)

        print(f"[3/4] Reloading encoded scene from: {encoded_path}")
        scene_to_render = load_scene_properties(encoded_path)

    print(f"[4/4] Rendering {args.num_frames} frames → {args.output_video}")
    render_scene_video(
        scene_properties=scene_to_render,
        output_dir=args.output_dir,
        output_mp4_path=args.output_video,
        num_frames=args.num_frames,
        framerate=args.framerate,
        width=args.width,
        height=args.height,
        orbit_radius=args.orbit_radius,
        orbit_speed=args.orbit_speed,
        time_step=args.time_step,
    )

    if encoded_path is not None:
        print_size_report(raw_scene, args.input_path, encoded_path, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
