import argparse
import os

from encoder import save_encoded_scene
from render_pipeline import render_scene_video
from scene_io import load_scene_properties


def build_parser():
    parser = argparse.ArgumentParser(
        description="Decode, render, and optionally re-encode 4D Gaussian scene assets."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="ours_cook_spinach.4dgs",
        help="Input scene file (.npz or .4dgs). Defaults to ours_cook_spinach.4dgs.",
    )
    parser.add_argument(
        "--output-dir",
        default="rendered_frames_4dgs",
        help="Directory for rendered frames.",
    )
    parser.add_argument(
        "--output-video",
        default="output_4dgs_scene.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--encoded-output",
        help="Optional output path for a re-encoded scene archive.",
    )
    parser.add_argument(
        "--decode-aux",
        action="store_true",
        help="Decode auxiliary Gaussian properties from the original compressed NPZ.",
    )
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--framerate", type=int, default=15)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--orbit-radius", type=float, default=5.0)
    parser.add_argument("--orbit-speed", type=float, default=0.5)
    parser.add_argument("--time-step", type=float, default=0.1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Could not find input scene file: {args.input_path}")

    scene = load_scene_properties(args.input_path, decode_auxiliary_properties=args.decode_aux)

    render_scene_video(
        scene_properties=scene,
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

    if args.encoded_output:
        save_encoded_scene(scene, args.encoded_output)
        print(f"Saved re-encoded scene to {args.encoded_output}")


if __name__ == "__main__":
    main()
