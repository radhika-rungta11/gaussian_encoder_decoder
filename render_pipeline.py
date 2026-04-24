import os
import time

import numpy as np
from PIL import Image

from camera import Camera
from export_video import export_frames_to_mp4
from renderer import render_scene_cpu
from scene import Scene


def compute_scene_center(scene, sample_time=0.0):
    pts = scene.get_points_at_time(sample_time)
    if pts is None or len(pts) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.mean(pts, axis=0).astype(np.float32)


def build_orbit_camera(center, width=512, height=512):
    cam = Camera(width=width, height=height)
    cam_pos = center + np.array([0.0, 0.0, 5.0], dtype=np.float32)
    cam.set_pose(position=cam_pos, target=center, up=np.array([0, 1, 0], dtype=np.float32))
    return cam


def render_scene_video(
    scene_properties,
    output_dir,
    output_mp4_path,
    num_frames=30,
    framerate=15,
    width=512,
    height=512,
    orbit_radius=5.0,
    orbit_speed=0.5,
    time_step=0.1,
):
    scene = Scene(scene_properties)

    if scene.num_points == 0:
        raise ValueError("No points found in the scene.")

    os.makedirs(output_dir, exist_ok=True)

    center = compute_scene_center(scene)
    cam = build_orbit_camera(center=center, width=width, height=height)

    print(f"Rendering {num_frames} frames to {output_dir}/ ...")

    for frame in range(num_frames):
        t0 = time.time()
        t = frame * time_step

        pts_t = scene.get_points_at_time(t)
        colors_t = scene.get_color_at_time(t)
        opacity_t = scene.get_opacity_at_time(t)
        scale_t = scene.get_scale_at_time(t)
        rotation_t = scene.get_rotation_at_time(t)

        angle = t * orbit_speed
        orbit_pos = center + np.array(
            [np.sin(angle) * orbit_radius, 0.0, np.cos(angle) * orbit_radius],
            dtype=np.float32,
        )
        cam.set_pose(position=orbit_pos, target=center)

        image_np = render_scene_cpu(
            pts_t,
            colors_t,
            cam,
            opacity=opacity_t,
            scale=scale_t,
            rotation=rotation_t,
            background_color=(0.02, 0.02, 0.03),
        )

        out_path = os.path.join(output_dir, f"frame_{frame:04d}.png")
        Image.fromarray(image_np).save(out_path)

        elapsed = time.time() - t0
        print(f"  Rendered frame {frame:04d} in {elapsed:.3f}s")

    export_frames_to_mp4(output_dir, output_mp4_path, framerate=framerate)
    print(f"Finished video: {output_mp4_path}")
