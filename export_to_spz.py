import os
import struct
import zlib
import numpy as np

from scene_io import load_scene_properties_from_npz, load_scene_properties_from_4dgs
from scene import Scene

def export_spz(path, scene_props, time_t=0.0):
    scene = Scene(scene_props)
    
    pts = scene.get_points_at_time(time_t)
    colors = scene.get_color_at_time(time_t)
    opacity = scene.get_opacity_at_time(time_t)
    scale = scene.get_scale_at_time(time_t)
    rotation = scene.get_rotation_at_time(time_t)

    if pts is None or len(pts) == 0:
        print("No points to export.")
        return

    numPoints = len(pts)
    fractionalBits = 16
    magic = 0x5053474e
    version = 2
    shDegree = 0
    flags = 0
    reserved = 0
    
    header = struct.pack('<IIIBBBB', magic, version, numPoints, shDegree, fractionalBits, flags, reserved)
    
    scale_factor = 1.0 / (1 << fractionalBits)
    
    # Pack positions (3x 24-bit little endian)
    positions_raw = np.clip(np.round(pts / scale_factor), -8388608, 8388607).astype(np.int32)
    pos_bytes = positions_raw.view(np.uint8).reshape(numPoints, 3, 4)[:, :, :3].tobytes()
    
    # Pack alphas
    if opacity is None:
        opacity = np.ones((numPoints, 1), dtype=np.float32)
    alphas_bytes = np.clip(np.round(opacity * 255.0), 0, 255).astype(np.uint8).tobytes()
    
    # Pack colors (RGB)
    if colors is None:
        colors = np.ones((numPoints, 3), dtype=np.float32)
    colors_bytes = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8).tobytes()
    
    # Pack scales
    if scale is None:
        scale = np.ones((numPoints, 3), dtype=np.float32) * 0.01
    scale_clipped = np.clip(scale, 1e-6, None)
    log_scale = np.log(scale_clipped)
    scales_bytes = np.clip(np.round((log_scale + 10.0) * 16.0), 0, 255).astype(np.uint8).tobytes()
    
    # Pack rotations
    if rotation is None:
        rotation = np.zeros((numPoints, 4), dtype=np.float32)
        rotation[:, 0] = 1.0
    rot = rotation.copy()
    w_mask = rot[:, 0] < 0
    rot[w_mask] *= -1.0
    
    rot_xyz = rot[:, 1:4]
    rotations_bytes = np.clip(np.round((rot_xyz + 1.0) * 127.5), 0, 255).astype(np.uint8).tobytes()
    
    payload = header + pos_bytes + alphas_bytes + colors_bytes + scales_bytes + rotations_bytes
    
    compressed = zlib.compress(payload)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(compressed)
    print(f"Exported {numPoints} points to {path}")

if __name__ == "__main__":
    npz_path = "point_cloud_pp.npz"
    fourdgs_path = "ours_cook_spinach.4dgs"
    
    if os.path.exists(fourdgs_path):
        props_4dgs = load_scene_properties_from_4dgs(fourdgs_path)
        export_spz("flamsplat/mac/out/spz/ours_reference.spz", props_4dgs)
        
    if os.path.exists(npz_path):
        props_npz = load_scene_properties_from_npz(npz_path, decode_auxiliary_properties=True)
        export_spz("flamsplat/mac/out/spz/ours_decoded.spz", props_npz)
