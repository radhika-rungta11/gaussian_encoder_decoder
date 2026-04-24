import numpy as np

def _infer_fallback_colors(scene_points):
    mins = np.min(scene_points, axis=0, keepdims=True)
    maxs = np.max(scene_points, axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-5)
    normalized = (scene_points - mins) / span
    return np.clip(0.25 + normalized * 0.75, 0.0, 1.0)

def build_rotation_matrix(q):
    # q is [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*y*y - 2*z*z
    R[:, 0, 1] = 2*x*y - 2*w*z
    R[:, 0, 2] = 2*x*z + 2*w*y
    R[:, 1, 0] = 2*x*y + 2*w*z
    R[:, 1, 1] = 1 - 2*x*x - 2*z*z
    R[:, 1, 2] = 2*y*z - 2*w*x
    R[:, 2, 0] = 2*x*z - 2*w*y
    R[:, 2, 1] = 2*y*z + 2*w*x
    R[:, 2, 2] = 1 - 2*x*x - 2*y*y
    return R

def render_scene_cpu(
    scene_points,
    colors,
    camera,
    opacity=None,
    scale=None,
    rotation=None,
    background_color=(0, 0, 0),
):
    """
    A CPU Gaussian splat renderer (EWA Volume Splatting).
    Projects 3D covariance into 2D and alpha-composites.
    """
    width, height = int(camera.width), int(camera.height)
    image = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)

    image[:, :] = background_color

    if len(scene_points) == 0:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    colors = np.asarray(colors, dtype=np.float32) if colors is not None else _infer_fallback_colors(scene_points)
    
    if opacity is None:
        opacity = np.ones((len(scene_points), 1), dtype=np.float32)
    opacity = np.asarray(opacity, dtype=np.float32).reshape(-1)

    if scale is None:
        scale = np.ones((len(scene_points), 3), dtype=np.float32) * 0.01
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 1:
        scale = np.repeat(scale[:, None], 3, axis=1)

    if rotation is None:
        rotation = np.zeros((len(scene_points), 4), dtype=np.float32)
        rotation[:, 0] = 1.0
    rotation = np.asarray(rotation, dtype=np.float32)

    # Convert to homogenous to project to camera space
    points_homo = np.pad(scene_points, ((0,0), (0,1)), constant_values=1.0)
    
    # View matrix (4x4)
    W = camera.extrinsic
    points_cam = (W @ points_homo.T).T
    depths = -points_cam[:, 2]

    # Mask points behind camera or too close
    valid_depth_mask = depths > 0.2
    
    if not np.any(valid_depth_mask):
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    scene_points = scene_points[valid_depth_mask]
    colors = colors[valid_depth_mask]
    opacity = opacity[valid_depth_mask]
    scale = scale[valid_depth_mask]
    rotation = rotation[valid_depth_mask]
    points_cam = points_cam[valid_depth_mask]
    depths = depths[valid_depth_mask]

    # 2D projection coordinates
    x_proj = (points_cam[:, 0] / depths) * camera.fx + camera.cx
    y_proj = (points_cam[:, 1] / depths) * camera.fy + camera.cy
    coords_2d = np.stack((x_proj, y_proj), axis=1)

    # Screen mask
    valid_mask = (coords_2d[:, 0] >= -200) & (coords_2d[:, 0] < width + 200) & \
                 (coords_2d[:, 1] >= -200) & (coords_2d[:, 1] < height + 200)

    coords_2d = coords_2d[valid_mask]
    depths = depths[valid_mask]
    colors = colors[valid_mask]
    opacity = opacity[valid_mask]
    scale = scale[valid_mask]
    rotation = rotation[valid_mask]
    points_cam = points_cam[valid_mask]

    if len(depths) == 0:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # Sort by depth (back to front)
    sort_idx = np.argsort(depths)[::-1]
    
    coords_2d = coords_2d[sort_idx]
    depths = depths[sort_idx]
    colors = colors[sort_idx]
    opacity = opacity[sort_idx]
    scale = scale[sort_idx]
    rotation = rotation[sort_idx]
    points_cam = points_cam[sort_idx]

    # Precompute 3D covariance and project to 2D
    R = build_rotation_matrix(rotation)
    S = np.zeros((len(scale), 3, 3), dtype=np.float32)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]

    # M = R @ S
    M = R @ S
    # Cov3D = M @ M^T
    Cov3D = M @ np.transpose(M, (0, 2, 1))

    # Transform to camera space
    W_3x3 = W[:3, :3]
    CovCam = W_3x3 @ Cov3D @ W_3x3.T
    
    for i in range(len(coords_2d)):
        if opacity[i] <= 1e-5:
            continue
            
        x, y = coords_2d[i]
        t = points_cam[i]
        t_x, t_y, t_z = t[0], t[1], t[2]
        
        # Avoid division by zero
        tz = -t_z  # t_z is negative in OpenGL coords (looking down -Z)
        if tz < 1e-3: tz = 1e-3
        
        fx, fy = camera.fx, camera.fy
        
        # Jacobian of perspective projection
        J = np.array([
            [fx / tz, 0, (fx * t_x) / (tz * tz)],
            [0, fy / tz, (fy * t_y) / (tz * tz)]
        ], dtype=np.float32)
        
        cov_cam_i = CovCam[i]
        cov_2d = J @ cov_cam_i @ J.T
        
        # Apply low-pass filter to prevent aliasing
        cov_2d[0, 0] += 0.3
        cov_2d[1, 1] += 0.3
        
        # Compute inverse covariance and determinant
        a = cov_2d[0, 0]
        b = cov_2d[0, 1]
        c = cov_2d[1, 1]
        
        det = a * c - b * b
        if det < 1e-6:
            continue
            
        inv_cov = np.array([
            [c, -b],
            [-b, a]
        ], dtype=np.float32) / det
        
        # Bounding box
        radius_x = int(np.ceil(3.0 * np.sqrt(a)))
        radius_y = int(np.ceil(3.0 * np.sqrt(c)))
        
        cx = int(np.round(x))
        cy = int(np.round(y))
        x0 = max(0, cx - radius_x)
        x1 = min(width - 1, cx + radius_x)
        y0 = max(0, cy - radius_y)
        y1 = min(height - 1, cy + radius_y)
        if x0 > x1 or y0 > y1:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32) - x
        ys = np.arange(y0, y1 + 1, dtype=np.float32) - y
        
        dx, dy = np.meshgrid(xs, ys)
        
        dist2 = dx * dx * inv_cov[0, 0] + 2.0 * dx * dy * inv_cov[0, 1] + dy * dy * inv_cov[1, 1]
        
        dist2 = np.maximum(dist2, 0)
        
        gaussian = np.exp(-0.5 * dist2)
        src_alpha = np.clip(opacity[i], 0.0, 1.0) * gaussian
        
        patch_alpha = alpha[y0:y1 + 1, x0:x1 + 1]
        patch_image = image[y0:y1 + 1, x0:x1 + 1]

        contrib = src_alpha * (1.0 - patch_alpha)
        patch_image += contrib[..., None] * colors[i][None, None, :]
        patch_alpha += contrib

        image[y0:y1 + 1, x0:x1 + 1] = patch_image
        alpha[y0:y1 + 1, x0:x1 + 1] = np.clip(patch_alpha, 0.0, 1.0)

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)
