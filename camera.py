import numpy as np

class Camera:
    """
    Extremely simple Pinhole Camera Model.
    """
    def __init__(self, width=512, height=512, focal_xy=None):
        self.width = width
        self.height = height
        
        # Simple orthographic/pinhole default setup
        if focal_xy is None:
            # FoV approx 90 degrees
            focal = width / 2.0 
            self.fx = focal
            self.fy = focal
        else:
            self.fx, self.fy = focal_xy
            
        self.cx = width / 2.0
        self.cy = height / 2.0

        # View matrix (World to Camera)
        # Default: looking down -Z axis
        self.extrinsic = np.eye(4, dtype=np.float32)
        
    def set_pose(self, position, target, up=np.array([0, 1, 0])):
        """
        Set camera extrinsic matrix using LookAt method.
        """
        position = np.array(position, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        # Handle colinear up and forward vectors
        if np.linalg.norm(right) < 1e-6:
             right = np.array([1, 0, 0], dtype=np.float32)
        else:
            right = right / np.linalg.norm(right)
            
        new_up = np.cross(right, forward)
        new_up = new_up / np.linalg.norm(new_up)
        
        # We assume standard OpenGL convention (Camera looks down -Z)
        R = np.array([
            [right[0], right[1], right[2]],
            [new_up[0], new_up[1], new_up[2]],
            [-forward[0], -forward[1], -forward[2]]
        ])
        
        t = -R @ position
        
        self.extrinsic = np.eye(4, dtype=np.float32)
        self.extrinsic[:3, :3] = R
        self.extrinsic[:3, 3] = t

    def project_points(self, points_3d):
        """
        Projects (N, 3) points into 2D pixel coordinates (N, 2)
        and returns depths (N,).
        """
        points_homo = np.pad(points_3d, ((0,0), (0,1)), constant_values=1.0)
        
        # To Camera Space
        points_cam = (self.extrinsic @ points_homo.T).T
        
        # Points behind camera (Z > 0 in standard GL, but let's assume we mapped Z to depth)
        # Our LookAt maps looking down -Z, so depth is -Z
        depths = -points_cam[:, 2]
        
        # Avoid division by zero
        valid_depth = np.maximum(depths, 1e-5)
        
        x_proj = (points_cam[:, 0] / valid_depth) * self.fx + self.cx
        y_proj = (points_cam[:, 1] / valid_depth) * self.fy + self.cy
        
        coords_2d = np.stack((x_proj, y_proj), axis=1)
        return coords_2d, depths
