import numpy as np

class Scene:
    """
    In-memory representation of a dynamic point cloud / Gaussian scene.
    Provides methods to step forward in time.
    """
    def __init__(self, scene_properties):
        self.base_xyz = scene_properties.xyz
        self.motion = scene_properties.motion  # Shape could be (N, 3) or (N, 9) representing polynomial coeffs
        
        self.opacity = scene_properties.opacity
        self.scale = scene_properties.scale
        self.rotation = scene_properties.rotation
        self.rgb = scene_properties.rgb
        
        self.num_points = self.base_xyz.shape[0] if self.base_xyz is not None else 0
        
    def get_points_at_time(self, t):
        """
        Computes the positions of the points at timestep t.
        We assume a simple linear motion or polynomial based on the 'motion' shape.
        """
        if self.base_xyz is None:
            return None
            
        if self.motion is None:
            return self.base_xyz.copy()
            
        # Treat 9D motion as a per-axis quadratic polynomial when available:
        # xyz_t = xyz_0 + v * t + a * t^2 + j * t^3
        if self.motion.shape[1] >= 9:
            velocity = self.motion[:, 0:3]
            acceleration = self.motion[:, 3:6]
            jerk = self.motion[:, 6:9]
            xyz_t = self.base_xyz + velocity * t + acceleration * (t * t) + jerk * (t * t * t)
        else:
            velocity = self.motion[:, :3]
            xyz_t = self.base_xyz + velocity * t
        
        return xyz_t

    def get_color_at_time(self, t, camera_dir=None):
        """
        If SH (Spherical Harmonics) features were decoded, they would depend on view direction.
        For a simple prototype we return the base RGB.
        """
        return self.rgb

    def get_opacity_at_time(self, t):
        return self.opacity

    def get_scale_at_time(self, t):
        return self.scale

    def get_rotation_at_time(self, t):
        return self.rotation
