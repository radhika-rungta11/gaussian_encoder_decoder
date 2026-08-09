import numpy as np


class Scene:
    """
    Dynamic 4D Gaussian scene.
    Time is normalized to [0, 1]. Each Gaussian has:
      - xyz0           static position
      - motion[N,9]    polynomial coefficients: vel(0:3), acc(3:6), jerk(6:9)
      - tcen           temporal center in [0, 1]
      - tsca           temporal precision (per-Gaussian temporal sharpness)
      - opacity        baseline opacity in [0, 1]

    Position at time t:
        xyz(t) = xyz0 + v*t + a*t^2 + j*t^3

    Temporal opacity gating (Dynamic_C3DGS style):
        a(t) = opacity * exp(-0.5 * ((t - tcen) * tsca)^2)
    """

    def __init__(self, scene_properties):
        self.base_xyz = scene_properties.xyz
        self.motion = scene_properties.motion
        self.opacity = scene_properties.opacity
        self.scale = scene_properties.scale
        self.rotation = scene_properties.rotation
        self.rgb = scene_properties.rgb
        self.tcen = getattr(scene_properties, "tcen", None)
        self.tsca = getattr(scene_properties, "tsca", None)
        self.num_points = self.base_xyz.shape[0] if self.base_xyz is not None else 0

    def get_points_at_time(self, t):
        if self.base_xyz is None:
            return None
        if self.motion is None:
            return self.base_xyz.copy()
        if self.motion.shape[1] >= 9:
            v = self.motion[:, 0:3]
            a = self.motion[:, 3:6]
            j = self.motion[:, 6:9]
            return self.base_xyz + v * t + a * (t * t) + j * (t * t * t)
        return self.base_xyz + self.motion[:, :3] * t

    def get_color_at_time(self, t, camera_dir=None):
        return self.rgb

    def get_opacity_at_time(self, t):
        if self.opacity is None:
            return None
        op = np.asarray(self.opacity, dtype=np.float32).reshape(-1, 1)
        if self.tcen is None or self.tsca is None:
            return op
        # Temporal Gaussian gate: peaks at tcen with sharpness tsca.
        dt = (t - self.tcen).astype(np.float32)
        gate = np.exp(-0.5 * (dt * self.tsca.astype(np.float32)) ** 2).reshape(-1, 1)
        return np.clip(op * gate, 0.0, 1.0)

    def get_scale_at_time(self, t):
        return self.scale

    def get_rotation_at_time(self, t):
        return self.rotation
