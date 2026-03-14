from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Transform2D:
    mat: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        self.mat = np.asarray(self.mat, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)

    def apply_point(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float32)
        return (pts @ self.mat.T) + self.bias

    def apply_vector(self, xy: np.ndarray) -> np.ndarray:
        vec = np.asarray(xy, dtype=np.float32)
        return vec @ self.mat.T

    def inverse(self) -> 'Transform2D':
        inv_mat = np.linalg.inv(self.mat).astype(np.float32)
        inv_bias = (-self.bias @ inv_mat.T).astype(np.float32)
        return Transform2D(mat=inv_mat, bias=inv_bias)

    def inverse_apply_point(self, xy: np.ndarray) -> np.ndarray:
        return self.inverse().apply_point(xy)

    def inverse_apply_vector(self, xy: np.ndarray) -> np.ndarray:
        return self.inverse().apply_vector(xy)


@dataclass
class NeighborIndices:
    front: Optional[int]
    rear: Optional[int]


@dataclass
class RawAirSimState:
    vehicle_name: str
    timestamp: float
    pose_world_xy: np.ndarray
    yaw_world: float
    z_world: float
    vel_world_xy: np.ndarray
    acc_world_xy: np.ndarray
    yaw_rate: float
    imu_acc_body: np.ndarray
    imu_gyro_body: np.ndarray
    gps_lat_lon_alt: Optional[np.ndarray] = None


@dataclass
class CanonicalVehicleState:
    index: int
    vehicle_name: str
    pose_map_xy: np.ndarray
    yaw_map: float
    vel_map_xy: np.ndarray
    acc_map_xy: np.ndarray
    speed: float
    steering_feedback: float = 0.0
    last_action_acc: float = 0.0
    last_action_steer: float = 0.0


@dataclass
class RoadProjection:
    s: float
    projection_point_map: np.ndarray
    projection_point_mode: str
    closest_center_xy: np.ndarray
    tangent_yaw: float
    ref_v: float
    short_term_ref: np.ndarray
    left_boundary_pts: np.ndarray
    right_boundary_pts: np.ndarray
    dist_to_ref: float
    dist_to_left_boundary: float
    dist_to_right_boundary: float
    vertices_xy: np.ndarray


@dataclass
class ActorAction:
    acceleration_mps2: float
    front_wheel_angle_rad: float


@dataclass
class LowLevelCommand:
    throttle: float
    brake: float
    steering: float


@dataclass
class SceneFrame:
    states: List[CanonicalVehicleState]
    projections: List[RoadProjection]
    timestamp: float


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    reward: Dict[str, float]
    terminated: Dict[str, bool]
    truncated: Dict[str, bool]
    info: Dict
