import numpy as np

from airsim_occt_schema import CanonicalVehicleState, RawAirSimState, Transform2D


class WorldToMapTransformer:
    def __init__(self, transform: Transform2D, flip_world_y: bool = False) -> None:
        self.transform = transform
        self.flip_world_y = bool(flip_world_y)

    def _maybe_flip_y(self, xy_world: np.ndarray) -> np.ndarray:
        arr = np.asarray(xy_world, dtype=np.float32).copy()
        arr[..., 1] *= -1.0 if self.flip_world_y else 1.0
        return arr

    def point_world_to_map(self, xy_world: np.ndarray) -> np.ndarray:
        return self.transform.apply_point(self._maybe_flip_y(xy_world))

    def vector_world_to_map(self, vec_world: np.ndarray) -> np.ndarray:
        return self.transform.apply_vector(self._maybe_flip_y(vec_world))

    def yaw_world_to_map(self, yaw_world: float) -> float:
        heading_world = np.asarray([np.cos(yaw_world), np.sin(yaw_world)], dtype=np.float32)
        heading_map = self.vector_world_to_map(heading_world)
        return float(np.arctan2(heading_map[1], heading_map[0]))

    def convert(self, raw: RawAirSimState, agent_index: int) -> CanonicalVehicleState:
        pose_map_xy = self.point_world_to_map(raw.pose_world_xy)
        vel_map_xy = self.vector_world_to_map(raw.vel_world_xy)
        acc_map_xy = self.vector_world_to_map(raw.acc_world_xy)
        yaw_map = self.yaw_world_to_map(raw.yaw_world)
        speed = float(np.linalg.norm(vel_map_xy))
        return CanonicalVehicleState(
            index=agent_index,
            vehicle_name=raw.vehicle_name,
            pose_map_xy=pose_map_xy,
            yaw_map=yaw_map,
            vel_map_xy=vel_map_xy,
            acc_map_xy=acc_map_xy,
            speed=speed,
        )

