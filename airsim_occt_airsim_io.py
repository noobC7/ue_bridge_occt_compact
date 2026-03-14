import math
import time
from typing import Dict, List, Optional

import numpy as np

import ivs

from airsim_occt_schema import LowLevelCommand, RawAirSimState


class AirSimIO:
    def __init__(self, env_cfg) -> None:
        self.cfg = env_cfg
        self.client = ivs.VSimClient(ip=env_cfg.host, port=env_cfg.port)

    def list_vehicles(self) -> List[str]:
        return sorted(self.client.listVehicles(), key=self._vehicle_sort_key)

    def get_settings_string(self) -> str:
        return self.client.getSettingsString()

    def connect(self) -> None:
        self.client.confirmConnection()

    def enable_api(self, vehicle_names: List[str]) -> None:
        for vehicle_name in vehicle_names:
            self.client.enableApiControl(True, vehicle_name)

    def disable_api(self, vehicle_names: List[str]) -> None:
        for vehicle_name in vehicle_names:
            self.client.enableApiControl(False, vehicle_name)

    def pause(self, is_paused: bool) -> None:
        self.client.simPause(is_paused)

    def reset(self) -> None:
        self.client.reset()

    def advance(self, dt: float) -> None:
        self.client.simContinueForTime(dt)

    def read_all(self, vehicle_names: List[str]) -> List[RawAirSimState]:
        return [self.read_vehicle_state(vehicle_name) for vehicle_name in vehicle_names]

    def read_vehicle_state(self, vehicle_name: str) -> RawAirSimState:
        pose = self.client.simGetObjectPose(vehicle_name)
        if getattr(pose.position, "containsNan", lambda: False)():
            pose = self.client.simGetVehiclePose(vehicle_name)
        kin = self.client.simGetGroundTruthKinematics(vehicle_name)
        imu = self.client.getImuData(vehicle_name=vehicle_name)
        gps = self.client.getGpsData(vehicle_name=vehicle_name)
        timestamp = float(getattr(imu, "time_stamp", 0)) or time.time()
        pose_world_xy = np.asarray([pose.position.x_val, pose.position.y_val], dtype=np.float32)
        vel_world_xy = np.asarray([kin.linear_velocity.x_val, kin.linear_velocity.y_val], dtype=np.float32)
        acc_world_xy = np.asarray([kin.linear_acceleration.x_val, kin.linear_acceleration.y_val], dtype=np.float32)
        imu_acc_body = np.asarray(
            [imu.linear_acceleration.x_val, imu.linear_acceleration.y_val],
            dtype=np.float32,
        )
        imu_gyro_body = np.asarray(
            [imu.angular_velocity.x_val, imu.angular_velocity.y_val],
            dtype=np.float32,
        )
        return RawAirSimState(
            vehicle_name=vehicle_name,
            timestamp=float(timestamp),
            pose_world_xy=pose_world_xy,
            yaw_world=self._quat_to_yaw(pose.orientation),
            z_world=float(pose.position.z_val),
            vel_world_xy=vel_world_xy,
            acc_world_xy=acc_world_xy,
            yaw_rate=float(kin.angular_velocity.z_val),
            imu_acc_body=imu_acc_body,
            imu_gyro_body=imu_gyro_body,
            gps_lat_lon_alt=self._gps_to_array_or_none(gps),
        )

    def send_control(self, vehicle_name: str, cmd: LowLevelCommand) -> None:
        controls = ivs.CarControls()
        controls.throttle = float(cmd.throttle)
        controls.brake = float(cmd.brake)
        controls.steering = float(cmd.steering)
        self.client.setCarControls(controls, vehicle_name)

    def send_all(self, command_map: Dict[str, LowLevelCommand]) -> None:
        for vehicle_name, command in command_map.items():
            self.send_control(vehicle_name, command)

    def flush_persistent_markers(self) -> None:
        self.client.simFlushPersistentMarkers()

    def plot_line_strip_world(
        self,
        points_xyz: np.ndarray,
        color_rgba: Optional[List[float]] = None,
        thickness: float = 5.0,
        duration: float = -1.0,
        is_persistent: bool = True,
    ) -> None:
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"points_xyz must have shape [N,2] or [N,3], got {points.shape}")
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        ivs_points = [ivs.Vector3r(float(x), float(y), float(z)) for x, y, z in points]
        self.client.simPlotLineStrip(
            ivs_points,
            color_rgba=color_rgba or [1.0, 0.0, 0.0, 1.0],
            thickness=thickness,
            duration=duration,
            is_persistent=is_persistent,
        )

    def plot_line_list_world(
        self,
        points_xyz: np.ndarray,
        color_rgba: Optional[List[float]] = None,
        thickness: float = 5.0,
        duration: float = -1.0,
        is_persistent: bool = True,
    ) -> None:
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"points_xyz must have shape [N,2] or [N,3], got {points.shape}")
        if points.shape[0] % 2 != 0:
            raise ValueError(f"simPlotLineList requires an even number of points, got {points.shape[0]}")
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        ivs_points = [ivs.Vector3r(float(x), float(y), float(z)) for x, y, z in points]
        self.client.simPlotLineList(
            ivs_points,
            color_rgba=color_rgba or [1.0, 0.0, 0.0, 1.0],
            thickness=thickness,
            duration=duration,
            is_persistent=is_persistent,
        )

    def plot_points_world(
        self,
        points_xyz: np.ndarray,
        color_rgba: Optional[List[float]] = None,
        size: float = 10.0,
        duration: float = -1.0,
        is_persistent: bool = True,
    ) -> None:
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(f"points_xyz must have shape [N,2] or [N,3], got {points.shape}")
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        ivs_points = [ivs.Vector3r(float(x), float(y), float(z)) for x, y, z in points]
        self.client.simPlotPoints(
            ivs_points,
            color_rgba=color_rgba or [1.0, 0.0, 0.0, 1.0],
            size=size,
            duration=duration,
            is_persistent=is_persistent,
        )

    def _quat_to_yaw(self, quat) -> float:
        x = float(quat.x_val)
        y = float(quat.y_val)
        z = float(quat.z_val)
        w = float(quat.w_val)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    def _gps_to_array_or_none(self, gps) -> Optional[np.ndarray]:
        if not getattr(gps, "is_valid", False):
            return None
        geo = gps.gnss.geo_point
        return np.asarray([geo.latitude, geo.longitude, geo.altitude], dtype=np.float64)

    def _vehicle_sort_key(self, vehicle_name: str):
        prefix = vehicle_name.rstrip('0123456789')
        suffix = vehicle_name[len(prefix):]
        if suffix.isdigit():
            return (prefix, int(suffix))
        return (vehicle_name, -1)
