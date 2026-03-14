from typing import List, Optional, Tuple, Union

import numpy as np

from airsim_occt_geometry import (
    build_arc_length,
    distance_point_to_polyline,
    interpolate_polyline_by_s,
    project_point_to_polyline,
    vehicle_rectangle_vertices,
)
from airsim_occt_schema import RoadProjection


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class OcctMapProjector:
    def __init__(self, road, fleet_registry, obs_cfg, road_env_index: int = 0) -> None:
        self.road = road
        self.registry = fleet_registry
        self.obs_cfg = obs_cfg
        self.road_env_index = road_env_index
        self.center_s, self.center_xy, self.left_xy, self.right_xy, self.ref_v = self._extract_road_arrays()
        self.s_min = float(self.center_s[0])
        self.s_max = float(self.center_s[-1])

    def project_all(self, states: List, prev_s_list: List[Optional[float]]) -> List[RoadProjection]:
        return [self.project_one(state, prev_s) for state, prev_s in zip(states, prev_s_list)]

    def project_one(self, state, prev_s: Optional[float]) -> RoadProjection:
        projection_point_xy, projection_mode = self._get_projection_point(state)
        s = self._estimate_s(projection_point_xy, prev_s)
        short_term_ref = self._build_short_term_ref(s)
        left_pts, right_pts = self._build_boundary_points(s)
        vehicle_cfg = self.registry.config_of(state.index)
        vertices_xy = vehicle_rectangle_vertices(
            center_xy=state.pose_map_xy,
            yaw=state.yaw_map,
            length=vehicle_cfg.length,
            width=vehicle_cfg.width,
        )
        dist_to_ref = distance_point_to_polyline(projection_point_xy, self.center_xy, self.center_s)
        dist_to_left_boundary = self._boundary_clearance(
            center_xy=state.pose_map_xy,
            vertices_xy=vertices_xy,
            boundary_xy=self.left_xy,
            width=vehicle_cfg.width,
        )
        dist_to_right_boundary = self._boundary_clearance(
            center_xy=state.pose_map_xy,
            vertices_xy=vertices_xy,
            boundary_xy=self.right_xy,
            width=vehicle_cfg.width,
        )
        return RoadProjection(
            s=s,
            projection_point_map=projection_point_xy.astype(np.float32),
            projection_point_mode=projection_mode,
            closest_center_xy=self._query_xy(self.center_xy, s),
            tangent_yaw=self._query_tangent_yaw(s),
            ref_v=float(self._query_ref_v(s)),
            short_term_ref=short_term_ref,
            left_boundary_pts=left_pts,
            right_boundary_pts=right_pts,
            dist_to_ref=float(dist_to_ref),
            dist_to_left_boundary=float(dist_to_left_boundary),
            dist_to_right_boundary=float(dist_to_right_boundary),
            vertices_xy=vertices_xy,
        )

    def _get_projection_point(self, state) -> Tuple[np.ndarray, str]:
        return np.asarray(state.pose_map_xy, dtype=np.float32), "body_origin"

    def _extract_road_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if hasattr(self.road, "batch_s") and hasattr(self.road, "batch_center_vertices"):
            s = _to_numpy(self.road.batch_s[self.road_env_index]).astype(np.float32)
            center_xy = _to_numpy(self.road.batch_center_vertices[self.road_env_index]).astype(np.float32)
            left_xy = _to_numpy(self.road.batch_left_vertices[self.road_env_index]).astype(np.float32)
            right_xy = _to_numpy(self.road.batch_right_vertices[self.road_env_index]).astype(np.float32)
            ref_v = _to_numpy(self.road.batch_ref_v[self.road_env_index]).reshape(-1).astype(np.float32)
            valid = ~np.isnan(s)
            return s[valid], center_xy[valid], left_xy[valid], right_xy[valid], ref_v[valid]
        center_xy = _to_numpy(self.road.get_road_center_pts()).astype(np.float32)
        left_xy = _to_numpy(self.road.get_road_left_pts()).astype(np.float32)
        right_xy = _to_numpy(self.road.get_road_right_pts()).astype(np.float32)
        if center_xy.ndim == 3:
            center_xy = center_xy[self.road_env_index]
            left_xy = left_xy[self.road_env_index]
            right_xy = right_xy[self.road_env_index]
        center_s = build_arc_length(center_xy)
        ref_v = np.zeros((len(center_s),), dtype=np.float32)
        return center_s, center_xy, left_xy, right_xy, ref_v

    def _estimate_s(self, pose_map_xy: np.ndarray, prev_s: Optional[float]) -> float:
        if prev_s is None:
            return self._global_search_s(pose_map_xy)
        return self._local_search_s(pose_map_xy, prev_s)

    def _global_search_s(self, pose_map_xy: np.ndarray) -> float:
        _, best_s, _, _ = project_point_to_polyline(pose_map_xy, self.center_xy, self.center_s)
        return float(best_s)

    def _local_search_s(self, pose_map_xy: np.ndarray, prev_s: float, search_radius: float = 15.0) -> float:
        low = max(self.s_min, float(prev_s - search_radius))
        high = min(self.s_max, float(prev_s + search_radius))
        start = max(int(np.searchsorted(self.center_s, low, side="left")) - 1, 0)
        end = min(int(np.searchsorted(self.center_s, high, side="right")) + 1, len(self.center_s))
        if end - start < 2:
            return self._global_search_s(pose_map_xy)
        sub_s = self.center_s[start:end]
        sub_xy = self.center_xy[start:end]
        _, best_s, _, _ = project_point_to_polyline(pose_map_xy, sub_xy, sub_s)
        return float(best_s)

    def _clamp_s(self, query_s: np.ndarray) -> np.ndarray:
        return np.clip(query_s, self.s_min, self.s_max).astype(np.float32)

    def _query_xy(self, polyline_xy: np.ndarray, s: Union[float, np.ndarray]) -> np.ndarray:
        return interpolate_polyline_by_s(self.center_s, polyline_xy, self._clamp_s(np.asarray(s)))

    def _query_ref_v(self, s: Union[float, np.ndarray]) -> np.ndarray:
        return interpolate_polyline_by_s(self.center_s, self.ref_v, self._clamp_s(np.asarray(s)))

    def _query_tangent_yaw(self, s: float) -> float:
        ds = min(0.5, max((self.s_max - self.s_min) / max(len(self.center_s), 2), 0.05))
        s0 = max(self.s_min, s - ds)
        s1 = min(self.s_max, s + ds)
        p0 = self._query_xy(self.center_xy, s0)
        p1 = self._query_xy(self.center_xy, s1)
        tangent = np.asarray(p1 - p0, dtype=np.float32)
        return float(np.arctan2(tangent[1], tangent[0]))

    def _build_short_term_ref(self, s: float) -> np.ndarray:
        query_s = self._clamp_s(s + np.arange(self.obs_cfg.n_points_short_term, dtype=np.float32) * self.obs_cfg.sample_interval)
        ref_xy = self._query_xy(self.center_xy, query_s)
        ref_v = self._query_ref_v(query_s).reshape(-1, 1)
        return np.concatenate([ref_xy, ref_v], axis=-1).astype(np.float32)

    def _build_boundary_points(self, s: float) -> Tuple[np.ndarray, np.ndarray]:
        start_s = s + float(self.obs_cfg.boundary_offset)
        query_s = self._clamp_s(
            start_s + np.arange(self.obs_cfg.n_points_nearing_boundary, dtype=np.float32) * self.obs_cfg.sample_interval
        )
        left_pts = self._query_xy(self.left_xy, query_s)
        right_pts = self._query_xy(self.right_xy, query_s)
        return left_pts.astype(np.float32), right_pts.astype(np.float32)

    def _boundary_clearance(
        self,
        center_xy: np.ndarray,
        vertices_xy: np.ndarray,
        boundary_xy: np.ndarray,
        width: float,
    ) -> float:
        center_dist = distance_point_to_polyline(center_xy, boundary_xy, self.center_s) - 0.5 * width
        vertex_dists = [distance_point_to_polyline(v, boundary_xy, self.center_s) for v in vertices_xy]
        return float(min([center_dist] + vertex_dists))
