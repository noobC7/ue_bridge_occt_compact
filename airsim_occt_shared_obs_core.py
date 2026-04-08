from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from airsim_occt_geometry import project_point_to_segment, transform_points_global_to_local, wrap_angle
from airsim_occt_obs_manifest import build_obs_layout


@dataclass
class NormalizerConfig:
    pos: np.ndarray
    error_pos: float
    v: float
    error_v: float
    rot: float
    action_steering: float
    action_acc: float
    distance_lanelet: float
    distance_ref: float
    distance_agent: float
    obs_relative_velocity_scale: float
    obs_relative_acceleration_scale: float


class SharedObsCore:
    def __init__(self, obs_cfg, control_cfg, fleet_registry, projector=None) -> None:
        self.obs_cfg = obs_cfg
        self.control_cfg = control_cfg
        self.registry = fleet_registry
        self.projector = projector
        self.layout = build_obs_layout(obs_cfg)
        self.agent_width = 1.5
        self.lane_width = 6.0

    def build_normalizers(self, agent_length: float, agent_width: float, lane_width: float) -> NormalizerConfig:
        obs_relative_velocity_scale = self.obs_cfg.obs_relative_velocity_scale
        if obs_relative_velocity_scale is None:
            obs_relative_velocity_scale = max(self.control_cfg.max_speed / 4.0, 1.0)
        obs_relative_acceleration_scale = self.obs_cfg.obs_relative_acceleration_scale
        if obs_relative_acceleration_scale is None:
            obs_relative_acceleration_scale = max(self.control_cfg.max_acceleration, 0.5)
        self.agent_width = float(agent_width)
        self.lane_width = float(lane_width)
        return NormalizerConfig(
            pos=np.asarray([agent_length * 5.0, agent_width * 5.0], dtype=np.float32),
            error_pos=float(agent_length),
            v=float(self.control_cfg.max_speed),
            error_v=float(obs_relative_velocity_scale),
            rot=float(2.0 * np.pi),
            action_steering=float(self.control_cfg.max_steering_angle),
            action_acc=float(self.control_cfg.max_acceleration),
            distance_lanelet=float(lane_width * 3.0),
            distance_ref=float(lane_width * 3.0),
            distance_agent=float(agent_length * 10.0),
            obs_relative_velocity_scale=float(obs_relative_velocity_scale),
            obs_relative_acceleration_scale=float(obs_relative_acceleration_scale),
        )

    def update(self, history, states: List, projections: List, last_actions: np.ndarray) -> None:
        history.set_agent_s(np.asarray([projection.s for projection in projections], dtype=np.float32))
        self._update_distances(history, states, projections)
        self._update_relative_buffers(history, states, projections)
        self._update_action_buffers(history, last_actions)
        self._update_tracking_terms(history, states, projections)

    def encode(self, agent_index: int, history, state=None) -> np.ndarray:
        named_obs = self.encode_named(agent_index, history, state=state)
        blocks = [np.asarray(named_obs[key], dtype=np.float32).reshape(-1) for key in self.layout.actor_block_order]
        obs = np.concatenate(blocks, axis=0).astype(np.float32)
        if obs.shape[0] != self.layout.total_dim:
            raise ValueError(f"obs dim mismatch, expected {self.layout.total_dim}, got {obs.shape[0]}")
        return obs

    def encode_named(self, agent_index: int, history, state=None) -> Dict[str, np.ndarray]:
        current_state = state
        if current_state is None:
            raise ValueError("encode_named() requires the current canonical vehicle state")

        normalizers = history.normalizers
        self_ref = history.latest_self_ref(agent_index)
        left_boundary, right_boundary = history.latest_self_boundary(agent_index)
        n_short = int(self.obs_cfg.n_points_short_term)
        left_dis = np.linalg.norm(left_boundary[1 : 1 + n_short] - self_ref[:, :2], axis=-1, keepdims=True)
        right_dis = np.linalg.norm(right_boundary[1 : 1 + n_short] - self_ref[:, :2], axis=-1, keepdims=True)

        vel_long = history.past_vel.latest()[agent_index, agent_index, 0:1]
        hinge_preview = self._build_self_hinge_preview(agent_index, history, current_state)
        named_obs = {
            "self_vel": vel_long.astype(np.float32),
            "self_speed": np.abs(vel_long).astype(np.float32),
            "self_steering": history.past_steering.latest()[agent_index : agent_index + 1].astype(np.float32),
            "self_acc": history.past_action_acc.latest()[agent_index : agent_index + 1].astype(np.float32),
            "self_ref_velocity": self_ref[:, 2:3].astype(np.float32),
            "self_ref_points": self_ref[:, :2].astype(np.float32),
            "self_left_boundary_distance": left_dis.astype(np.float32),
            "self_right_boundary_distance": right_dis.astype(np.float32),
            "self_hinge_status": np.asarray([float(history.hinge_status[agent_index])], dtype=np.float32),
            "self_hinge_preview_info": hinge_preview.astype(np.float32),
            "self_hinge_past_info": hinge_preview[0].astype(np.float32),
            "self_hinge_error_vel": np.asarray(
                [history.past_hinge_error_vel.latest()[agent_index, 0] / normalizers.error_v],
                dtype=np.float32,
            ),
            "self_distance_to_ref": np.asarray(
                [float(np.linalg.norm(self_ref[0, :2]))],
                dtype=np.float32,
            ),
            "self_distance_to_left_boundary": history.past_distance_to_left_boundary.latest()[
                agent_index : agent_index + 1
            ].astype(np.float32),
            "self_distance_to_right_boundary": history.past_distance_to_right_boundary.latest()[
                agent_index : agent_index + 1
            ].astype(np.float32),
            "self_platoon_error_vel": (
                history.past_platoon_error_vel.latest()[agent_index] / normalizers.error_v
            ).astype(np.float32),
            "self_platoon_error_space": (
                history.self_platoon_error_space.latest()[agent_index] / normalizers.error_pos
            ).astype(np.float32),
        }
        named_obs.update(self._encode_nearest_neighbors(agent_index, history))
        return named_obs

    def _build_self_hinge_preview(self, agent_index: int, history, state) -> np.ndarray:
        normalizers = history.normalizers
        hinge_raw = history.hinge_short_term[agent_index]
        hinge_local_xy = transform_points_global_to_local(
            state.pose_map_xy,
            state.yaw_map,
            hinge_raw[:, :2],
        )
        hinge_pos = np.clip(hinge_local_xy / normalizers.pos, -1.0, 1.0)
        hinge_speed = np.linalg.norm(hinge_raw[:, 2:4], axis=-1, keepdims=True) / max(normalizers.v, 1e-6)
        hinge_boundary_margin = np.clip(
            hinge_raw[:, 4:5] / max(self.lane_width * 0.5, 1e-6),
            -1.0,
            1.0,
        )
        return np.concatenate([hinge_pos, hinge_speed, hinge_boundary_margin], axis=-1).astype(np.float32)

    def _encode_nearest_neighbors(self, ego_index: int, history) -> Dict[str, np.ndarray]:
        normalizers = history.normalizers
        neighbor_indices = self._get_nearest_neighbor_indices(ego_index, history)
        n_near = int(self.obs_cfg.n_nearing_agents_observed)
        pos = np.zeros((n_near, 2), dtype=np.float32)
        rot = np.zeros((n_near, 1), dtype=np.float32)
        relative_long = np.zeros((n_near, 1), dtype=np.float32)
        relative_acc = np.zeros((n_near, 1), dtype=np.float32)
        distance = np.zeros((n_near, 1), dtype=np.float32)

        for slot, neighbor_index in enumerate(neighbor_indices):
            if neighbor_index is None:
                continue
            pos[slot] = history.past_pos.latest()[ego_index, neighbor_index]
            rot[slot, 0] = history.past_rot.latest()[ego_index, neighbor_index]
            distance[slot, 0] = history.latest_neighbor_distance(ego_index, neighbor_index)

            relative_long_hist = self._get_relative_longitudinal_velocity_history(history, ego_index, neighbor_index)
            relative_long[slot, 0] = relative_long_hist[0] / normalizers.obs_relative_velocity_scale
            if relative_long_hist.size > 1:
                relative_acc[slot, 0] = (
                    (relative_long_hist[0] - relative_long_hist[1])
                    / max(self.control_cfg.dt, 1e-6)
                    / normalizers.obs_relative_acceleration_scale
                )

        return {
            "others_pos": pos,
            "others_rot": rot,
            "others_relative_longitudinal_velocity": relative_long,
            "others_relative_acceleration": relative_acc,
            "others_distance": distance,
        }

    def _get_nearest_neighbor_indices(self, ego_index: int, history) -> List[Optional[int]]:
        pairwise = history.past_distance_to_agents.latest()[ego_index].copy()
        candidate_indices = [idx for idx in range(self.registry.n_agents) if idx != ego_index]
        candidate_indices.sort(key=lambda idx: float(pairwise[idx]))
        selected = sorted(candidate_indices[: int(self.obs_cfg.n_nearing_agents_observed)])
        while len(selected) < int(self.obs_cfg.n_nearing_agents_observed):
            selected.append(None)
        return selected

    def _get_relative_longitudinal_velocity_history(self, history, ego_index: int, neighbor_index: int) -> np.ndarray:
        n_observed_steps = int(self.obs_cfg.n_observed_steps)
        other_hist = history.last_n_longitudinal_velocities(ego_index, neighbor_index, n_observed_steps) * history.normalizers.v
        ego_hist = history.last_n_longitudinal_velocities(ego_index, ego_index, n_observed_steps) * history.normalizers.v
        return (other_hist - ego_hist).astype(np.float32)

    def _update_distances(self, history, states: List, projections: List) -> None:
        n_agents = len(states)
        pairwise = np.zeros((n_agents, n_agents), dtype=np.float32)
        for i in range(n_agents):
            for j in range(n_agents):
                pairwise[i, j] = float(np.linalg.norm(states[i].pose_map_xy - states[j].pose_map_xy))
        history.past_distance_to_agents.add(pairwise / history.normalizers.distance_lanelet)
        history.past_distance_to_ref_path.add(
            np.asarray([projection.dist_to_ref for projection in projections], dtype=np.float32)
            / history.normalizers.distance_lanelet
        )
        history.past_distance_to_left_boundary.add(
            np.asarray([projection.dist_to_left_boundary for projection in projections], dtype=np.float32)
            / history.normalizers.distance_lanelet
        )
        history.past_distance_to_right_boundary.add(
            np.asarray([projection.dist_to_right_boundary for projection in projections], dtype=np.float32)
            / history.normalizers.distance_lanelet
        )

    def _update_relative_buffers(self, history, states: List, projections: List) -> None:
        n_agents = len(states)
        n_short = int(self.obs_cfg.n_points_short_term)
        n_boundary = int(self.obs_cfg.n_points_nearing_boundary)
        pos_rel = np.zeros((n_agents, n_agents, 2), dtype=np.float32)
        rot_rel = np.zeros((n_agents, n_agents), dtype=np.float32)
        vel_rel = np.zeros((n_agents, n_agents, 2), dtype=np.float32)
        steering_agents = np.zeros((n_agents,), dtype=np.float32)
        ref_rel = np.zeros((n_agents, n_agents, n_short, 3), dtype=np.float32)
        left_rel = np.zeros((n_agents, n_agents, n_boundary, 2), dtype=np.float32)
        right_rel = np.zeros((n_agents, n_agents, n_boundary, 2), dtype=np.float32)
        vertices_rel = np.zeros((n_agents, n_agents, 4, 2), dtype=np.float32)

        for ego_index, ego_state in enumerate(states):
            steering_agents[ego_index] = ego_state.steering_feedback
            for other_index, other_state in enumerate(states):
                pos_rel[ego_index, other_index] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    other_state.pose_map_xy,
                )
                rel_yaw = float(wrap_angle(other_state.yaw_map - ego_state.yaw_map))
                rot_rel[ego_index, other_index] = rel_yaw
                vel_rel[ego_index, other_index] = np.asarray(
                    [
                        other_state.speed * np.cos(rel_yaw),
                        other_state.speed * np.sin(rel_yaw),
                    ],
                    dtype=np.float32,
                )
                ref_rel[ego_index, other_index, :, :2] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    projections[other_index].short_term_ref[:, :2],
                )
                ref_rel[ego_index, other_index, :, 2] = projections[other_index].short_term_ref[:, 2]
                left_rel[ego_index, other_index] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    projections[other_index].left_boundary_pts,
                )
                right_rel[ego_index, other_index] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    projections[other_index].right_boundary_pts,
                )
                vertices_rel[ego_index, other_index] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    projections[other_index].vertices_xy,
                )
        history.past_pos.add(pos_rel / history.normalizers.pos)
        history.past_rot.add(rot_rel / history.normalizers.rot)
        history.past_vel.add(vel_rel / history.normalizers.v)
        history.past_steering.add(steering_agents / history.normalizers.action_steering)
        ref_norm = ref_rel.copy()
        ref_norm[:, :, :, :2] = ref_norm[:, :, :, :2] / history.normalizers.pos
        ref_norm[:, :, :, 2] = ref_norm[:, :, :, 2] / history.normalizers.v
        history.past_relative_ref_info.add(ref_norm)
        history.past_left_boundary.add(left_rel / history.normalizers.pos)
        history.past_right_boundary.add(right_rel / history.normalizers.pos)
        history.past_vertices.add(vertices_rel / history.normalizers.pos)

    def _update_action_buffers(self, history, last_actions: np.ndarray) -> None:
        history.past_action_acc.add(last_actions[:, 0] / history.normalizers.action_acc)
        history.past_action_steering.add(last_actions[:, 1] / history.normalizers.action_steering)

    def _update_tracking_terms(self, history, states: List, projections: List) -> None:
        target_agent_s = self._get_dynamic_target_arc_positions(history.agent_s)
        platoon_error_space = np.zeros((self.registry.n_agents, 2), dtype=np.float32)
        platoon_error_space[:, 0] = history.agent_s - target_agent_s
        history.self_platoon_error_space.add(platoon_error_space)

        hinge_short_term = self._build_hinge_short_term(states, projections)
        hinge_status = self._compute_hinge_status(history.agent_s, hinge_short_term)
        agent_hinge_status = self._compute_agent_hinge_status(history, states, hinge_short_term, hinge_status)

        platoon_error_vel = np.zeros((self.registry.n_agents, 2), dtype=np.float32)
        hinge_error_vel = np.zeros((self.registry.n_agents, 2), dtype=np.float32)
        speeds = np.asarray([state.speed for state in states], dtype=np.float32)
        for agent_index in range(1, self.registry.n_agents - 1):
            platoon_error_vel[agent_index, 0] = speeds[agent_index - 1] - speeds[agent_index]
            platoon_error_vel[agent_index, 1] = speeds[agent_index + 1] - speeds[agent_index]
            hinge_speed = float(np.linalg.norm(hinge_short_term[agent_index, 0, 2:4]))
            hinge_error = speeds[agent_index] - hinge_speed
            hinge_error_vel[agent_index] = hinge_error

        history.hinge_short_term = hinge_short_term.astype(np.float32)
        history.hinge_status = hinge_status.astype(np.bool_)
        history.agent_hinge_status.add(agent_hinge_status.astype(np.bool_))
        history.platoon_error_vel = platoon_error_vel.astype(np.float32)
        history.hinge_error_vel = hinge_error_vel.astype(np.float32)
        history.error_vel = platoon_error_vel.astype(np.float32)
        history.past_platoon_error_vel.add(platoon_error_vel.astype(np.float32))
        history.past_hinge_error_vel.add(hinge_error_vel.astype(np.float32))

    def _get_dynamic_target_arc_positions(self, agent_s: np.ndarray) -> np.ndarray:
        if self.registry.n_agents <= 1:
            return agent_s.copy()
        s_front = float(agent_s[0])
        s_rear = float(agent_s[-1])
        desired_gap_s = (s_front - s_rear) / max(self.registry.n_agents - 1, 1)
        agent_indices = np.arange(self.registry.n_agents, dtype=np.float32)
        return (s_front - desired_gap_s * agent_indices).astype(np.float32)

    def _build_hinge_short_term(self, states: List, projections: List) -> np.ndarray:
        n_agents = self.registry.n_agents
        n_short = int(self.obs_cfg.n_points_short_term)
        hinge_short_term = np.zeros((n_agents, n_short, 5), dtype=np.float32)
        if n_agents == 0:
            return hinge_short_term

        front_ref = np.asarray(projections[0].short_term_ref[:, :2], dtype=np.float32)
        rear_ref = np.asarray(projections[-1].short_term_ref[:, :2], dtype=np.float32)
        front_vel = np.asarray(states[0].vel_map_xy, dtype=np.float32)
        rear_vel = np.asarray(states[-1].vel_map_xy, dtype=np.float32)
        for agent_index in range(n_agents):
            ratio = 0.0 if n_agents == 1 else float(agent_index) / float(n_agents - 1)
            hinge_short_term[agent_index, :, :2] = front_ref + ratio * (rear_ref - front_ref)
            hinge_short_term[agent_index, :, 2:4] = front_vel[None, :] + ratio * (rear_vel - front_vel)[None, :]
            for point_index in range(n_short):
                hinge_short_term[agent_index, point_index, 4] = self._signed_boundary_margin(
                    hinge_short_term[agent_index, point_index, :2]
                )
        return hinge_short_term

    def _compute_hinge_status(self, agent_s: np.ndarray, hinge_short_term: np.ndarray) -> np.ndarray:
        n_agents = self.registry.n_agents
        hinge_status = np.zeros((n_agents,), dtype=np.bool_)
        hinge_edge_buffer = self.obs_cfg.hinge_edge_buffer
        if hinge_edge_buffer is None:
            hinge_edge_buffer = self.agent_width * 0.6
        corner_s_begin = self._get_corner_s_begin()
        for agent_index in range(1, n_agents - 1):
            hinge_ready = hinge_short_term[agent_index, :, 4] > float(hinge_edge_buffer)
            is_block, block_order = self._check_boolean_block(hinge_ready)
            is_after_corner = float(agent_s[agent_index]) > corner_s_begin
            hinge_status[agent_index] = bool(is_block and block_order in (0, 2) and is_after_corner)
        return hinge_status

    def _compute_agent_hinge_status(self, history, states: List, hinge_short_term: np.ndarray, hinge_status: np.ndarray) -> np.ndarray:
        current = np.zeros((self.registry.n_agents,), dtype=np.bool_)
        if self.registry.n_agents <= 2:
            return current

        prev = history.agent_hinge_status.latest().astype(np.bool_)
        hinge_heading = hinge_short_term[:, 1, :2] - hinge_short_term[:, 0, :2]
        hinge_heading_norm = np.linalg.norm(hinge_heading, axis=-1, keepdims=True)
        hinge_heading_tangent = hinge_heading / np.clip(hinge_heading_norm, 1e-6, None)
        hinge_pos = hinge_short_term[:, 0, :2]
        hinge_vel = hinge_short_term[:, 0, 2:4]
        hinge_vel_mag = np.linalg.norm(hinge_vel, axis=-1)
        heading_cos_threshold = float(np.cos(np.deg2rad(2.0)))

        for agent_index in range(1, self.registry.n_agents - 1):
            state = states[agent_index]
            agent_pos = np.asarray(state.pose_map_xy, dtype=np.float32)
            agent_heading = np.asarray([np.cos(state.yaw_map), np.sin(state.yaw_map)], dtype=np.float32)
            agent_speed = float(np.linalg.norm(np.asarray(state.vel_map_xy, dtype=np.float32)))
            agent_pos_legal = float(np.linalg.norm(hinge_pos[agent_index] - agent_pos)) < 0.15
            agent_heading_legal = float(np.dot(hinge_heading_tangent[agent_index], agent_heading)) > heading_cos_threshold
            agent_vel_legal = abs(agent_speed - float(hinge_vel_mag[agent_index])) < 0.75
            current[agent_index] = bool((agent_pos_legal and agent_heading_legal and agent_vel_legal) or prev[agent_index])

        return np.logical_and(current, hinge_status)

    def _get_corner_s_begin(self) -> float:
        if self.projector is None:
            return 0.0
        road = getattr(self.projector, "road", None)
        road_env_index = int(getattr(self.projector, "road_env_index", 0))
        if road is None:
            return 0.0
        for attr_name in ("batch_corner_s_begin", "batch_corner_s"):
            if hasattr(road, attr_name):
                value = getattr(road, attr_name)
                value_np = np.asarray(value[road_env_index], dtype=np.float32).reshape(-1)
                if value_np.size:
                    return float(value_np[0])
        return 0.0

    def _signed_boundary_margin(self, point_xy: np.ndarray) -> float:
        if self.projector is None:
            return 0.0
        left_dist, left_in_bound = self._distance_to_nearest_boundary_segment(
            point_xy,
            np.asarray(self.projector.left_xy, dtype=np.float32),
            is_left_boundary=True,
        )
        right_dist, right_in_bound = self._distance_to_nearest_boundary_segment(
            point_xy,
            np.asarray(self.projector.right_xy, dtype=np.float32),
            is_left_boundary=False,
        )
        left_margin = left_dist if left_in_bound else -left_dist
        right_margin = right_dist if right_in_bound else -right_dist
        return float(min(left_margin, right_margin))

    def _distance_to_nearest_boundary_segment(
        self,
        point_xy: np.ndarray,
        polyline_xy: np.ndarray,
        *,
        is_left_boundary: bool,
    ) -> tuple[float, bool]:
        best_dist = float("inf")
        best_in_bound = False
        point = np.asarray(point_xy, dtype=np.float32)
        for segment_index in range(len(polyline_xy) - 1):
            seg_a = np.asarray(polyline_xy[segment_index], dtype=np.float32)
            seg_b = np.asarray(polyline_xy[segment_index + 1], dtype=np.float32)
            _, _, dist = project_point_to_segment(point, seg_a, seg_b)
            if dist >= best_dist:
                continue
            seg_vec = seg_b - seg_a
            rel = point - seg_a
            cross = float(rel[0] * seg_vec[1] - rel[1] * seg_vec[0])
            best_dist = float(dist)
            best_in_bound = cross > 0.0 if is_left_boundary else cross < 0.0
        return best_dist, best_in_bound

    def _check_boolean_block(self, values: np.ndarray) -> tuple[bool, int]:
        bool_values = np.asarray(values, dtype=np.bool_)
        if bool_values.size == 0:
            return False, -1
        as_int = bool_values.astype(np.int32)
        diff = np.diff(as_int)
        switch_count = int(np.abs(diff).sum())
        is_block = switch_count <= 1
        if bool_values.all():
            block_order = 2
        elif (~bool_values).all():
            block_order = -1
        elif diff.max(initial=0) > 0:
            block_order = 0
        else:
            block_order = 1
        return is_block, block_order
