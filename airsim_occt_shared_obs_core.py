from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from airsim_occt_geometry import heading_to_local_velocity, transform_points_global_to_local, wrap_angle
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
    def __init__(self, obs_cfg, control_cfg, fleet_registry) -> None:
        self.obs_cfg = obs_cfg
        self.control_cfg = control_cfg
        self.registry = fleet_registry
        self.layout = build_obs_layout(obs_cfg)

    def build_normalizers(self, agent_length: float, agent_width: float, lane_width: float) -> NormalizerConfig:
        obs_relative_velocity_scale = self.obs_cfg.obs_relative_velocity_scale
        if obs_relative_velocity_scale is None:
            obs_relative_velocity_scale = max(self.control_cfg.max_speed / 4.0, 1.0)
        obs_relative_acceleration_scale = self.obs_cfg.obs_relative_acceleration_scale
        if obs_relative_acceleration_scale is None:
            obs_relative_acceleration_scale = max(self.control_cfg.max_acceleration, 0.5)
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
        vel_rel_raw = self._update_relative_buffers(history, states, projections)
        self._update_action_buffers(history, last_actions)
        self._update_error_terms(history, vel_rel_raw)
        history.agent_target_hinge_short_term = self._build_hinge_info(states, projections)

    def encode(self, agent_index: int, history) -> np.ndarray:
        obs_self = self.encode_self(agent_index, history)
        obs_others = self.encode_adjacent(agent_index, history)
        obs = np.hstack([obs_self, obs_others]).astype(np.float32)
        if obs.shape[0] != self.layout.total_dim:
            raise ValueError(f"obs dim mismatch, expected {self.layout.total_dim}, got {obs.shape[0]}")
        return obs

    def encode_self(self, agent_index: int, history) -> np.ndarray:
        normalizers = history.normalizers
        self_ref = history.latest_self_ref(agent_index)
        left_boundary, right_boundary = history.latest_self_boundary(agent_index)
        n_short = int(self.obs_cfg.n_points_short_term)
        left_dis = np.linalg.norm(left_boundary[1 : 1 + n_short] - self_ref[:, :2], axis=-1)
        right_dis = np.linalg.norm(right_boundary[1 : 1 + n_short] - self_ref[:, :2], axis=-1)
        vel_long = float(history.past_vel.latest()[agent_index, agent_index, 0])
        hinge_raw = history.agent_target_hinge_short_term[agent_index]
        hinge_info = np.concatenate(
            [
                hinge_raw[:, :2] / normalizers.pos,
                hinge_raw[:, 2:4] / normalizers.v,
                hinge_raw[:, 4:5],
            ],
            axis=-1,
        ).reshape(-1)
        blocks = [
            np.asarray([vel_long], dtype=np.float32),
            np.asarray([abs(vel_long)], dtype=np.float32),
            np.asarray([vel_long], dtype=np.float32),
            np.asarray([history.past_steering.latest()[agent_index]], dtype=np.float32),
        ]
        if not self.obs_cfg.mask_ref_v:
            blocks.append(self_ref[:, 2].reshape(-1))
        blocks.extend(
            [
                self_ref[:, :2].reshape(-1),
                left_dis.reshape(-1),
                right_dis.reshape(-1),
            ]
        )
        if self.obs_cfg.include_hinge_info:
            blocks.append(hinge_info.astype(np.float32))
        blocks.extend(
            [
                np.asarray([history.past_distance_to_ref_path.latest()[agent_index]], dtype=np.float32),
                np.asarray([history.past_distance_to_left_boundary.latest()[agent_index]], dtype=np.float32),
                np.asarray([history.past_distance_to_right_boundary.latest()[agent_index]], dtype=np.float32),
                (history.error_vel[agent_index] / normalizers.error_v).astype(np.float32),
                (history.error_space.latest()[agent_index] / normalizers.error_pos).astype(np.float32),
            ]
        )
        return np.hstack(blocks).astype(np.float32)

    def encode_adjacent(self, agent_index: int, history) -> np.ndarray:
        neighbors = self.registry.neighbors_of(agent_index)
        front_block = self._encode_one_neighbor(agent_index, neighbors.front, history)
        rear_block = self._encode_one_neighbor(agent_index, neighbors.rear, history)
        return np.hstack([front_block, rear_block]).astype(np.float32)

    def _encode_one_neighbor(self, ego_index: int, neighbor_index: Optional[int], history) -> np.ndarray:
        if neighbor_index is None:
            return self._zero_neighbor_block()
        normalizers = history.normalizers
        obs_pos = history.past_pos.latest()[ego_index, neighbor_index].reshape(-1)
        obs_rot = np.asarray([history.past_rot.latest()[ego_index, neighbor_index]], dtype=np.float32)
        relative_long_hist = self._get_relative_longitudinal_velocity_history(history, ego_index, neighbor_index)
        if relative_long_hist.size > 1:
            relative_acc_hist = np.concatenate(
                [
                    np.zeros((1,), dtype=np.float32),
                    (relative_long_hist[1:] - relative_long_hist[:-1]) / max(self.control_cfg.dt, 1e-6),
                ]
            )
        else:
            relative_acc_hist = np.zeros((1,), dtype=np.float32)
        obs_relative_long = np.asarray(
            [relative_long_hist[0] / normalizers.obs_relative_velocity_scale],
            dtype=np.float32,
        )
        obs_relative_acc = (relative_acc_hist / normalizers.obs_relative_acceleration_scale).astype(np.float32)
        obs_distance = np.asarray([history.latest_neighbor_distance(ego_index, neighbor_index)], dtype=np.float32)
        return np.hstack([obs_pos, obs_rot, obs_relative_long, obs_relative_acc, obs_distance]).astype(np.float32)

    def _zero_neighbor_block(self) -> np.ndarray:
        return np.zeros((self.layout.neighbor_dim,), dtype=np.float32)

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

    def _update_relative_buffers(self, history, states: List, projections: List) -> np.ndarray:
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
                rel_yaw = float(wrap_angle(other_state.yaw_map - ego_state.yaw_map))
                pos_rel[ego_index, other_index] = transform_points_global_to_local(
                    ego_state.pose_map_xy,
                    ego_state.yaw_map,
                    other_state.pose_map_xy,
                )
                rot_rel[ego_index, other_index] = rel_yaw
                vel_rel[ego_index, other_index] = heading_to_local_velocity(other_state.speed, rel_yaw)
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
        history.past_short_term_ref_points.add(ref_norm)
        history.past_left_boundary.add(left_rel / history.normalizers.pos)
        history.past_right_boundary.add(right_rel / history.normalizers.pos)
        history.past_vertices.add(vertices_rel / history.normalizers.pos)
        return vel_rel

    def _update_action_buffers(self, history, last_actions: np.ndarray) -> None:
        history.past_action_acc.add(last_actions[:, 0] / history.normalizers.action_acc)
        history.past_action_steering.add(last_actions[:, 1] / history.normalizers.action_steering)

    def _update_error_terms(self, history, vel_rel_raw: np.ndarray) -> None:
        n_agents = self.registry.n_agents
        error_space = np.zeros((n_agents, 2), dtype=np.float32)
        error_vel = np.zeros((n_agents, 2), dtype=np.float32)
        for agent_index in range(n_agents):
            neighbors = self.registry.neighbors_of(agent_index)
            ego_speed = vel_rel_raw[agent_index, agent_index, 0]
            if neighbors.front is not None:
                error_space[agent_index, 0] = (
                    history.agent_s[neighbors.front] - history.agent_s[agent_index] - self.obs_cfg.desired_gap_s
                )
                error_vel[agent_index, 0] = vel_rel_raw[agent_index, neighbors.front, 0] - ego_speed
            if neighbors.rear is not None:
                error_space[agent_index, 1] = (
                    history.agent_s[agent_index] - history.agent_s[neighbors.rear] - self.obs_cfg.desired_gap_s
                )
                error_vel[agent_index, 1] = vel_rel_raw[agent_index, neighbors.rear, 0] - ego_speed
        history.error_vel = error_vel.astype(np.float32)
        history.error_space.add(error_space.astype(np.float32))

    def _build_hinge_info(self, states: List, projections: List) -> np.ndarray:
        n_agents = len(states)
        n_short = int(self.obs_cfg.n_points_short_term)
        hinge_info = np.zeros((n_agents, n_short, 5), dtype=np.float32)
        for agent_index, state in enumerate(states):
            local_ref_xy = transform_points_global_to_local(
                state.pose_map_xy,
                state.yaw_map,
                projections[agent_index].short_term_ref[:, :2],
            )
            hinge_info[agent_index, :, :2] = local_ref_xy
            hinge_info[agent_index, :, 2] = projections[agent_index].short_term_ref[:, 2]
            hinge_info[agent_index, :, 3] = 0.0
            hinge_info[agent_index, :, 4] = 1.0 if 0 < agent_index < (n_agents - 1) else 0.0
        return hinge_info

