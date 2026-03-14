from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CircularArray:
    data: np.ndarray
    valid_size: int = 0
    write_index: int = 0

    def reset(self) -> None:
        self.data[...] = 0
        self.valid_size = 0
        self.write_index = 0

    def add(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=self.data.dtype)
        if arr.shape != self.data.shape[1:]:
            raise ValueError(f"shape mismatch, expected {self.data.shape[1:]}, got {arr.shape}")
        self.data[self.write_index] = arr
        self.write_index = (self.write_index + 1) % self.data.shape[0]
        self.valid_size = min(self.valid_size + 1, self.data.shape[0])

    def latest(self, n: int = 1) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        if self.valid_size == 0 or n > self.valid_size:
            return np.zeros(self.data.shape[1:], dtype=self.data.dtype)
        index = (self.write_index - n) % self.data.shape[0]
        return self.data[index].copy()


class ObservationHistory:
    def __init__(self, n_agents: int, obs_cfg, normalizers) -> None:
        self.n_agents = n_agents
        self.obs_cfg = obs_cfg
        self.normalizers = normalizers
        self.reset()

    def reset(self) -> None:
        n_store = int(self.obs_cfg.n_stored_steps)
        n_agents = int(self.n_agents)
        n_short = int(self.obs_cfg.n_points_short_term)
        n_boundary = int(self.obs_cfg.n_points_nearing_boundary)

        self.past_pos = CircularArray(np.zeros((n_store, n_agents, n_agents, 2), dtype=np.float32))
        self.past_rot = CircularArray(np.zeros((n_store, n_agents, n_agents), dtype=np.float32))
        self.past_vel = CircularArray(np.zeros((n_store, n_agents, n_agents, 2), dtype=np.float32))
        self.past_steering = CircularArray(np.zeros((n_store, n_agents), dtype=np.float32))
        self.past_short_term_ref_points = CircularArray(
            np.zeros((n_store, n_agents, n_agents, n_short, 3), dtype=np.float32)
        )
        self.past_left_boundary = CircularArray(
            np.zeros((n_store, n_agents, n_agents, n_boundary, 2), dtype=np.float32)
        )
        self.past_right_boundary = CircularArray(
            np.zeros((n_store, n_agents, n_agents, n_boundary, 2), dtype=np.float32)
        )
        self.past_vertices = CircularArray(
            np.zeros((n_store, n_agents, n_agents, 4, 2), dtype=np.float32)
        )
        self.past_action_acc = CircularArray(np.zeros((n_store, n_agents), dtype=np.float32))
        self.past_action_steering = CircularArray(np.zeros((n_store, n_agents), dtype=np.float32))
        self.past_distance_to_agents = CircularArray(
            np.zeros((n_store, n_agents, n_agents), dtype=np.float32)
        )
        self.past_distance_to_ref_path = CircularArray(np.zeros((n_store, n_agents), dtype=np.float32))
        self.past_distance_to_left_boundary = CircularArray(
            np.zeros((n_store, n_agents), dtype=np.float32)
        )
        self.past_distance_to_right_boundary = CircularArray(
            np.zeros((n_store, n_agents), dtype=np.float32)
        )
        self.error_space = CircularArray(np.zeros((n_store, n_agents, 2), dtype=np.float32))
        self.error_vel = np.zeros((n_agents, 2), dtype=np.float32)
        self.agent_s = np.zeros((n_agents,), dtype=np.float32)
        self.agent_target_hinge_short_term = np.zeros((n_agents, n_short, 5), dtype=np.float32)

    def set_agent_s(self, s_values: np.ndarray) -> None:
        values = np.asarray(s_values, dtype=np.float32)
        if values.shape != (self.n_agents,):
            raise ValueError(f"agent_s shape mismatch, expected {(self.n_agents,)}, got {values.shape}")
        self.agent_s = values.copy()

    def last_n_longitudinal_velocities(self, ego_index: int, other_index: int, n: int) -> np.ndarray:
        seq = []
        for offset in range(n):
            seq.append(self.past_vel.latest(offset + 1)[ego_index, other_index, 0])
        return np.asarray(seq, dtype=np.float32)

    def latest_neighbor_distance(self, ego_index: int, other_index: int) -> float:
        return float(self.past_distance_to_agents.latest()[ego_index, other_index])

    def latest_self_ref(self, agent_index: int) -> np.ndarray:
        return self.past_short_term_ref_points.latest()[agent_index, agent_index]

    def latest_self_boundary(self, agent_index: int) -> Tuple[np.ndarray, np.ndarray]:
        left = self.past_left_boundary.latest()[agent_index, agent_index]
        right = self.past_right_boundary.latest()[agent_index, agent_index]
        return left, right

