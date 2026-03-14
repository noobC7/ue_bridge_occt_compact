from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ObsLayout:
    self_block_dims: Dict[str, int]
    neighbor_block_dims: Dict[str, int]

    @property
    def self_dim(self) -> int:
        return int(sum(self.self_block_dims.values()))

    @property
    def neighbor_dim(self) -> int:
        return int(sum(self.neighbor_block_dims.values()))

    @property
    def total_dim(self) -> int:
        return int(self.self_dim + 2 * self.neighbor_dim)


SELF_BLOCK_ORDER = [
    "self_vel_local",
    "self_speed",
    "self_vel_longitudinal",
    "self_steering",
    "self_ref_velocity",
    "self_ref_points",
    "self_left_boundary_distance",
    "self_right_boundary_distance",
    "self_hinge_info",
    "self_distance_to_ref",
    "self_distance_to_left_boundary",
    "self_distance_to_right_boundary",
    "self_error_vel",
    "self_error_space",
]


NEIGHBOR_BLOCK_ORDER = [
    "relative_pos",
    "relative_rot",
    "relative_longitudinal_velocity",
    "relative_acceleration_history",
    "distance",
]


def build_obs_layout(obs_cfg) -> ObsLayout:
    n_short = int(obs_cfg.n_points_short_term)
    n_observed_steps = int(obs_cfg.n_observed_steps)
    self_block_dims = {
        "self_vel_local": 1,
        "self_speed": 1,
        "self_vel_longitudinal": 1,
        "self_steering": 1,
        "self_ref_velocity": 0 if obs_cfg.mask_ref_v else n_short,
        "self_ref_points": n_short * 2,
        "self_left_boundary_distance": n_short,
        "self_right_boundary_distance": n_short,
        "self_hinge_info": n_short * 5 if obs_cfg.include_hinge_info else 0,
        "self_distance_to_ref": 1,
        "self_distance_to_left_boundary": 1,
        "self_distance_to_right_boundary": 1,
        "self_error_vel": 2,
        "self_error_space": 2,
    }
    neighbor_block_dims = {
        "relative_pos": 2,
        "relative_rot": 1,
        "relative_longitudinal_velocity": 1,
        "relative_acceleration_history": n_observed_steps,
        "distance": 1,
    }
    return ObsLayout(
        self_block_dims=self_block_dims,
        neighbor_block_dims=neighbor_block_dims,
    )

