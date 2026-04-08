from dataclasses import dataclass
from typing import Dict, List


FULL_KEY_ORDER = [
    "self_vel",
    "self_speed",
    "self_steering",
    "self_acc",
    "self_ref_velocity",
    "self_ref_points",
    "self_left_boundary_distance",
    "self_right_boundary_distance",
    "self_hinge_status",
    "self_hinge_preview_info",
    "self_hinge_past_info",
    "self_hinge_error_vel",
    "self_distance_to_ref",
    "self_distance_to_left_boundary",
    "self_distance_to_right_boundary",
    "self_platoon_error_vel",
    "self_platoon_error_space",
    "others_pos",
    "others_rot",
    "others_relative_longitudinal_velocity",
    "others_relative_acceleration",
    "others_distance",
]

ACTOR_EXCLUDED_KEYS = [
    "self_hinge_status",
    "self_hinge_past_info",
]


@dataclass(frozen=True)
class ObsLayout:
    full_block_dims: Dict[str, int]
    actor_block_order: List[str]

    @property
    def full_dim(self) -> int:
        return int(sum(self.full_block_dims.values()))

    @property
    def total_dim(self) -> int:
        return int(sum(self.full_block_dims[key] for key in self.actor_block_order))


def build_obs_layout(obs_cfg) -> ObsLayout:
    n_short = int(obs_cfg.n_points_short_term)
    n_near = int(obs_cfg.n_nearing_agents_observed)
    full_block_dims = {
        "self_vel": 1,
        "self_speed": 1,
        "self_steering": 1,
        "self_acc": 1,
        "self_ref_velocity": n_short,
        "self_ref_points": n_short * 2,
        "self_left_boundary_distance": n_short,
        "self_right_boundary_distance": n_short,
        "self_hinge_status": 1,
        "self_hinge_preview_info": n_short * 4,
        "self_hinge_past_info": 4,
        "self_hinge_error_vel": 1,
        "self_distance_to_ref": 1,
        "self_distance_to_left_boundary": 1,
        "self_distance_to_right_boundary": 1,
        "self_platoon_error_vel": 2,
        "self_platoon_error_space": 2,
        "others_pos": n_near * 2,
        "others_rot": n_near,
        "others_relative_longitudinal_velocity": n_near,
        "others_relative_acceleration": n_near,
        "others_distance": n_near,
    }
    actor_block_order = [
        key for key in FULL_KEY_ORDER if key not in set(ACTOR_EXCLUDED_KEYS)
    ]
    return ObsLayout(
        full_block_dims=full_block_dims,
        actor_block_order=actor_block_order,
    )
