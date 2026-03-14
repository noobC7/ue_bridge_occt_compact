from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VehicleConfig:
    vehicle_name: str
    length: float
    width: float
    l_f: float
    l_r: float
    role: str = "follower"


@dataclass
class ObsConfig:
    n_agents: int = 1
    n_points_short_term: int = 4
    n_points_nearing_boundary: int = 5
    n_stored_steps: int = 5
    n_observed_steps: int = 5
    sample_interval: float = 2.0
    boundary_offset: float = -2.0
    neighbor_mode: str = "adjacent"
    task_class: str = "occt_platoon"
    mask_ref_v: bool = False
    is_add_noise: bool = False
    desired_gap_s: float = 6.0
    include_hinge_info: bool = True
    obs_relative_velocity_scale: Optional[float] = None
    obs_relative_acceleration_scale: Optional[float] = None


@dataclass
class ControlConfig:
    dt: float = 0.05
    max_speed: float = 5.0
    max_acceleration: float = 3.0
    max_steering_angle: float = 0.6108652382
    max_steering_rate: float = 0.6108652382
    speed_pid_kp: float = 1.0
    speed_pid_ki: float = 0.0
    speed_pid_kd: float = 0.0
    steering_lowpass_alpha: float = 1.0
    throttle_limit: float = 1.0
    brake_limit: float = 1.0
    steering_command_sign: float = -1.0
    stanley_heading_gain: float = 1.0
    stanley_cross_track_gain: float = 1.2
    stanley_feedforward_gain: float = 0.35
    stanley_soft_speed: float = 0.5


@dataclass
class AlignmentConfig:
    transform_file: Optional[str] = None
    anchor_object_names: List[str] = field(default_factory=list)
    allow_reflection: bool = True
    flip_world_y: bool = False


@dataclass
class MapConfig:
    cr_map_dir: str = ""
    sample_gap: float = 1.0
    min_lane_width: float = 2.1
    min_lane_len: float = 70.0
    max_ref_v: float = 20.0 / 3.6
    is_constant_ref_v: bool = True


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 41451
    vehicle_configs: List[VehicleConfig] = field(default_factory=list)
    obs: ObsConfig = field(default_factory=ObsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    map_cfg: MapConfig = field(default_factory=MapConfig)
    enable_api_control: bool = True
    use_sim_pause_clock: bool = True
    reset_on_close: bool = True
    road_env_index: int = 0

    def __post_init__(self) -> None:
        if self.vehicle_configs:
            self.obs.n_agents = len(self.vehicle_configs)
