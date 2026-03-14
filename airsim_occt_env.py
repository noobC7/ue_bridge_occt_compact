from typing import Dict, Optional

import numpy as np

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_config import EnvConfig
from airsim_occt_controllers import BaseDeploymentController, LowLevelController
from airsim_occt_fleet_registry import FleetRegistry
from airsim_occt_history import ObservationHistory
from airsim_occt_map_projector import OcctMapProjector
from airsim_occt_obs_manifest import build_obs_layout
from airsim_occt_plotting import plot_agent_observation_points_in_airsim, plot_all_agent_observation_points_in_airsim, plot_selected_road_in_airsim
from airsim_occt_schema import ActorAction, SceneFrame, Transform2D
from airsim_occt_shared_obs_core import SharedObsCore
from airsim_occt_transform import WorldToMapTransformer


class AirSimOcctMARLEnv:
    def __init__(self, env_cfg: EnvConfig, road, transform: Optional[Transform2D] = None) -> None:
        self.cfg = env_cfg
        self.road = road
        self.registry = FleetRegistry(env_cfg.vehicle_configs)
        self.io = AirSimIO(env_cfg)
        self.transform = transform or Transform2D(
            mat=np.eye(2, dtype=np.float32),
            bias=np.zeros((2,), dtype=np.float32),
        )
        self.flip_world_y = bool(env_cfg.alignment.flip_world_y)
        self.transformer = WorldToMapTransformer(self.transform, flip_world_y=self.flip_world_y)
        self.projector = OcctMapProjector(
            road=road,
            fleet_registry=self.registry,
            obs_cfg=env_cfg.obs,
            road_env_index=env_cfg.road_env_index,
        )
        first_vehicle = self.registry.config_of(0)
        agent_length = first_vehicle.length
        agent_width = first_vehicle.width
        self.obs_core = SharedObsCore(env_cfg.obs, env_cfg.control, self.registry)
        self.normalizers = self.obs_core.build_normalizers(
            agent_length=agent_length,
            agent_width=agent_width,
            lane_width=float(self._get_lane_width()),
        )
        self.history = ObservationHistory(
            n_agents=self.registry.n_agents,
            obs_cfg=env_cfg.obs,
            normalizers=self.normalizers,
        )
        self.layout = build_obs_layout(env_cfg.obs)
        self.controllers = {
            vehicle_name: LowLevelController(env_cfg.control)
            for vehicle_name in self.registry.vehicle_names
        }
        self.last_actions = np.zeros((self.registry.n_agents, 2), dtype=np.float32)
        self.scene_frame: Optional[SceneFrame] = None

    def reset(self):
        self.io.connect()
        if self.cfg.enable_api_control:
            self.io.enable_api(self.registry.vehicle_names)
        if self.cfg.use_sim_pause_clock:
            self.io.pause(True)
        self.last_actions[...] = 0.0
        for controller in self.controllers.values():
            controller.reset()
        raw_states = self.io.read_all(self.registry.vehicle_names)
        states = self._canonicalize(raw_states)
        projections = self.projector.project_all(states, [None] * self.registry.n_agents)
        self.history.reset()
        self.obs_core.update(self.history, states, projections, self.last_actions)
        self.scene_frame = SceneFrame(states=states, projections=projections, timestamp=raw_states[0].timestamp)
        return self._build_obs(), {"vehicle_names": self.registry.vehicle_names}

    def step(self, action_dict: Dict[str, ActorAction]):
        if self.scene_frame is None:
            raise RuntimeError("reset() must be called before step()")
        command_map = {}
        for agent_index, vehicle_name in self.registry.ordered_pairs():
            action = action_dict[vehicle_name]
            state = self.scene_frame.states[agent_index]
            command_map[vehicle_name] = self.controllers[vehicle_name].step(action, state)
            self.last_actions[agent_index] = np.asarray(
                [action.acceleration_mps2, action.front_wheel_angle_rad],
                dtype=np.float32,
            )
        return self.step_low_level(command_map, actor_actions=self.last_actions.copy())

    def step_low_level(self, command_map, actor_actions=None):
        if self.scene_frame is None:
            raise RuntimeError("reset() must be called before step_low_level()")
        if actor_actions is None:
            self.last_actions[...] = 0.0
        else:
            self.last_actions[...] = np.asarray(actor_actions, dtype=np.float32)
        self.io.send_all(command_map)
        if self.cfg.use_sim_pause_clock:
            self.io.advance(self.cfg.control.dt)
        raw_states = self.io.read_all(self.registry.vehicle_names)
        states = self._canonicalize(raw_states)
        prev_s = [projection.s for projection in self.scene_frame.projections]
        projections = self.projector.project_all(states, prev_s)
        self.obs_core.update(self.history, states, projections, self.last_actions)
        self.scene_frame = SceneFrame(states=states, projections=projections, timestamp=raw_states[0].timestamp)
        obs = self._build_obs()
        reward = self._build_dummy_reward()
        terminated = self._build_dummy_done(False)
        truncated = self._build_dummy_done(False)
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def step_with_controller(self, deployment_controller: BaseDeploymentController):
        if self.scene_frame is None:
            raise RuntimeError("reset() must be called before step_with_controller()")
        obs_dict = self._build_obs()
        command_map = deployment_controller.compute_commands(
            obs_dict=obs_dict,
            scene_frame=self.scene_frame,
            vehicle_names=self.registry.vehicle_names,
        )
        result = self.step_low_level(command_map, actor_actions=None)
        controller_debug = getattr(deployment_controller, 'last_debug_info', None)
        if controller_debug:
            result[4]['controller_debug'] = controller_debug
        actor_debug = getattr(deployment_controller, 'last_actor_debug_info', None)
        if actor_debug:
            result[4]['actor_debug'] = actor_debug
        return result

    def close(self) -> None:
        if self.cfg.reset_on_close:
            try:
                self.io.reset()
            except Exception as exc:
                print(f"[ENV_CLOSE] reset failed: {exc}")
        if self.cfg.use_sim_pause_clock:
            self.io.pause(False)
        if self.cfg.enable_api_control:
            self.io.disable_api(self.registry.vehicle_names)

    def plot_selected_road(
        self,
        plot_z: float = 0.0,
        thickness: float = 3.0,
        center_dash_stride: int = 3,
        center_gap_stride: int = 2,
        duration: float = -1.0,
        is_persistent: bool = True,
        clear_existing: bool = False,
        start_point_size: float = 15.0,
    ) -> None:
        plot_selected_road_in_airsim(
            io=self.io,
            road=self.road,
            world_to_map=self.transform,
            road_env_index=self.cfg.road_env_index,
            plot_z=plot_z,
            thickness=thickness,
            center_dash_stride=center_dash_stride,
            center_gap_stride=center_gap_stride,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=clear_existing,
            start_point_size=start_point_size,
            flip_world_y=self.flip_world_y,
        )

    def plot_agent_observation_points(
        self,
        agent_index: int = 0,
        plot_z: float = 0.0,
        point_size: float = 12.0,
        duration: float = -1.0,
        is_persistent: bool = True,
        clear_existing: bool = False,
    ) -> None:
        if self.scene_frame is None:
            raise RuntimeError('reset() must be called before plotting agent observation points')
        if agent_index < 0 or agent_index >= self.registry.n_agents:
            raise IndexError(f'agent_index {agent_index} out of range for n_agents={self.registry.n_agents}')
        plot_agent_observation_points_in_airsim(
            io=self.io,
            projection=self.scene_frame.projections[agent_index],
            world_to_map=self.transform,
            plot_z=plot_z,
            point_size=point_size,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=clear_existing,
            flip_world_y=self.flip_world_y,
        )

    def plot_all_agent_observation_points(
        self,
        plot_z: float = 0.0,
        point_size: float = 12.0,
        duration: float = -1.0,
        is_persistent: bool = True,
        clear_existing: bool = False,
        skip_terminal_agents: bool = True,
    ) -> None:
        if self.scene_frame is None:
            raise RuntimeError('reset() must be called before plotting all agent observation points')
        skip_agent_indices = []
        if skip_terminal_agents and self.registry.n_agents > 1:
            skip_agent_indices = [0, self.registry.n_agents - 1]
        plot_all_agent_observation_points_in_airsim(
            io=self.io,
            projections=self.scene_frame.projections,
            world_to_map=self.transform,
            plot_z=plot_z,
            point_size=point_size,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=clear_existing,
            flip_world_y=self.flip_world_y,
            skip_agent_indices=skip_agent_indices,
        )

    def render_debug_markers(
        self,
        plot_road: bool = True,
        plot_observation_points: bool = True,
        plot_z: float = 0.0,
        road_thickness: float = 3.0,
        point_size: float = 12.0,
        duration: float = -1.0,
        is_persistent: bool = True,
        clear_existing: bool = True,
    ) -> None:
        if plot_road:
            self.plot_selected_road(
                plot_z=plot_z,
                thickness=road_thickness,
                duration=duration,
                is_persistent=is_persistent,
                clear_existing=clear_existing,
            )
            clear_existing = False
        if plot_observation_points:
            self.plot_all_agent_observation_points(
                plot_z=plot_z,
                point_size=point_size,
                duration=duration,
                is_persistent=is_persistent,
                clear_existing=clear_existing,
            )

    def _canonicalize(self, raw_states):
        states = []
        for index, raw_state in enumerate(raw_states):
            state = self.transformer.convert(raw_state, agent_index=index)
            state.steering_feedback = float(self.last_actions[index, 1])
            state.last_action_acc = float(self.last_actions[index, 0])
            state.last_action_steer = float(self.last_actions[index, 1])
            states.append(state)
        return states

    def _build_obs(self) -> Dict[str, np.ndarray]:
        obs = {}
        for agent_index, vehicle_name in self.registry.ordered_pairs():
            obs_vec = self.obs_core.encode(agent_index, self.history)
            if obs_vec.shape[0] != self.layout.total_dim:
                raise ValueError(
                    f"observation dim mismatch for {vehicle_name}, expected {self.layout.total_dim}, got {obs_vec.shape[0]}"
                )
            obs[vehicle_name] = obs_vec
        return obs

    def _build_dummy_reward(self) -> Dict[str, float]:
        return {vehicle_name: 0.0 for vehicle_name in self.registry.vehicle_names}

    def _build_dummy_done(self, value: bool) -> Dict[str, bool]:
        return {vehicle_name: value for vehicle_name in self.registry.vehicle_names}

    def _build_info(self) -> Dict:
        if self.scene_frame is None:
            return {}
        return {
            "s": {vehicle_name: float(self.scene_frame.projections[index].s) for index, vehicle_name in self.registry.ordered_pairs()},
            "pose_map_xy": {
                vehicle_name: self.scene_frame.states[index].pose_map_xy.copy()
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "projection_point_map": {
                vehicle_name: self.scene_frame.projections[index].projection_point_map.copy()
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "projection_point_mode": {
                vehicle_name: self.scene_frame.projections[index].projection_point_mode
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "yaw_map": {
                vehicle_name: float(self.scene_frame.states[index].yaw_map)
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "speed": {
                vehicle_name: float(self.scene_frame.states[index].speed)
                for index, vehicle_name in self.registry.ordered_pairs()
            },
        }

    def _get_lane_width(self) -> float:
        if hasattr(self.road, "get_lane_width"):
            return float(self.road.get_lane_width("mean"))
        return 6.0
