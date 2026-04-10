import time
import time
from typing import Dict, List, Optional

import numpy as np

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_config import EnvConfig
from airsim_occt_controllers import BaseDeploymentController, LowLevelController
from airsim_occt_fleet_registry import FleetRegistry
from airsim_occt_history import ObservationHistory
from airsim_occt_map_projector import OcctMapProjector
from airsim_occt_obs_manifest import build_obs_layout
from airsim_occt_plotting import (
    plot_marl_debug_in_airsim,
    plot_agent_observation_points_in_airsim,
    plot_all_agent_observation_points_in_airsim,
    plot_mppi_debug_in_airsim,
    plot_selected_road_in_airsim,
)
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
        self.obs_core = SharedObsCore(env_cfg.obs, env_cfg.control, self.registry, projector=self.projector)
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
        self.target_front_wheel_angles = np.zeros((self.registry.n_agents,), dtype=np.float32)
        self.estimated_front_wheel_angles = np.zeros((self.registry.n_agents,), dtype=np.float32)
        self.scene_frame: Optional[SceneFrame] = None
        self._has_reset_once = False

    def reset(self):
        self.io.connect()
        if self._has_reset_once:
            if self.cfg.use_sim_pause_clock:
                self.io.pause(False)
            self.io.reset()
            time.sleep(1.0)
        if self.cfg.enable_api_control:
            self.io.enable_api(self.registry.vehicle_names)
        if self.cfg.use_sim_pause_clock:
            self.io.pause(True)
        self.last_actions[...] = 0.0
        self.target_front_wheel_angles[...] = 0.0
        self.estimated_front_wheel_angles[...] = 0.0
        for controller in self.controllers.values():
            controller.reset()
        raw_states = self.io.read_all(self.registry.vehicle_names)
        states = self._canonicalize(raw_states)
        projections = self.projector.project_all(states, [None] * self.registry.n_agents)
        self.history.reset()
        self.obs_core.update(self.history, states, projections, self.last_actions)
        self.scene_frame = SceneFrame(states=states, projections=projections, timestamp=raw_states[0].timestamp)
        self._has_reset_once = True
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
        self._apply_occt_state(command_map)
        self.io.send_all(command_map)
        if self.cfg.use_sim_pause_clock:
            self.io.advance(self.cfg.control.dt)
        self.target_front_wheel_angles[...] = self._extract_target_front_wheel_angles(actor_actions, command_map)
        self._update_estimated_front_wheel_angles()
        raw_states = self.io.read_all(self.registry.vehicle_names)
        states = self._canonicalize(raw_states)
        prev_s = [projection.s for projection in self.scene_frame.projections]
        projections = self.projector.project_all(states, prev_s)
        self.obs_core.update(self.history, states, projections, self.last_actions)
        self.scene_frame = SceneFrame(states=states, projections=projections, timestamp=raw_states[0].timestamp)
        obs = self._build_obs()
        reward = self._build_dummy_reward()
        goal_done, terminal_vehicle_names, s_max, goal_tolerance_s = self._check_goal_done()
        terminated = self._build_dummy_done(goal_done)
        truncated = self._build_dummy_done(False)
        info = self._build_info()
        info["road_s_max"] = float(s_max)
        info["goal_tolerance_s"] = float(goal_tolerance_s)
        if goal_done:
            info["done_reason"] = "goal_reached"
            info["terminal_vehicle_names"] = list(terminal_vehicle_names)
        return obs, reward, terminated, truncated, info

    def step_with_controller(self, deployment_controller: BaseDeploymentController):
        if self.scene_frame is None:
            raise RuntimeError("reset() must be called before step_with_controller()")
        obs_dict = self._build_obs()
        compute_begin = time.perf_counter()
        command_map = deployment_controller.compute_commands(
            obs_dict=obs_dict,
            scene_frame=self.scene_frame,
            vehicle_names=self.registry.vehicle_names,
        )
        controller_compute_time_sec = time.perf_counter() - compute_begin
        actor_actions = self._extract_actor_actions_from_controller(deployment_controller)
        result = self.step_low_level(command_map, actor_actions=actor_actions)
        result[4]['controller_compute_time_sec'] = float(controller_compute_time_sec)
        result[4]['controller_compute_time_ms'] = float(controller_compute_time_sec * 1000.0)
        controller_metadata = getattr(deployment_controller, 'metadata', None)
        if controller_metadata is not None:
            result[4]['controller_metadata'] = controller_metadata
        controller_debug = getattr(deployment_controller, 'last_debug_info', None)
        if controller_debug:
            result[4]['controller_debug'] = controller_debug
        actor_debug = getattr(deployment_controller, 'last_actor_debug_info', None)
        if actor_debug:
            result[4]['actor_debug'] = actor_debug
        mppi_debug = getattr(deployment_controller, 'last_mppi_debug_info', None)
        if mppi_debug:
            result[4]['mppi_debug'] = mppi_debug
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

    def render_mppi_debug_markers(
        self,
        mppi_debug_info: Dict[str, Dict],
        plot_z: float = 0.0,
        duration: float = 0.2,
        is_persistent: bool = False,
        clear_existing: bool = False,
    ) -> None:
        if not mppi_debug_info:
            return
        plot_mppi_debug_in_airsim(
            io=self.io,
            mppi_debug_info=mppi_debug_info,
            world_to_map=self.transform,
            plot_z=plot_z,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=clear_existing,
            flip_world_y=self.flip_world_y,
        )

    def render_marl_debug_markers(
        self,
        actor_debug_info: Dict[str, Dict],
        plot_z: float = -3.0,
        duration: float = 0.2,
        is_persistent: bool = False,
        clear_existing: bool = False,
    ) -> None:
        if not actor_debug_info:
            return
        plot_marl_debug_in_airsim(
            io=self.io,
            scene_frame=self.scene_frame,
            registry=self.registry,
            actor_debug_info=actor_debug_info,
            world_to_map=self.transform,
            plot_z=plot_z,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=clear_existing,
            flip_world_y=self.flip_world_y,
        )

    def _canonicalize(self, raw_states):
        states = []
        for index, raw_state in enumerate(raw_states):
            state = self.transformer.convert(raw_state, agent_index=index)
            state.steering_feedback = float(self.estimated_front_wheel_angles[index])
            state.steering_target_rad = float(self.target_front_wheel_angles[index])
            state.last_action_acc = float(self.last_actions[index, 0])
            state.last_action_steer = float(self.last_actions[index, 1])
            states.append(state)
        return states

    def _extract_target_front_wheel_angles(self, actor_actions, command_map) -> np.ndarray:
        target = np.zeros((self.registry.n_agents,), dtype=np.float32)
        estimation_max_angle = getattr(self.cfg.control, "steering_estimation_max_angle", None)
        if estimation_max_angle is None:
            estimation_max_angle = self.cfg.control.max_steering_angle
        if actor_actions is not None:
            actor_actions_np = np.asarray(actor_actions, dtype=np.float32)
            if actor_actions_np.shape == (self.registry.n_agents, 2):
                target[...] = np.clip(
                    actor_actions_np[:, 1],
                    -float(estimation_max_angle),
                    float(estimation_max_angle),
                )
        steering_sign = float(getattr(self.cfg.control, "steering_command_sign", -1.0))
        for agent_index, vehicle_name in self.registry.ordered_pairs():
            if abs(float(target[agent_index])) > 1e-8:
                continue
            command = command_map.get(vehicle_name)
            if command is None:
                continue
            normalized_steering = float(command.steering)
            target[agent_index] = float(
                np.clip(
                    normalized_steering * float(estimation_max_angle) / steering_sign,
                    -float(estimation_max_angle),
                    float(estimation_max_angle),
                )
            )
        return target

    def _update_estimated_front_wheel_angles(self) -> None:
        dt = float(self.cfg.control.dt)
        tau = float(getattr(self.cfg.control, "steering_estimation_time_constant", 0.12))
        max_rate = float(getattr(self.cfg.control, "steering_estimation_max_rate", 1.5707963268))
        estimation_max_angle = getattr(self.cfg.control, "steering_estimation_max_angle", None)
        if estimation_max_angle is None:
            estimation_max_angle = self.cfg.control.max_steering_angle
        tau = max(tau, 1e-6)
        delta_dot_cmd = (self.target_front_wheel_angles - self.estimated_front_wheel_angles) / tau
        delta_dot = np.clip(delta_dot_cmd, -max_rate, max_rate)
        self.estimated_front_wheel_angles[...] = np.clip(
            self.estimated_front_wheel_angles + delta_dot * dt,
            -float(estimation_max_angle),
            float(estimation_max_angle),
        ).astype(np.float32)

    def _extract_actor_actions_from_controller(self, deployment_controller: BaseDeploymentController) -> np.ndarray:
        actor_actions = np.zeros((self.registry.n_agents, 2), dtype=np.float32)
        actor_action_map = getattr(deployment_controller, "last_actor_action_map", {}) or {}
        for agent_index, vehicle_name in self.registry.ordered_pairs():
            actor_action = actor_action_map.get(vehicle_name)
            if actor_action is None:
                continue
            actor_actions[agent_index, 0] = float(actor_action.acceleration_mps2)
            actor_actions[agent_index, 1] = float(actor_action.front_wheel_angle_rad)
        return actor_actions

    def _apply_occt_state(self, command_map) -> None:
        occt_state = self._build_occt_state()
        for command in command_map.values():
            command.occt_state = list(occt_state)

    def _build_occt_state(self) -> List[bool]:
        n_agents = self.registry.n_agents
        if n_agents <= 0:
            return []
        if n_agents == 1:
            return [True]
        if self.scene_frame is None:
            return [True] + [False] * max(n_agents - 2, 0) + [True]
        middle_status = self.history.agent_hinge_status.latest()[1:-1].astype(bool).tolist()
        return [True] + middle_status + [True]

    def _build_obs(self) -> Dict[str, np.ndarray]:
        obs = {}
        for agent_index, vehicle_name in self.registry.ordered_pairs():
            current_state = self.scene_frame.states[agent_index] if self.scene_frame is not None else None
            obs_vec = self.obs_core.encode(agent_index, self.history, state=current_state)
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

    def _check_goal_done(self) -> tuple[bool, List[str], float, float]:
        if self.scene_frame is None:
            return False, [], float(self.projector.s_max), 0.75
        s_max = float(self.projector.s_max)
        goal_tolerance_s = 0.75
        terminal_vehicle_names = [
            vehicle_name
            for index, vehicle_name in self.registry.ordered_pairs()
            if float(self.scene_frame.projections[index].s) >= (s_max - goal_tolerance_s)
        ]
        return bool(terminal_vehicle_names), terminal_vehicle_names, s_max, goal_tolerance_s

    def _build_info(self) -> Dict:
        if self.scene_frame is None:
            return {}
        agent_hinge_status = self.history.agent_hinge_status.latest().astype(bool).tolist()
        hinge_ready_status = self.history.hinge_status.astype(bool).tolist()
        occt_state = self._build_occt_state()
        target_agent_s = self.obs_core._get_dynamic_target_arc_positions(self.history.agent_s)
        hinge_target_point_map = self.history.hinge_short_term[:, 0, :2]
        hinge_target_speed = np.linalg.norm(self.history.hinge_short_term[:, 0, 2:4], axis=-1)
        hinge_distance = np.linalg.norm(
            np.asarray([state.pose_map_xy for state in self.scene_frame.states], dtype=np.float32)
            - np.asarray(hinge_target_point_map, dtype=np.float32),
            axis=-1,
        )
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
            "closest_center_map": {
                vehicle_name: self.scene_frame.projections[index].closest_center_xy.copy()
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "distance_to_ref": {
                vehicle_name: float(self.scene_frame.projections[index].dist_to_ref)
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "target_agent_s": {
                vehicle_name: float(target_agent_s[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "hinge_target_speed": {
                vehicle_name: float(hinge_target_speed[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "hinge_distance": {
                vehicle_name: float(hinge_distance[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "road_s_max": float(self.projector.s_max),
            "occt_state": {
                vehicle_name: bool(occt_state[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "agent_hinge_status": {
                vehicle_name: bool(agent_hinge_status[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
            "hinge_ready_status": {
                vehicle_name: bool(hinge_ready_status[index])
                for index, vehicle_name in self.registry.ordered_pairs()
            },
        }

    def _get_lane_width(self) -> float:
        if hasattr(self.road, "get_lane_width"):
            return float(self.road.get_lane_width("mean"))
        return 6.0
