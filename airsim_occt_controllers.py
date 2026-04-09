from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from airsim_occt_geometry import transform_points_global_to_local
from airsim_occt_schema import ActorAction, LowLevelCommand


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PIDLongitudinalController:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, v_now: float, v_target: float, dt: float) -> tuple[float, float]:
        error = float(v_target - v_now)
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        command = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        throttle = float(np.clip(command, 0.0, 1.0))
        brake = float(np.clip(-command, 0.0, 1.0))
        return throttle, brake


class SteeringAdapter:
    def __init__(self, max_steering_angle: float, alpha: float = 1.0) -> None:
        self.max_steering_angle = max_steering_angle
        self.alpha = alpha
        self.prev_steer = 0.0

    def reset(self) -> None:
        self.prev_steer = 0.0

    def step(self, delta_des: float) -> float:
        raw = float(np.clip(delta_des / max(self.max_steering_angle, 1e-6), -1.0, 1.0))
        steer = self.alpha * raw + (1.0 - self.alpha) * self.prev_steer
        self.prev_steer = steer
        return float(np.clip(steer, -1.0, 1.0))


class LowLevelController:
    def __init__(self, control_cfg) -> None:
        self.cfg = control_cfg
        self.steering = SteeringAdapter(
            max_steering_angle=control_cfg.max_steering_angle,
            alpha=control_cfg.steering_lowpass_alpha,
        )

    def reset(self) -> None:
        self.steering.reset()

    def step(self, actor_action, state) -> LowLevelCommand:
        heading_vec = np.asarray(
            [np.cos(state.yaw_map), np.sin(state.yaw_map)],
            dtype=np.float32,
        )
        measured_acc_long = float(np.dot(np.asarray(state.acc_map_xy, dtype=np.float32), heading_vec))
        desired_acc = float(np.clip(actor_action.acceleration_mps2, -self.cfg.max_acceleration, self.cfg.max_acceleration))
        acc_error = desired_acc - measured_acc_long

        if desired_acc >= 0.0:
            throttle = (
                self.cfg.throttle_deadzone
                + self.cfg.accel_throttle_gain * desired_acc
                + self.cfg.accel_feedback_gain * acc_error
            )
            if state.speed < self.cfg.launch_speed_threshold and desired_acc > self.cfg.launch_accel_threshold:
                throttle = max(throttle, self.cfg.launch_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = (
                self.cfg.brake_deadzone
                + self.cfg.accel_brake_gain * (-desired_acc)
                + self.cfg.accel_feedback_gain * max(-acc_error, 0.0)
            )

        steering = self.steering.step(self.cfg.steering_command_sign * actor_action.front_wheel_angle_rad)
        throttle = float(np.clip(throttle, 0.0, self.cfg.throttle_limit))
        brake = float(np.clip(brake, 0.0, self.cfg.brake_limit))
        return LowLevelCommand(throttle=throttle, brake=brake, steering=steering)


class BaseDeploymentController:
    def reset(self) -> None:
        return None

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        raise NotImplementedError


class ConstantLowLevelController(BaseDeploymentController):
    def __init__(self, throttle: float = 0.15, steering: float = 0.0, brake: float = 0.0) -> None:
        self.throttle = float(throttle)
        self.steering = float(steering)
        self.brake = float(brake)

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        return {
            vehicle_name: LowLevelCommand(
                throttle=self.throttle,
                brake=self.brake,
                steering=self.steering,
            )
            for vehicle_name in vehicle_names
        }


class SharedCheckpointActor:
    def __init__(
        self,
        checkpoint_path: str,
        action_scale: np.ndarray,
        device: str = "cpu",
    ) -> None:
        import re
        import torch
        import torch.nn as nn

        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_state_dict = checkpoint["policy_state_dict"]
        layer_pattern = re.compile(r"^(?P<prefix>.+\.params)\.(?P<index>\d+)\.(?P<kind>weight|bias)$")
        grouped_layers = {}
        for key, value in policy_state_dict.items():
            matched = layer_pattern.match(key)
            if matched is None:
                continue
            prefix = matched.group("prefix")
            index = int(matched.group("index"))
            grouped_layers.setdefault(prefix, {})[(index, matched.group("kind"))] = value
        if not grouped_layers:
            raise ValueError(
                "Unsupported checkpoint format: no shared MLP parameter block ending with '.params' was found."
            )

        supported_prefix = max(
            grouped_layers.keys(),
            key=lambda prefix: len([index for index, kind in grouped_layers[prefix] if kind == "weight"]),
        )
        layer_entries = grouped_layers[supported_prefix]
        layer_indices = sorted({index for index, kind in layer_entries if kind == "weight"})
        weights = [layer_entries[(index, "weight")] for index in layer_indices]
        biases = [layer_entries[(index, "bias")] for index in layer_indices]

        obs_dim = int(weights[0].shape[1])
        output_dim = int(weights[-1].shape[0])
        if output_dim % 2 != 0:
            raise ValueError(f"Actor output dim must be even, got {output_dim}")
        self.obs_dim = obs_dim
        self.action_dim = output_dim // 2
        self.device = device
        self.torch = torch
        modules = []
        for layer_pos, weight in enumerate(weights):
            in_features = int(weight.shape[1])
            out_features = int(weight.shape[0])
            modules.append(nn.Linear(in_features, out_features))
            if layer_pos != len(weights) - 1:
                modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules).to(device)
        with torch.no_grad():
            linear_modules = [module for module in self.model if isinstance(module, nn.Linear)]
            for module, weight, bias in zip(linear_modules, weights, biases):
                module.weight.copy_(weight.to(device))
                module.bias.copy_(bias.to(device))
        self.model.eval()
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
        self.checkpoint_path = checkpoint_path
        self.iteration = checkpoint.get("iteration")
        self.total_frames = checkpoint.get("total_frames")
        self.parameter_prefix = supported_prefix

    def act(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs_batch, dtype=np.float32)
        if obs_np.ndim != 2:
            raise ValueError(f"obs_batch must have shape [N, obs_dim], got {obs_np.shape}")
        if obs_np.shape[1] != self.obs_dim:
            raise ValueError(f"Checkpoint actor expects obs_dim={self.obs_dim}, got {obs_np.shape[1]}")
        with self.torch.no_grad():
            obs_tensor = self.torch.as_tensor(obs_np, dtype=self.torch.float32, device=self.device)
            params = self.model(obs_tensor)
            loc = params[:, : self.action_dim]
            actions = self.torch.tanh(loc) * self.action_scale
        return actions.detach().cpu().numpy().astype(np.float32)


class ActorDeploymentController(BaseDeploymentController):
    def __init__(self, actor_fn, control_cfg, vehicle_names: Iterable[str]) -> None:
        self.actor_fn = actor_fn
        self.vehicle_names: List[str] = list(vehicle_names)
        self.low_level = {
            vehicle_name: LowLevelController(control_cfg)
            for vehicle_name in self.vehicle_names
        }
        self.expected_obs_dim: Optional[int] = None
        self.metadata: Dict = {}
        self.last_actor_debug_info: Dict[str, Dict] = {}
        self.last_actor_action_map: Dict[str, ActorAction] = {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        control_cfg,
        vehicle_names: Iterable[str],
        device: str = "cpu",
    ):
        ordered_names = list(vehicle_names)
        actor = SharedCheckpointActor(
            checkpoint_path=checkpoint_path,
            action_scale=np.asarray(
                [control_cfg.max_acceleration, control_cfg.max_steering_angle],
                dtype=np.float32,
            ),
            device=device,
        )

        def actor_fn(obs_dict):
            obs_batch = np.stack([np.asarray(obs_dict[name], dtype=np.float32) for name in ordered_names], axis=0)
            actions = actor.act(obs_batch)
            return {
                name: ActorAction(
                    acceleration_mps2=float(actions[index, 0]),
                    front_wheel_angle_rad=float(actions[index, 1]),
                )
                for index, name in enumerate(ordered_names)
            }

        controller = cls(actor_fn=actor_fn, control_cfg=control_cfg, vehicle_names=ordered_names)
        controller.expected_obs_dim = actor.obs_dim
        controller.metadata = {
            "checkpoint_path": checkpoint_path,
            "iteration": actor.iteration,
            "total_frames": actor.total_frames,
            "obs_dim": actor.obs_dim,
            "action_dim": actor.action_dim,
            "parameter_prefix": actor.parameter_prefix,
        }
        return controller

    def reset(self) -> None:
        for controller in self.low_level.values():
            controller.reset()

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        actor_outputs = self.actor_fn(obs_dict)
        commands = {}
        ordered_names = list(vehicle_names)
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}
        for agent_index, vehicle_name in enumerate(ordered_names):
            actor_action = self._coerce_actor_action(actor_outputs[vehicle_name])
            self.last_actor_action_map[vehicle_name] = actor_action
            state = scene_frame.states[agent_index]
            heading_vec = np.asarray(
                [np.cos(state.yaw_map), np.sin(state.yaw_map)],
                dtype=np.float32,
            )
            measured_acc_long = float(np.dot(np.asarray(state.acc_map_xy, dtype=np.float32), heading_vec))
            command = self.low_level[vehicle_name].step(actor_action, state)
            commands[vehicle_name] = command
            self.last_actor_debug_info[vehicle_name] = {
                "acceleration_mps2": float(actor_action.acceleration_mps2),
                "front_wheel_angle_rad": float(actor_action.front_wheel_angle_rad),
                "measured_acc_long": measured_acc_long,
                "throttle_cmd": float(command.throttle),
                "brake_cmd": float(command.brake),
                "steering_cmd": float(command.steering),
                "current_speed": float(state.speed),
            }
        return commands

    def _coerce_actor_action(self, value) -> ActorAction:
        if isinstance(value, ActorAction):
            return value
        if isinstance(value, dict):
            return ActorAction(
                acceleration_mps2=float(value["acceleration_mps2"]),
                front_wheel_angle_rad=float(value["front_wheel_angle_rad"]),
            )
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
            return ActorAction(
                acceleration_mps2=float(value[0]),
                front_wheel_angle_rad=float(value[1]),
            )
        raise TypeError(f"Unsupported actor output type: {type(value)}")


@dataclass
class ReferenceTarget:
    vehicle_name: str
    agent_index: int
    current_s: float
    target_s: float
    lookahead_distance: float
    target_point_map: np.ndarray
    closest_center_map: np.ndarray
    reference_speed: float
    reference_heading: float
    target_heading: float
    short_term_index: int




@dataclass
class TrackingDebugInfo:
    vehicle_name: str
    agent_index: int
    current_s: float
    target_s: float
    lookahead_distance: float
    short_term_index: int
    lateral_error: float
    speed_error: float
    target_local_x: float
    target_local_y: float
    target_point_map: np.ndarray
    control_point_map: np.ndarray
    closest_center_map: np.ndarray
    reference_heading: float
    target_heading: float
    heading_error: float
    heading_feedforward: float
    cross_track_term: float
    delta_des: float
    steering_cmd: float
    throttle_cmd: float
    brake_cmd: float
    reference_speed: float
    current_speed: float


class FrontRearReferencePlanner:
    def __init__(
        self,
        registry,
        sample_interval: float,
        front_lookahead_base: float = 6.0,
        front_lookahead_speed_gain: float = 0.8,
        front_lookahead_min: float = 4.0,
        front_lookahead_max: float = 14.0,
        rear_lookahead_base: float = 4.0,
        rear_lookahead_speed_gain: float = 0.3,
        rear_lookahead_min: float = 2.5,
        rear_lookahead_max: float = 8.0,
        controlled_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self.registry = registry
        self.sample_interval = float(sample_interval)
        self.front_lookahead_base = float(front_lookahead_base)
        self.front_lookahead_speed_gain = float(front_lookahead_speed_gain)
        self.front_lookahead_min = float(front_lookahead_min)
        self.front_lookahead_max = float(front_lookahead_max)
        self.rear_lookahead_base = float(rear_lookahead_base)
        self.rear_lookahead_speed_gain = float(rear_lookahead_speed_gain)
        self.rear_lookahead_min = float(rear_lookahead_min)
        self.rear_lookahead_max = float(rear_lookahead_max)
        self.controlled_indices = None if controlled_indices is None else list(controlled_indices)

    def plan(self, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, ReferenceTarget]:
        ordered_names = list(vehicle_names)
        controlled_indices = self._controlled_indices(len(ordered_names))
        targets = {}
        for agent_index in controlled_indices:
            state = scene_frame.states[agent_index]
            projection = scene_frame.projections[agent_index]
            base, gain, min_v, max_v = self._get_lookahead_params(agent_index, len(ordered_names))
            lookahead_distance = float(np.clip(
                base + gain * state.speed,
                min_v,
                max_v,
            ))
            target_idx = self._choose_target_index(state.pose_map_xy, projection.short_term_ref[:, :2], lookahead_distance)
            target_idx = min(target_idx, int(len(projection.short_term_ref) - 1))
            target_point = np.asarray(projection.short_term_ref[target_idx, :2], dtype=np.float32)
            reference_speed = float(projection.short_term_ref[target_idx, 2])
            target_s = float(projection.s + target_idx * self.sample_interval)
            reference_heading = float(projection.tangent_yaw)
            target_heading = self._estimate_target_heading(
                projection.short_term_ref[:, :2],
                target_idx,
                fallback_heading=reference_heading,
            )
            targets[ordered_names[agent_index]] = ReferenceTarget(
                vehicle_name=ordered_names[agent_index],
                agent_index=agent_index,
                current_s=float(projection.s),
                target_s=target_s,
                lookahead_distance=lookahead_distance,
                target_point_map=target_point,
                closest_center_map=np.asarray(projection.closest_center_xy, dtype=np.float32),
                reference_speed=reference_speed,
                reference_heading=reference_heading,
                target_heading=target_heading,
                short_term_index=int(target_idx),
            )
        return targets

    def _choose_target_index(self, pose_map_xy: np.ndarray, ref_points_xy: np.ndarray, lookahead_distance: float) -> int:
        del pose_map_xy, ref_points_xy
        target_idx = int(np.ceil(lookahead_distance / max(self.sample_interval, 1e-6)))
        return max(0, target_idx)

    def _estimate_target_heading(
        self,
        ref_points_xy: np.ndarray,
        target_idx: int,
        fallback_heading: float,
    ) -> float:
        points = np.asarray(ref_points_xy, dtype=np.float32)
        if len(points) < 2:
            return float(fallback_heading)
        next_idx = min(target_idx + 1, len(points) - 1)
        prev_idx = max(target_idx - 1, 0)
        if next_idx == target_idx and prev_idx == target_idx:
            return float(fallback_heading)
        if next_idx == target_idx:
            direction = points[target_idx] - points[prev_idx]
        else:
            direction = points[next_idx] - points[target_idx]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return float(fallback_heading)
        return float(np.arctan2(direction[1], direction[0]))

    def _get_lookahead_params(self, agent_index: int, n_agents: int):
        if agent_index == 0 or n_agents == 1:
            return (
                self.front_lookahead_base,
                self.front_lookahead_speed_gain,
                self.front_lookahead_min,
                self.front_lookahead_max,
            )
        return (
            self.rear_lookahead_base,
            self.rear_lookahead_speed_gain,
            self.rear_lookahead_min,
            self.rear_lookahead_max,
        )

    def _controlled_indices(self, n_agents: int) -> List[int]:
        if self.controlled_indices is not None:
            return [index for index in self.controlled_indices if 0 <= index < n_agents]
        if n_agents <= 0:
            return []
        if n_agents == 1:
            return [0]
        return [0, n_agents - 1]


class TractorStanleyController:
    def __init__(self, control_cfg, vehicle_cfg) -> None:
        self.cfg = control_cfg
        self.vehicle_cfg = vehicle_cfg
        self.longitudinal = PIDLongitudinalController(
            kp=control_cfg.speed_pid_kp,
            ki=control_cfg.speed_pid_ki,
            kd=control_cfg.speed_pid_kd,
        )
        self.steering = SteeringAdapter(
            max_steering_angle=control_cfg.max_steering_angle,
            alpha=control_cfg.steering_lowpass_alpha,
        )
        self.wheelbase = float(vehicle_cfg.l_f + vehicle_cfg.l_r)
        self.steering_command_sign = float(getattr(control_cfg, 'steering_command_sign', -1.0))
        self.heading_gain = float(getattr(control_cfg, 'stanley_heading_gain', 1.0))
        self.cross_track_gain = float(getattr(control_cfg, 'stanley_cross_track_gain', 1.2))
        self.feedforward_gain = float(getattr(control_cfg, 'stanley_feedforward_gain', 0.35))
        self.soft_speed = float(getattr(control_cfg, 'stanley_soft_speed', 0.5))

    def reset(self) -> None:
        self.longitudinal.reset()
        self.steering.reset()

    def compute_command(self, state, target: ReferenceTarget):
        control_point = np.asarray(state.pose_map_xy, dtype=np.float32)
        closest_center_local = transform_points_global_to_local(
            ego_xy=control_point,
            ego_yaw=state.yaw_map,
            points_xy=np.asarray([target.closest_center_map], dtype=np.float32),
        )[0]
        target_local = transform_points_global_to_local(
            ego_xy=control_point,
            ego_yaw=state.yaw_map,
            points_xy=np.asarray([target.target_point_map], dtype=np.float32),
        )[0]
        lateral_error = float(closest_center_local[1])
        heading_error = wrap_angle(target.reference_heading - state.yaw_map)
        heading_feedforward = wrap_angle(target.target_heading - target.reference_heading)
        cross_track_term = float(
            np.arctan2(
                self.cross_track_gain * lateral_error,
                self.soft_speed + max(state.speed, 0.0),
            )
        )
        delta_des = float(
            self.heading_gain * heading_error
            + cross_track_term
            + self.feedforward_gain * heading_feedforward
        )
        throttle, brake = self.longitudinal.step(state.speed, target.reference_speed, self.cfg.dt)
        steering = self.steering.step(self.steering_command_sign * delta_des)
        throttle = float(np.clip(throttle, 0.0, self.cfg.throttle_limit))
        brake = float(np.clip(brake, 0.0, self.cfg.brake_limit))
        command = LowLevelCommand(throttle=throttle, brake=brake, steering=steering)
        debug = TrackingDebugInfo(
            vehicle_name=target.vehicle_name,
            agent_index=target.agent_index,
            current_s=target.current_s,
            target_s=target.target_s,
            lookahead_distance=target.lookahead_distance,
            short_term_index=target.short_term_index,
            lateral_error=lateral_error,
            speed_error=float(target.reference_speed - state.speed),
            target_local_x=float(target_local[0]),
            target_local_y=float(target_local[1]),
            target_point_map=np.asarray(target.target_point_map, dtype=np.float32),
            control_point_map=control_point,
            closest_center_map=np.asarray(target.closest_center_map, dtype=np.float32),
            reference_heading=float(target.reference_heading),
            target_heading=float(target.target_heading),
            heading_error=heading_error,
            heading_feedforward=heading_feedforward,
            cross_track_term=cross_track_term,
            delta_des=delta_des,
            steering_cmd=steering,
            throttle_cmd=throttle,
            brake_cmd=brake,
            reference_speed=float(target.reference_speed),
            current_speed=float(state.speed),
        )
        return command, debug


class CenterlinePIDController(BaseDeploymentController):
    def __init__(
        self,
        registry,
        control_cfg,
        sample_interval: float,
        platoon_position_gain: float = 0.8,
        front_lookahead_base: float = 6.0,
        front_lookahead_speed_gain: float = 0.8,
        front_lookahead_min: float = 4.0,
        front_lookahead_max: float = 14.0,
        rear_lookahead_base: float = 4.0,
        rear_lookahead_speed_gain: float = 0.3,
        rear_lookahead_min: float = 2.5,
        rear_lookahead_max: float = 8.0,
    ) -> None:
        self.registry = registry
        self.control_cfg = control_cfg
        self.platoon_position_gain = float(platoon_position_gain)
        self.controlled_indices = self._controlled_indices(registry.n_agents)
        self.planner = FrontRearReferencePlanner(
            registry=registry,
            sample_interval=sample_interval,
            front_lookahead_base=front_lookahead_base,
            front_lookahead_speed_gain=front_lookahead_speed_gain,
            front_lookahead_min=front_lookahead_min,
            front_lookahead_max=front_lookahead_max,
            rear_lookahead_base=rear_lookahead_base,
            rear_lookahead_speed_gain=rear_lookahead_speed_gain,
            rear_lookahead_min=rear_lookahead_min,
            rear_lookahead_max=rear_lookahead_max,
            controlled_indices=self.controlled_indices,
        )
        self.trackers = {
            agent_index: TractorStanleyController(control_cfg, registry.config_of(agent_index))
            for agent_index in self.controlled_indices
        }
        self.metadata = {
            "type": "CenterlinePIDController",
            "controlled_indices": list(self.controlled_indices),
            "platoon_position_gain": self.platoon_position_gain,
        }
        self.last_debug_info: Dict[str, TrackingDebugInfo] = {}
        self.last_actor_debug_info: Dict[str, Dict] = {}
        self.last_actor_action_map: Dict[str, ActorAction] = {}

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()
        self.last_debug_info = {}
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        del obs_dict
        ordered_names = list(vehicle_names)
        targets = self.planner.plan(scene_frame, ordered_names)
        target_agent_s = self._get_dynamic_target_arc_positions(scene_frame)
        commands: Dict[str, LowLevelCommand] = {}
        self.last_debug_info = {}
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}
        for vehicle_name, target in targets.items():
            front_speed = float(scene_frame.states[target.agent_index - 1].speed)
            s_error = float(target_agent_s[target.agent_index] - target.current_s)
            target.target_s = float(target_agent_s[target.agent_index])
            target.reference_speed = float(
                np.clip(
                    front_speed + self.platoon_position_gain * s_error,
                    0.0,
                    self.control_cfg.max_speed,
                )
            )
            state = scene_frame.states[target.agent_index]
            command, debug = self.trackers[target.agent_index].compute_command(state, target)
            commands[vehicle_name] = command
            self.last_debug_info[vehicle_name] = debug
        return commands

    def _controlled_indices(self, n_agents: int) -> List[int]:
        if n_agents <= 2:
            return []
        return list(range(1, n_agents - 1))

    def _get_dynamic_target_arc_positions(self, scene_frame) -> np.ndarray:
        s_front = float(scene_frame.projections[0].s)
        s_rear = float(scene_frame.projections[-1].s)
        if self.registry.n_agents <= 1:
            return np.asarray([s_front], dtype=np.float32)
        desired_gap_s = (s_front - s_rear) / max(self.registry.n_agents - 1, 1)
        agent_indices = np.arange(self.registry.n_agents, dtype=np.float32)
        return (s_front - desired_gap_s * agent_indices).astype(np.float32)


class CenterlineMPPIController(BaseDeploymentController):
    def __init__(
        self,
        registry,
        control_cfg,
        sample_interval: float,
        projector,
        device: str = "cpu",
        horizon_steps: int = 3,
        num_samples: int = 256,
        param_lambda: float = 10.0,
        exploration: float = 0.1,
        debug_top_k: int = 8,
    ) -> None:
        import importlib.util
        from pathlib import Path

        import torch

        module_path = Path("/home/yons/Graduation/VMAS_occt/vmas/scenarios/simple_mppi.py")
        spec = importlib.util.spec_from_file_location("vmas_simple_mppi", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load SimpleMPPIController from {module_path}")
        simple_mppi_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(simple_mppi_module)
        SimpleMPPIController = simple_mppi_module.SimpleMPPIController

        self.registry = registry
        self.sample_interval = float(sample_interval)
        self.control_cfg = control_cfg
        if projector is None:
            raise ValueError("CenterlineMPPIController requires a valid road projector")
        self.projector = projector
        self.device = device
        self.controlled_indices = self._controlled_indices(registry.n_agents)
        self.low_level = {
            agent_index: LowLevelController(control_cfg)
            for agent_index in self.controlled_indices
        }
        vehicle_cfg = registry.config_of(0)
        self.simple_mppi = SimpleMPPIController(
            num_agents=registry.n_agents,
            device=torch.device(device),
            dt=float(control_cfg.dt),
            l_f=float(vehicle_cfg.l_f),
            l_r=float(vehicle_cfg.l_r),
            max_steer_abs=float(control_cfg.max_steering_angle),
            max_accel_abs=float(control_cfg.max_acceleration),
            max_speed=float(control_cfg.max_speed),
            horizon_step_T=int(horizon_steps),
            number_of_samples_K=int(num_samples),
            param_exploration=float(exploration),
            param_lambda=float(param_lambda),
            debug_top_k=int(debug_top_k),
        )
        self.metadata = {
            "type": "CenterlineMPPIController",
            "controlled_indices": list(self.controlled_indices),
            "device": device,
            "horizon_steps": int(horizon_steps),
            "num_samples": int(num_samples),
            "lambda": float(param_lambda),
            "exploration": float(exploration),
        }
        self.last_actor_debug_info: Dict[str, Dict] = {}
        self.last_actor_action_map: Dict[str, ActorAction] = {}
        self.last_mppi_debug_info: Dict[str, Dict] = {}

    def reset(self) -> None:
        self.simple_mppi.reset()
        for controller in self.low_level.values():
            controller.reset()
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}
        self.last_mppi_debug_info = {}

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        del obs_dict
        ordered_names = list(vehicle_names)
        commands: Dict[str, LowLevelCommand] = {}
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}
        self.last_mppi_debug_info = {}
        target_agent_s = self._get_dynamic_target_arc_positions(scene_frame)
        for agent_index in self.controlled_indices:
            vehicle_name = ordered_names[agent_index]
            state = scene_frame.states[agent_index]
            ref_points, ref_speeds = self._build_platoon_reference(scene_frame, agent_index, target_agent_s)
            observed_x = np.asarray(
                [state.pose_map_xy[0], state.pose_map_xy[1], state.yaw_map, state.speed],
                dtype=np.float32,
            )
            action, _, _ = self.simple_mppi.command(
                agent_idx=agent_index,
                observed_x=observed_x,
                ref_points=ref_points,
                ref_speeds=ref_speeds,
            )
            mppi_debug = self.simple_mppi.last_debug.get(agent_index, {})
            actor_action = ActorAction(
                acceleration_mps2=float(action[1].item() if hasattr(action[1], "item") else action[1]),
                front_wheel_angle_rad=float(action[0].item() if hasattr(action[0], "item") else action[0]),
            )
            self.last_actor_action_map[vehicle_name] = actor_action
            command = self.low_level[agent_index].step(actor_action, state)
            heading_vec = np.asarray(
                [np.cos(state.yaw_map), np.sin(state.yaw_map)],
                dtype=np.float32,
            )
            measured_acc_long = float(np.dot(np.asarray(state.acc_map_xy, dtype=np.float32), heading_vec))
            commands[vehicle_name] = command
            self.last_actor_debug_info[vehicle_name] = {
                "acceleration_mps2": float(actor_action.acceleration_mps2),
                "front_wheel_angle_rad": float(actor_action.front_wheel_angle_rad),
                "measured_acc_long": measured_acc_long,
                "throttle_cmd": float(command.throttle),
                "brake_cmd": float(command.brake),
                "steering_cmd": float(command.steering),
                "current_speed": float(state.speed),
                "controller_type": "mppi",
                "target_s": float(target_agent_s[agent_index]),
            }
            self.last_mppi_debug_info[vehicle_name] = {
                "target_s": float(target_agent_s[agent_index]),
                "ref_points": np.asarray(mppi_debug.get("ref_points", ref_points), dtype=np.float32),
                "ref_speeds": np.asarray(mppi_debug.get("ref_speeds", ref_speeds), dtype=np.float32),
                "optimal_traj": np.asarray(mppi_debug.get("optimal_traj", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32),
                #"sampled_trajs": np.asarray(mppi_debug.get("sampled_trajs", np.zeros((0, 0, 4), dtype=np.float32)), dtype=np.float32),
                "costs": np.asarray(mppi_debug.get("costs", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
            }
        return commands

    def _get_dynamic_target_arc_positions(self, scene_frame) -> np.ndarray:
        s_front = float(scene_frame.projections[0].s)
        s_rear = float(scene_frame.projections[-1].s)
        if self.registry.n_agents <= 1:
            return np.asarray([s_front], dtype=np.float32)
        desired_gap_s = (s_front - s_rear) / max(self.registry.n_agents - 1, 1)
        agent_indices = np.arange(self.registry.n_agents, dtype=np.float32)
        return (s_front - desired_gap_s * agent_indices).astype(np.float32)

    def _build_platoon_reference(self, scene_frame, agent_index: int, target_agent_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        front_index = max(agent_index - 1, 0)
        front_vehicle_speed = max(float(scene_frame.states[front_index].speed), 1e-3)
        query_interval = max(front_vehicle_speed * float(self.control_cfg.dt), 1e-3)
        query_s = self.projector._clamp_s(
            target_agent_s[agent_index]
            + np.arange(self.simple_mppi.T + 1, dtype=np.float32) * query_interval
        )
        ref_points = self.projector._query_xy(self.projector.center_xy, query_s).astype(np.float32)
        ref_speeds = np.full((self.simple_mppi.T + 1,), front_vehicle_speed, dtype=np.float32)
        return ref_points, ref_speeds

    def _controlled_indices(self, n_agents: int) -> List[int]:
        if n_agents <= 2:
            return []
        return list(range(1, n_agents - 1))


class FrontRearCooperativeController(BaseDeploymentController):
    def __init__(
        self,
        registry,
        control_cfg,
        sample_interval: float,
        middle_controller: Optional[BaseDeploymentController] = None,
        front_lookahead_base: float = 6.0,
        front_lookahead_speed_gain: float = 0.8,
        front_lookahead_min: float = 4.0,
        front_lookahead_max: float = 14.0,
        rear_lookahead_base: float = 4.0,
        rear_lookahead_speed_gain: float = 0.3,
        rear_lookahead_min: float = 2.5,
        rear_lookahead_max: float = 8.0,
        stanley_heading_gain: float = 1.0,
        stanley_cross_track_gain: float = 1.2,
        stanley_feedforward_gain: float = 0.35,
        stanley_soft_speed: float = 0.5,
    ) -> None:
        self.registry = registry
        self.middle_controller = middle_controller
        control_cfg.stanley_heading_gain = stanley_heading_gain
        control_cfg.stanley_cross_track_gain = stanley_cross_track_gain
        control_cfg.stanley_feedforward_gain = stanley_feedforward_gain
        control_cfg.stanley_soft_speed = stanley_soft_speed
        self.planner = FrontRearReferencePlanner(
            registry=registry,
            sample_interval=sample_interval,
            front_lookahead_base=front_lookahead_base,
            front_lookahead_speed_gain=front_lookahead_speed_gain,
            front_lookahead_min=front_lookahead_min,
            front_lookahead_max=front_lookahead_max,
            rear_lookahead_base=rear_lookahead_base,
            rear_lookahead_speed_gain=rear_lookahead_speed_gain,
            rear_lookahead_min=rear_lookahead_min,
            rear_lookahead_max=rear_lookahead_max,
        )
        controlled_indices = self._controlled_indices(registry.n_agents)
        self.front_rear_trackers = {
            agent_index: TractorStanleyController(control_cfg, registry.config_of(agent_index))
            for agent_index in controlled_indices
        }
        self.expected_obs_dim = getattr(middle_controller, "expected_obs_dim", None)
        self.metadata = {
            "type": "FrontRearCooperativeController",
            "controlled_indices": controlled_indices,
            "middle_controller": type(middle_controller).__name__ if middle_controller is not None else None,
            "front_lookahead_base": front_lookahead_base,
            "front_lookahead_speed_gain": front_lookahead_speed_gain,
            "front_lookahead_min": front_lookahead_min,
            "front_lookahead_max": front_lookahead_max,
            "rear_lookahead_base": rear_lookahead_base,
            "rear_lookahead_speed_gain": rear_lookahead_speed_gain,
            "rear_lookahead_min": rear_lookahead_min,
            "rear_lookahead_max": rear_lookahead_max,
            "stanley_heading_gain": stanley_heading_gain,
            "stanley_cross_track_gain": stanley_cross_track_gain,
            "stanley_feedforward_gain": stanley_feedforward_gain,
            "stanley_soft_speed": stanley_soft_speed,
        }
        if middle_controller is not None and hasattr(middle_controller, "metadata"):
            self.metadata["middle_metadata"] = getattr(middle_controller, "metadata")
        self.last_debug_info: Dict[str, TrackingDebugInfo] = {}
        self.last_actor_debug_info: Dict[str, Dict] = {}
        self.last_actor_action_map: Dict[str, ActorAction] = {}
        self.last_mppi_debug_info: Dict[str, Dict] = {}

    def reset(self) -> None:
        if self.middle_controller is not None:
            self.middle_controller.reset()
        for tracker in self.front_rear_trackers.values():
            tracker.reset()
        self.last_mppi_debug_info = {}

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        ordered_names = list(vehicle_names)
        commands = {}
        self.last_actor_debug_info = {}
        self.last_actor_action_map = {}
        self.last_mppi_debug_info = {}
        if self.middle_controller is not None:
            middle_commands = self.middle_controller.compute_commands(obs_dict, scene_frame, ordered_names)
            self.last_actor_debug_info = getattr(self.middle_controller, "last_actor_debug_info", {})
            self.last_actor_action_map = getattr(self.middle_controller, "last_actor_action_map", {})
            self.last_mppi_debug_info = getattr(self.middle_controller, "last_mppi_debug_info", {})
            for agent_index, vehicle_name in enumerate(ordered_names):
                if agent_index not in self._controlled_indices(len(ordered_names)) and vehicle_name in middle_commands:
                    commands[vehicle_name] = middle_commands[vehicle_name]
        ref_targets = self.planner.plan(scene_frame, ordered_names)
        self.last_debug_info = {}
        for vehicle_name, target in ref_targets.items():
            state = scene_frame.states[target.agent_index]
            command, debug = self.front_rear_trackers[target.agent_index].compute_command(state, target)
            commands[vehicle_name] = command
            self.last_debug_info[vehicle_name] = debug
        for vehicle_name in ordered_names:
            commands.setdefault(vehicle_name, LowLevelCommand(throttle=0.0, brake=0.0, steering=0.0))
        return commands

    def _controlled_indices(self, n_agents: int) -> List[int]:
        if n_agents <= 0:
            return []
        if n_agents == 1:
            return [0]
        return [0, n_agents - 1]
