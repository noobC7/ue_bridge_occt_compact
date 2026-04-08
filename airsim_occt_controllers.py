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
        import torch
        import torch.nn as nn

        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_state_dict = checkpoint["policy_state_dict"]
        supported_prefix = "module.0.module.0.group_mlp_0.params"
        if not any(key.startswith(supported_prefix) for key in policy_state_dict.keys()):
            raise ValueError(
                "Checkpoint format is not the expected shared GroupSharedMLP layout. "
                "This wrapper currently supports the shared-parameter actor checkpoint used in mappo_occt_3_followers."
            )

        weight0 = policy_state_dict[f"{supported_prefix}.0.weight"]
        weight1 = policy_state_dict[f"{supported_prefix}.2.weight"]
        weight2 = policy_state_dict[f"{supported_prefix}.4.weight"]
        bias0 = policy_state_dict[f"{supported_prefix}.0.bias"]
        bias1 = policy_state_dict[f"{supported_prefix}.2.bias"]
        bias2 = policy_state_dict[f"{supported_prefix}.4.bias"]

        obs_dim = int(weight0.shape[1])
        hidden_dim0 = int(weight0.shape[0])
        hidden_dim1 = int(weight1.shape[0])
        output_dim = int(weight2.shape[0])
        if output_dim % 2 != 0:
            raise ValueError(f"Actor output dim must be even, got {output_dim}")
        self.obs_dim = obs_dim
        self.action_dim = output_dim // 2
        self.device = device
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim0),
            nn.Tanh(),
            nn.Linear(hidden_dim0, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, output_dim),
        ).to(device)
        with torch.no_grad():
            self.model[0].weight.copy_(weight0.to(device))
            self.model[0].bias.copy_(bias0.to(device))
            self.model[2].weight.copy_(weight1.to(device))
            self.model[2].bias.copy_(bias1.to(device))
            self.model[4].weight.copy_(weight2.to(device))
            self.model[4].bias.copy_(bias2.to(device))
        self.model.eval()
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
        self.checkpoint_path = checkpoint_path
        self.iteration = checkpoint.get("iteration")
        self.total_frames = checkpoint.get("total_frames")

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
        for agent_index, vehicle_name in enumerate(ordered_names):
            actor_action = self._coerce_actor_action(actor_outputs[vehicle_name])
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

    def reset(self) -> None:
        if self.middle_controller is not None:
            self.middle_controller.reset()
        for tracker in self.front_rear_trackers.values():
            tracker.reset()

    def compute_commands(self, obs_dict, scene_frame, vehicle_names: Iterable[str]) -> Dict[str, LowLevelCommand]:
        ordered_names = list(vehicle_names)
        commands = {}
        self.last_actor_debug_info = {}
        if self.middle_controller is not None:
            middle_commands = self.middle_controller.compute_commands(obs_dict, scene_frame, ordered_names)
            self.last_actor_debug_info = getattr(self.middle_controller, "last_actor_debug_info", {})
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
