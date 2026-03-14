import argparse
from pathlib import Path
from typing import List

import numpy as np

import setup_vsim

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_calibration import AlignmentCalibrator
from airsim_occt_config import EnvConfig, MapConfig, VehicleConfig
from airsim_occt_controllers import ActorDeploymentController, ConstantLowLevelController, FrontRearCooperativeController
from airsim_occt_fleet_registry import FleetRegistry
from airsim_occt_schema import ActorAction, Transform2D
from airsim_occt_obs_manifest import NEIGHBOR_BLOCK_ORDER, SELF_BLOCK_ORDER, build_obs_layout
from airsim_occt_plotting import build_start_aligned_world_to_map, get_selected_road_metadata
from airsim_occt_tracking_recorder import TrackingLogRecorder, make_output_dir


def discover_vehicle_names(host: str, port: int):
    probe_cfg = EnvConfig(host=host, port=port)
    io = AirSimIO(probe_cfg)
    io.connect()
    vehicle_names = io.list_vehicles()
    vehicle_names.remove('cargo')
    if not vehicle_names:
        raise RuntimeError('No vehicles returned by AirSim listVehicles()')
    return vehicle_names


def build_env_config(args) -> EnvConfig:
    vehicle_cfgs = [
        VehicleConfig(
            vehicle_name=name,
            length=args.vehicle_length,
            width=args.vehicle_width,
            l_f=args.l_f,
            l_r=args.l_r,
        )
        for name in args.vehicles
    ]
    map_cfg = MapConfig(
        cr_map_dir=args.map_dir,
        sample_gap=args.map_sample_gap,
        min_lane_width=args.min_lane_width,
        min_lane_len=args.min_lane_len,
        max_ref_v=args.max_ref_v,
        is_constant_ref_v=args.is_constant_ref_v,
    )
    cfg = EnvConfig(
        host=args.host,
        port=args.port,
        vehicle_configs=vehicle_cfgs,
        map_cfg=map_cfg,
        use_sim_pause_clock=not args.no_pause,
        road_env_index=args.road_env_index,
    )
    cfg.obs.sample_interval = args.sample_interval
    cfg.obs.boundary_offset = args.boundary_offset
    cfg.obs.n_points_short_term = args.n_points_short_term
    cfg.obs.n_points_nearing_boundary = args.n_points_nearing_boundary
    cfg.obs.n_stored_steps = args.n_stored_steps
    cfg.obs.n_observed_steps = args.n_observed_steps
    cfg.obs.mask_ref_v = args.mask_ref_v
    cfg.obs.include_hinge_info = not args.disable_hinge_info
    cfg.control.dt = args.dt
    cfg.control.max_speed = args.max_speed
    cfg.control.max_acceleration = args.max_acceleration
    cfg.control.max_steering_angle = args.max_steering_angle
    cfg.control.stanley_heading_gain = args.stanley_heading_gain
    cfg.control.stanley_cross_track_gain = args.stanley_cross_track_gain
    cfg.control.stanley_feedforward_gain = args.stanley_feedforward_gain
    cfg.control.stanley_soft_speed = args.stanley_soft_speed
    return cfg


def build_transform(args) -> Transform2D:
    if args.transform_file:
        calibrator = AlignmentCalibrator()
        return calibrator.load(args.transform_file)
    return Transform2D(mat=np.eye(2, dtype=np.float32), bias=np.zeros((2,), dtype=np.float32))


def build_road(cfg: EnvConfig, args):
    try:
        import torch
        from ivs_python_example.occt_map import OcctCRMap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "build_road() requires `torch` and the map dependencies from your training env. "
            "Please run this script inside the same conda/venv used for training."
        ) from exc

    road = OcctCRMap(
        batch_dim=max(1, args.road_env_index + 1),
        device=torch.device(args.device),
        cr_map_dir=cfg.map_cfg.cr_map_dir,
        sample_gap=cfg.map_cfg.sample_gap,
        min_lane_width=cfg.map_cfg.min_lane_width,
        min_lane_len=cfg.map_cfg.min_lane_len,
        max_ref_v=cfg.map_cfg.max_ref_v,
        is_constant_ref_v=cfg.map_cfg.is_constant_ref_v,
        rod_len=args.rod_len,
        n_agents=cfg.obs.n_agents,
    )
    return road


def print_obs_summary(obs_dict, preview_dim: int) -> None:
    for vehicle_name, obs in obs_dict.items():
        obs = np.asarray(obs, dtype=np.float32)
        preview = np.round(obs[:preview_dim], 4).tolist()
        print(
            f"vehicle={vehicle_name} obs_dim={obs.shape[0]} "
            f"finite={bool(np.isfinite(obs).all())} mean={float(obs.mean()):.6f} "
            f"std={float(obs.std()):.6f} max_abs={float(np.abs(obs).max()):.6f}"
        )
        print(f"  obs_head={preview}")


def print_obs_block_summary(obs_dict, obs_cfg, agent_index: int) -> None:
    vehicle_names = list(obs_dict.keys())
    if agent_index < 0 or agent_index >= len(vehicle_names):
        raise IndexError(f"agent_index {agent_index} out of range for {len(vehicle_names)} vehicles")
    layout = build_obs_layout(obs_cfg)
    vehicle_name = vehicle_names[agent_index]
    obs = np.asarray(obs_dict[vehicle_name], dtype=np.float32)
    print(f"[DEMO] block summary for {vehicle_name}")
    cursor = 0
    for block_name in SELF_BLOCK_ORDER:
        dim = layout.self_block_dims[block_name]
        if dim <= 0:
            continue
        block = obs[cursor:cursor + dim]
        print(
            f"  self::{block_name} dim={dim} mean={float(block.mean()):.6f} "
            f"max_abs={float(np.abs(block).max()):.6f} head={np.round(block[:min(6, dim)], 4).tolist()}"
        )
        cursor += dim
    for neighbor_prefix in ["front", "rear"]:
        for block_name in NEIGHBOR_BLOCK_ORDER:
            dim = layout.neighbor_block_dims[block_name]
            block = obs[cursor:cursor + dim]
            print(
                f"  {neighbor_prefix}::{block_name} dim={dim} mean={float(block.mean()):.6f} "
                f"max_abs={float(np.abs(block).max()):.6f} head={np.round(block[:min(6, dim)], 4).tolist()}"
            )
            cursor += dim


def build_zero_actions(vehicle_names):
    return {
        vehicle_name: ActorAction(acceleration_mps2=0.0, front_wheel_angle_rad=0.0)
        for vehicle_name in vehicle_names
    }


def build_demo_controller(args, cfg, vehicle_names):
    if args.controller_mode == "constant":
        middle_controller = ConstantLowLevelController(
            throttle=args.demo_throttle,
            steering=args.demo_steering,
            brake=args.demo_brake,
        )
    elif args.controller_mode == "actor":
        middle_controller = ActorDeploymentController.from_checkpoint(
            checkpoint_path=args.actor_checkpoint,
            control_cfg=cfg.control,
            vehicle_names=vehicle_names,
            device=args.actor_device,
        )
    else:
        raise ValueError(f"Unsupported controller_mode: {args.controller_mode}")
    return FrontRearCooperativeController(
        registry=FleetRegistry(cfg.vehicle_configs),
        control_cfg=cfg.control,
        sample_interval=cfg.obs.sample_interval,
        middle_controller=middle_controller,
        front_lookahead_base=args.front_lookahead_base,
        front_lookahead_speed_gain=args.front_lookahead_speed_gain,
        front_lookahead_min=args.front_lookahead_min,
        front_lookahead_max=args.front_lookahead_max,
        rear_lookahead_base=args.rear_lookahead_base,
        rear_lookahead_speed_gain=args.rear_lookahead_speed_gain,
        rear_lookahead_min=args.rear_lookahead_min,
        rear_lookahead_max=args.rear_lookahead_max,
        stanley_heading_gain=args.stanley_heading_gain,
        stanley_cross_track_gain=args.stanley_cross_track_gain,
        stanley_feedforward_gain=args.stanley_feedforward_gain,
        stanley_soft_speed=args.stanley_soft_speed,
    )


def print_tracking_debug(info) -> None:
    controller_debug = info.get('controller_debug', {})
    if not controller_debug:
        print('[DEMO] no controller_debug available')
        return
    for vehicle_name, debug in controller_debug.items():
        print(
            f"[TRACK] vehicle={vehicle_name} idx={debug.agent_index} s={debug.current_s:.3f}->{debug.target_s:.3f} "
            f"lookahead={debug.lookahead_distance:.3f} ref_idx={debug.short_term_index} "
            f"e_lat={debug.lateral_error:.3f} e_v={debug.speed_error:.3f} "
            f"heading_ref={debug.reference_heading:.3f} heading_tgt={debug.target_heading:.3f} "
            f"e_heading={debug.heading_error:.3f} ff={debug.heading_feedforward:.3f} cte_term={debug.cross_track_term:.3f} "
            f"target_local=({debug.target_local_x:.3f},{debug.target_local_y:.3f}) "
            f"delta_des={debug.delta_des:.3f} steer={debug.steering_cmd:.3f} "
            f"throttle={debug.throttle_cmd:.3f} brake={debug.brake_cmd:.3f} "
            f"v={debug.current_speed:.3f}/{debug.reference_speed:.3f}"
        )


def print_actor_debug(info) -> None:
    actor_debug = info.get("actor_debug", {})
    if not actor_debug:
        print("[DEMO] no actor_debug available")
        return
    for vehicle_name, debug in actor_debug.items():
        print(
            f"[ACTOR] vehicle={vehicle_name} act_acc={debug['acceleration_mps2']:.3f} "
            f"act_delta={debug['front_wheel_angle_rad']:.3f} "
            f"cmd_throttle={debug['throttle_cmd']:.3f} cmd_brake={debug['brake_cmd']:.3f} "
            f"cmd_steer={debug['steering_cmd']:.3f} speed={debug['current_speed']:.3f}"
        )


def print_info_summary(info) -> None:
    s_info = info.get("s", {})
    pose_info = info.get("pose_map_xy", {})
    speed_info = info.get("speed", {})
    actor_info = info.get("actor_debug",{})
    controller_info = info.get("controller_debug",{})
    for vehicle_name in s_info:
        pose = np.round(np.asarray(pose_info[vehicle_name]), 4).tolist()
        speed = np.round(np.asarray(speed_info[vehicle_name]), 4).tolist()
        if vehicle_name in controller_info:
            throttle= controller_info[vehicle_name].throttle_cmd
            steering = controller_info[vehicle_name].steering_cmd
            print(f"vehicle={vehicle_name} s={float(s_info[vehicle_name]):.4f} speed={speed}, throttle:{throttle}, steering:{steering}")
        elif vehicle_name in actor_info:
            acc= actor_info[vehicle_name]["acceleration_mps2"]
            steering = actor_info[vehicle_name]["front_wheel_angle_rad"]/np.pi*180
            throttle= actor_info[vehicle_name]["throttle_cmd"]
            steering = actor_info[vehicle_name]["steering_cmd"]
            print(f"vehicle={vehicle_name} s={float(s_info[vehicle_name]):.4f} speed={speed}, acc:{acc}, steering:{steering}, throttle:{throttle}, steering:{steering}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo entry for AirSimOcctMARLEnv.reset()")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicles", nargs="*", default=None)
    parser.add_argument("--map-dir", required=True, help="CommonRoad map directory used by OcctCRMap")
    parser.add_argument("--transform-file", default=None, help="Optional world->map transform json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--road-env-index", type=int, default=0)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--boundary-offset", type=float, default=-2.0)
    parser.add_argument("--n-points-short-term", type=int, default=4)
    parser.add_argument("--n-points-nearing-boundary", type=int, default=5)
    parser.add_argument("--n-stored-steps", type=int, default=5)
    parser.add_argument("--n-observed-steps", type=int, default=5)
    parser.add_argument("--mask-ref-v", action="store_true")
    parser.add_argument("--disable-hinge-info", action="store_true")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--max-acceleration", type=float, default=3.0)
    parser.add_argument("--max-steering-angle", type=float, default=0.6108652382)
    parser.add_argument("--vehicle-length", type=float, default=3.82)
    parser.add_argument("--vehicle-width", type=float, default=1.5)
    parser.add_argument("--l-f", type=float, default=1.17)
    parser.add_argument("--l-r", type=float, default=1.15)
    parser.add_argument("--rod-len", type=float, default=None)
    parser.add_argument("--map-sample-gap", type=float, default=1.0)
    parser.add_argument("--min-lane-width", type=float, default=2.1)
    parser.add_argument("--min-lane-len", type=float, default=70.0)
    parser.add_argument("--max-ref-v", type=float, default=20.0 / 3.6)
    parser.add_argument("--is-constant-ref-v", action="store_true")
    parser.add_argument("--preview-dim", type=int, default=16)
    parser.add_argument("--inspect-agent-index", type=int, default=0)
    parser.add_argument("--plot-road", action="store_true")
    parser.add_argument("--plot-observation-points", action="store_true")
    parser.add_argument("--plot-all-observation-points", action="store_true")
    parser.add_argument("--plot-z", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--plot-duration", type=float, default=-1.0)
    parser.add_argument("--no-align-road-start", action="store_true")
    parser.add_argument("--step-count", type=int, default=0)
    parser.add_argument("--demo-throttle", type=float, default=0.15)
    parser.add_argument("--demo-steering", type=float, default=0.0)
    parser.add_argument("--demo-brake", type=float, default=0.0)
    parser.add_argument("--controller-mode", choices=["constant", "actor"], default="constant")
    parser.add_argument("--actor-checkpoint", default="/home/yons/Graduation/rl_occt/outputs/2026-03-12/11-41-19/checkpoints/checkpoint_iter_330_frames_19860000.pt")
    parser.add_argument("--actor-device", default="cpu")
    parser.add_argument("--front-lookahead-base", type=float, default=2.5)
    parser.add_argument("--front-lookahead-speed-gain", type=float, default=0.1)
    parser.add_argument("--front-lookahead-min", type=float, default=1.5)
    parser.add_argument("--front-lookahead-max", type=float, default=4.0)
    parser.add_argument("--rear-lookahead-base", type=float, default=2.0)
    parser.add_argument("--rear-lookahead-speed-gain", type=float, default=0.05)
    parser.add_argument("--rear-lookahead-min", type=float, default=1.0)
    parser.add_argument("--rear-lookahead-max", type=float, default=3.0)
    parser.add_argument("--stanley-heading-gain", type=float, default=1.0)
    parser.add_argument("--stanley-cross-track-gain", type=float, default=1.6)
    parser.add_argument("--stanley-feedforward-gain", type=float, default=0.45)
    parser.add_argument("--stanley-soft-speed", type=float, default=0.3)
    parser.add_argument("--print-obs-debug", action="store_true")
    parser.add_argument("--print-tracking-debug", action="store_true")
    parser.add_argument("--no-save-tracking-log", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(f"map dir not found: {map_dir}")

    if not args.vehicles:
        print('[DEMO] Discovering vehicles from AirSim...')
        args.vehicles = discover_vehicle_names(args.host, args.port)
        print(f"[DEMO] discovered vehicles={args.vehicles}")

    cfg = build_env_config(args)
    print("[DEMO] Building road...")
    road = build_road(cfg, args)
    if args.transform_file:
        transform = build_transform(args)
    elif not args.no_align_road_start:
        transform = build_start_aligned_world_to_map(road, road_env_index=args.road_env_index)
        cfg.alignment.flip_world_y = True
        print("[DEMO] using start-aligned transform with final world-y flip")
    else:
        transform = build_transform(args)
    metadata = get_selected_road_metadata(road, road_env_index=args.road_env_index)
    print(
        f"[DEMO] selected_path_index={metadata['selected_path_index']} "
        f"map_name={metadata['map_name']} path_ids={metadata['path_ids']} "
        f"s_max={metadata['s_max']:.3f} num_center_pts={metadata['num_center_pts']}"
    )
    try:
        from airsim_occt_env import AirSimOcctMARLEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Creating AirSimOcctMARLEnv requires the AirSim Python deps in your runtime env, "
            "including `msgpackrpc`. Please run this script inside the same env used by IVS/AirSim."
        ) from exc

    print("[DEMO] Creating env...")
    env = AirSimOcctMARLEnv(cfg, road=road, transform=transform)
    tracking_recorder = None
    if (not args.no_save_tracking_log) and args.step_count > 0:
        output_dir = make_output_dir(args.output_dir, prefix="tracking")
        tracking_recorder = TrackingLogRecorder(
            output_dir=output_dir,
            metadata={
                "vehicles": args.vehicles,
                "road_metadata": metadata,
                "controller_mode": args.controller_mode,
                "actor_checkpoint": args.actor_checkpoint if args.controller_mode == "actor" else None,
                "demo_throttle": args.demo_throttle,
                "demo_steering": args.demo_steering,
                "demo_brake": args.demo_brake,
                "front_lookahead_base": args.front_lookahead_base,
                "front_lookahead_speed_gain": args.front_lookahead_speed_gain,
                "front_lookahead_min": args.front_lookahead_min,
                "front_lookahead_max": args.front_lookahead_max,
                "rear_lookahead_base": args.rear_lookahead_base,
                "rear_lookahead_speed_gain": args.rear_lookahead_speed_gain,
                "rear_lookahead_min": args.rear_lookahead_min,
                "rear_lookahead_max": args.rear_lookahead_max,
                "stanley_heading_gain": args.stanley_heading_gain,
                "stanley_cross_track_gain": args.stanley_cross_track_gain,
                "stanley_feedforward_gain": args.stanley_feedforward_gain,
                "stanley_soft_speed": args.stanley_soft_speed,
                "dt": args.dt,
            },
        )
        print(f"[DEMO] tracking logs will be saved under {output_dir}")

    try:
        print("[DEMO] Calling reset()...")
        obs_dict, reset_info = env.reset()
        print("[DEMO] reset() succeeded.")
        print(f"[DEMO] vehicles={reset_info.get('vehicle_names')}")
        if args.print_obs_debug:
            print_obs_summary(obs_dict, preview_dim=args.preview_dim)
        info = env._build_info()
        if tracking_recorder is not None:
            tracking_recorder.add_step(-1, info)
        print_info_summary(info)
        if args.print_obs_debug:
            print_obs_block_summary(obs_dict, cfg.obs, agent_index=args.inspect_agent_index)
        if args.plot_road or args.plot_observation_points or args.plot_all_observation_points:
            env.render_debug_markers(
                plot_road=args.plot_road,
                plot_observation_points=args.plot_all_observation_points,
                plot_z=args.plot_z,
                road_thickness=3.0,
                point_size=args.point_size,
                duration=args.plot_duration,
                is_persistent=True,
                clear_existing=True,
            )
            if args.plot_observation_points and not args.plot_all_observation_points:
                env.plot_agent_observation_points(
                    agent_index=args.inspect_agent_index,
                    plot_z=args.plot_z,
                    point_size=args.point_size,
                    duration=args.plot_duration,
                    is_persistent=True,
                    clear_existing=not args.plot_road,
                )
            print(
                f"[DEMO] rendered debug markers: plot_road={args.plot_road} "
                f"plot_all_observation_points={args.plot_all_observation_points} "
                f"plot_single_observation={args.plot_observation_points and not args.plot_all_observation_points}"
            )
        if args.step_count > 0:
            demo_controller = build_demo_controller(args, cfg, reset_info.get('vehicle_names'))
            demo_controller.reset()
            if args.controller_mode == "actor":
                current_obs_dim = next(iter(obs_dict.values())).shape[0]
                expected_obs_dim = getattr(demo_controller, 'expected_obs_dim', None)
                print(f"[DEMO] actor obs_dim check current={current_obs_dim} expected={expected_obs_dim}")
                if expected_obs_dim is not None and current_obs_dim != expected_obs_dim:
                    raise ValueError(
                        f"Actor checkpoint obs_dim mismatch: current={current_obs_dim}, expected={expected_obs_dim}"
                    )
            if args.controller_mode == "constant":
                print(
                    f"[DEMO] controller=FrontRearCooperativeController+Constant middle_throttle={args.demo_throttle} "
                    f"middle_steering={args.demo_steering} middle_brake={args.demo_brake} metadata={getattr(demo_controller, 'metadata', {})}"
                )
            else:
                print(
                    f"[DEMO] controller=FrontRearCooperativeController+Actor checkpoint={args.actor_checkpoint} "
                    f"actor_device={args.actor_device} metadata={getattr(demo_controller, 'metadata', {})}"
                )
            for step_idx in range(args.step_count):
                obs_dict, reward, terminated, truncated, info = env.step_with_controller(demo_controller)
                if tracking_recorder is not None:
                    tracking_recorder.add_step(step_idx, info)
                print(f"[DEMO] step={step_idx} done={terminated} truncated={truncated}")
                print_info_summary(info)
                if args.print_tracking_debug:
                    print_tracking_debug(info)
                if args.print_obs_debug:
                    print_obs_block_summary(obs_dict, cfg.obs, agent_index=args.inspect_agent_index)
                if args.plot_road or args.plot_all_observation_points:
                    env.render_debug_markers(
                        plot_road=args.plot_road,
                        plot_observation_points=args.plot_all_observation_points,
                        plot_z=args.plot_z,
                        road_thickness=3.0,
                        point_size=args.point_size,
                        duration=args.plot_duration,
                        is_persistent=True,
                        clear_existing=True,
                    )
                if args.plot_observation_points and not args.plot_all_observation_points:
                    env.plot_agent_observation_points(
                        agent_index=args.inspect_agent_index,
                        plot_z=args.plot_z,
                        point_size=args.point_size,
                        duration=args.plot_duration,
                        is_persistent=True,
                        clear_existing=not args.plot_road,
                    )
        if tracking_recorder is not None:
            log_path = tracking_recorder.save()
            print(f"[DEMO] tracking log saved to {log_path}")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
