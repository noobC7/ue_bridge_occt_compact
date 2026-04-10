import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

import setup_vsim

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_calibration import AlignmentCalibrator
from airsim_occt_config import EnvConfig, MapConfig, VehicleConfig
from airsim_occt_controllers import (
    ActorDeploymentController,
    CenterlineMPPIController,
    CenterlinePIDController,
    ConstantLowLevelController,
    FrontRearCooperativeController,
)
from airsim_occt_fleet_registry import FleetRegistry
from airsim_occt_schema import ActorAction, Transform2D
from airsim_occt_obs_manifest import build_obs_layout
from airsim_occt_plotting import build_start_aligned_world_to_map, get_selected_road_metadata
from airsim_occt_tracking_recorder import TrackingLogRecorder, make_output_dir


def load_algorithm_config(config_path: str):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading algorithm YAML config requires `PyYAML`. Please install it in your runtime environment."
        ) from exc
    with open(config_path, "r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj)
    if not isinstance(cfg, dict):
        raise ValueError(f"Algorithm config at {config_path} must load as a mapping")
    for key in ["obs", "vehicle", "map", "control", "controller"]:
        if key not in cfg:
            raise KeyError(f"Algorithm config missing required top-level key: {key}")
    return cfg


def discover_vehicle_names(host: str, port: int):
    probe_cfg = EnvConfig(host=host, port=port)
    io = AirSimIO(probe_cfg)
    io.connect()
    vehicle_names = io.list_vehicles()
    vehicle_names.remove('cargo')
    if not vehicle_names:
        raise RuntimeError('No vehicles returned by AirSim listVehicles()')
    return vehicle_names


def build_env_config(args, algo_cfg) -> EnvConfig:
    vehicle_cfg = algo_cfg["vehicle"]
    vehicle_cfgs = [
        VehicleConfig(
            vehicle_name=name,
            length=vehicle_cfg["length"],
            width=vehicle_cfg["width"],
            l_f=vehicle_cfg["l_f"],
            l_r=vehicle_cfg["l_r"],
        )
        for name in args.vehicles
    ]
    map_algo = algo_cfg["map"]
    map_cfg = MapConfig(
        cr_map_dir=args.map_dir,
        sample_gap=map_algo["sample_gap"],
        min_lane_width=map_algo["min_lane_width"],
        min_lane_len=map_algo["min_lane_len"],
        max_ref_v=map_algo["max_ref_v"],
        is_constant_ref_v=map_algo["is_constant_ref_v"],
    )
    cfg = EnvConfig(
        host=args.host,
        port=args.port,
        vehicle_configs=vehicle_cfgs,
        map_cfg=map_cfg,
        use_sim_pause_clock=not args.no_pause,
        road_env_index=args.road_env_index,
    )
    obs_algo = algo_cfg["obs"]
    cfg.obs.sample_interval = obs_algo["sample_interval"]
    cfg.obs.boundary_offset = obs_algo["boundary_offset"]
    cfg.obs.n_points_short_term = obs_algo["n_points_short_term"]
    cfg.obs.n_points_nearing_boundary = obs_algo["n_points_nearing_boundary"]
    cfg.obs.n_stored_steps = obs_algo["n_stored_steps"]
    cfg.obs.n_observed_steps = obs_algo["n_observed_steps"]
    cfg.obs.n_nearing_agents_observed = obs_algo.get("n_nearing_agents_observed", cfg.obs.n_nearing_agents_observed)
    cfg.obs.mask_ref_v = obs_algo["mask_ref_v"]
    cfg.obs.include_hinge_info = obs_algo["include_hinge_info"]
    cfg.obs.hinge_edge_buffer = obs_algo.get("hinge_edge_buffer")
    control_algo = algo_cfg["control"]
    cfg.control.dt = control_algo["dt"]
    cfg.control.max_speed = control_algo["max_speed"]
    cfg.control.max_acceleration = control_algo["max_acceleration"]
    cfg.control.max_steering_angle = control_algo["max_steering_angle"]
    cfg.control.use_imu_acceleration = control_algo["use_imu_acceleration"]
    if "steering_estimation_time_constant" in control_algo:
        cfg.control.steering_estimation_time_constant = control_algo["steering_estimation_time_constant"]
    if "steering_estimation_max_rate" in control_algo:
        cfg.control.steering_estimation_max_rate = control_algo["steering_estimation_max_rate"]
    if "steering_estimation_max_angle" in control_algo:
        cfg.control.steering_estimation_max_angle = control_algo["steering_estimation_max_angle"]
    cfg.control.accel_throttle_gain = control_algo["accel_throttle_gain"]
    cfg.control.accel_brake_gain = control_algo["accel_brake_gain"]
    cfg.control.accel_feedback_gain = control_algo["accel_feedback_gain"]
    cfg.control.throttle_deadzone = control_algo["throttle_deadzone"]
    cfg.control.brake_deadzone = control_algo["brake_deadzone"]
    cfg.control.launch_speed_threshold = control_algo["launch_speed_threshold"]
    cfg.control.launch_accel_threshold = control_algo["launch_accel_threshold"]
    cfg.control.launch_throttle = control_algo["launch_throttle"]
    cfg.control.stanley_heading_gain = control_algo["stanley_heading_gain"]
    cfg.control.stanley_cross_track_gain = control_algo["stanley_cross_track_gain"]
    cfg.control.stanley_feedforward_gain = control_algo["stanley_feedforward_gain"]
    cfg.control.stanley_soft_speed = control_algo["stanley_soft_speed"]
    return cfg


def build_transform(args) -> Transform2D:
    if args.transform_file:
        calibrator = AlignmentCalibrator()
        return calibrator.load(args.transform_file)
    return Transform2D(mat=np.eye(2, dtype=np.float32), bias=np.zeros((2,), dtype=np.float32))


def build_road(cfg: EnvConfig, args, algo_cfg):
    try:
        import torch
        from ivs_python_example.occt_map import OcctCRMap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "build_road() requires `torch` and the map dependencies from your training env. "
            "Please run this script inside the same conda/venv used for training."
        ) from exc
    vehicle_cfg = algo_cfg["vehicle"]
    road = OcctCRMap(
        batch_dim=max(1, args.road_env_index + 1),
        device=torch.device(args.device),
        cr_map_dir=cfg.map_cfg.cr_map_dir,
        sample_gap=cfg.map_cfg.sample_gap,
        min_lane_width=cfg.map_cfg.min_lane_width,
        min_lane_len=cfg.map_cfg.min_lane_len,
        max_ref_v=cfg.map_cfg.max_ref_v,
        is_constant_ref_v=cfg.map_cfg.is_constant_ref_v,
        rod_len=vehicle_cfg["rod_len"],
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
    for block_name in layout.actor_block_order:
        dim = layout.full_block_dims[block_name]
        block = obs[cursor:cursor + dim]
        print(
            f"  {block_name} dim={dim} mean={float(block.mean()):.6f} "
            f"max_abs={float(np.abs(block).max()):.6f} head={np.round(block[:min(6, dim)], 4).tolist()}"
        )
        cursor += dim


def build_zero_actions(vehicle_names):
    return {
        vehicle_name: ActorAction(acceleration_mps2=0.0, front_wheel_angle_rad=0.0)
        for vehicle_name in vehicle_names
    }


def build_demo_controller(args, cfg, vehicle_names, algo_cfg, projector=None):
    controller_cfg = algo_cfg["controller"]
    controller_mode = controller_cfg["mode"]
    if controller_mode == "constant":
        constant_cfg = controller_cfg["constant"]
        middle_controller = ConstantLowLevelController(
            throttle=constant_cfg["throttle"],
            steering=constant_cfg["steering"],
            brake=constant_cfg["brake"],
        )
    elif controller_mode == "actor":
        actor_cfg = controller_cfg["actor"]
        middle_controller = ActorDeploymentController.from_checkpoint(
            checkpoint_path=actor_cfg["checkpoint"],
            control_cfg=cfg.control,
            vehicle_names=vehicle_names,
            device=actor_cfg["device"],
        )
    elif controller_mode == "pid":
        pid_cfg = controller_cfg.get("pid", {})
        middle_controller = CenterlinePIDController(
            registry=FleetRegistry(cfg.vehicle_configs),
            control_cfg=cfg.control,
            sample_interval=cfg.obs.sample_interval,
            platoon_position_gain=pid_cfg.get("platoon_position_gain", 0.8),
        )
    elif controller_mode == "mppi":
        mppi_cfg = controller_cfg.get("mppi", {})
        middle_controller = CenterlineMPPIController(
            registry=FleetRegistry(cfg.vehicle_configs),
            control_cfg=cfg.control,
            sample_interval=cfg.obs.sample_interval,
            projector=projector,
            device=mppi_cfg.get("device", "cpu"),
            horizon_steps=mppi_cfg.get("horizon_steps", 3),
            num_samples=mppi_cfg.get("num_samples", 256),
            param_lambda=mppi_cfg.get("lambda", 10.0),
            exploration=mppi_cfg.get("exploration", 0.1),
            debug_top_k=mppi_cfg.get("debug_top_k", 8),
        )
    else:
        raise ValueError(f"Unsupported controller_mode: {controller_mode}")
    front_cfg = controller_cfg["front_lookahead"]
    rear_cfg = controller_cfg["rear_lookahead"]
    return FrontRearCooperativeController(
        registry=FleetRegistry(cfg.vehicle_configs),
        control_cfg=cfg.control,
        sample_interval=cfg.obs.sample_interval,
        middle_controller=middle_controller,
        front_lookahead_base=front_cfg["base"],
        front_lookahead_speed_gain=front_cfg["speed_gain"],
        front_lookahead_min=front_cfg["min"],
        front_lookahead_max=front_cfg["max"],
        rear_lookahead_base=rear_cfg["base"],
        rear_lookahead_speed_gain=rear_cfg["speed_gain"],
        rear_lookahead_min=rear_cfg["min"],
        rear_lookahead_max=rear_cfg["max"],
        stanley_heading_gain=cfg.control.stanley_heading_gain,
        stanley_cross_track_gain=cfg.control.stanley_cross_track_gain,
        stanley_feedforward_gain=cfg.control.stanley_feedforward_gain,
        stanley_soft_speed=cfg.control.stanley_soft_speed,
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
            f"meas_acc={debug['measured_acc_long']:.3f} "
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


def compute_debug_marker_duration(args, dt: float) -> float:
    if args.plot_duration > 0:
        return float(args.plot_duration)
    return float(2*dt)


def maybe_print_render_time(show_render_time: bool, label: str, step_idx: int, elapsed_sec: float) -> None:
    if not show_render_time:
        return
    print(f"[RENDER] label={label} step={step_idx} cost={elapsed_sec:.6f}s ({elapsed_sec * 1000.0:.2f} ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo entry for AirSimOcctMARLEnv.reset()")
    parser.add_argument("--algo-config", default="configs/algorithm/default.yaml", help="Path to YAML file containing algorithm parameters")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicles", nargs="*", default=None)
    parser.add_argument("--map-dir", required=True, help="CommonRoad map directory used by OcctCRMap")
    parser.add_argument("--transform-file", default=None, help="Optional world->map transform json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--road-env-index", type=int, default=0)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--preview-dim", type=int, default=16)
    parser.add_argument("--inspect-agent-index", type=int, default=0)
    parser.add_argument("--plot-road", dest="plot_road", action="store_true")
    parser.add_argument("--no-plot-road", dest="plot_road", action="store_false")
    parser.add_argument("--plot-marl-debug", action="store_true")
    parser.add_argument("--plot-observation-points", action="store_true")
    parser.add_argument("--plot-all-observation-points", action="store_true")
    parser.add_argument("--plot-mppi-debug", action="store_true")
    parser.add_argument("--plot-z", type=float, default=0.0)
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--plot-duration", type=float, default=-1.0)
    parser.add_argument("--no-align-road-start", action="store_true")
    parser.add_argument("--step-count", type=int, default=2000)
    parser.add_argument("--show-log", action="store_true")
    parser.add_argument("--show-render-time", action="store_true")
    parser.add_argument("--print-obs-debug", action="store_true")
    parser.add_argument("--print-tracking-debug", action="store_true")
    parser.add_argument("--no-save-tracking-log", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-suffix", default=None)
    parser.add_argument("--output-filename", default="tracking_log.json")
    parser.add_argument("--use-output-dir-as-run-dir", action="store_true")
    parser.set_defaults(plot_road=True)
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(f"map dir not found: {map_dir}")

    algo_config_path = Path(args.algo_config)
    if not algo_config_path.is_absolute():
        algo_config_path = Path.cwd() / algo_config_path
    if not algo_config_path.exists():
        raise FileNotFoundError(f"algorithm config not found: {algo_config_path}")
    algo_cfg = load_algorithm_config(str(algo_config_path))
    print(f"[DEMO] loaded algorithm config from {algo_config_path}")

    if not args.vehicles:
        print("[DEMO] Discovering vehicles from AirSim...")
        args.vehicles = discover_vehicle_names(args.host, args.port)
        print(f"[DEMO] discovered vehicles={args.vehicles}")

    cfg = build_env_config(args, algo_cfg)
    print("[DEMO] Building road...")
    road = build_road(cfg, args, algo_cfg)
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
    initial_render_times = []
    step_render_times = []
    if (not args.no_save_tracking_log) and args.step_count > 0:
        if args.use_output_dir_as_run_dir and args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = make_output_dir(args.output_dir, prefix="tracking", suffix=args.output_suffix)
        tracking_recorder = TrackingLogRecorder(
            output_dir=output_dir,
            metadata={
                "vehicles": args.vehicles,
                "map_dir": str(map_dir),
                "road_env_index": int(args.road_env_index),
                "output_suffix": args.output_suffix,
                "method": "marl" if algo_cfg.get("controller", {}).get("mode") == "actor" else algo_cfg.get("controller", {}).get("mode"),
                "road_metadata": metadata,
                "algorithm_config_path": str(algo_config_path),
                "algorithm_config": algo_cfg,
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
        if args.show_log:
            print_info_summary(info)
        if args.print_obs_debug:
            print_obs_block_summary(obs_dict, cfg.obs, agent_index=args.inspect_agent_index)
        if args.plot_road or args.plot_observation_points or args.plot_all_observation_points:
            render_begin = time.perf_counter()
            env.io.flush_persistent_markers()
            env.render_debug_markers(
                plot_road=args.plot_road,
                plot_observation_points=args.plot_all_observation_points,
                plot_z=args.plot_z,
                road_thickness=3.0,
                point_size=args.point_size,
                duration=args.plot_duration,
                is_persistent=True,
                clear_existing=False,
            )
            if args.plot_observation_points and not args.plot_all_observation_points:
                env.plot_agent_observation_points(
                    agent_index=args.inspect_agent_index,
                    plot_z=args.plot_z,
                    point_size=args.point_size,
                    duration=args.plot_duration,
                    is_persistent=True,
                    clear_existing=False,
                )
            if args.plot_mppi_debug and algo_cfg["controller"]["mode"] == "mppi":
                mppi_debug_duration = compute_debug_marker_duration(args, cfg.control.dt)
                env.render_mppi_debug_markers(
                    getattr(demo_controller if 'demo_controller' in locals() else None, "last_mppi_debug_info", {}),
                    plot_z=-3,
                    duration=mppi_debug_duration,
                    is_persistent=False,
                    clear_existing=False,
                )
            render_elapsed = time.perf_counter() - render_begin
            initial_render_times.append(render_elapsed)
            maybe_print_render_time(args.show_render_time, "initial", -1, render_elapsed)
            print(
                f"[DEMO] rendered debug markers: plot_road={args.plot_road} "
                f"plot_all_observation_points={args.plot_all_observation_points} "
                f"plot_single_observation={args.plot_observation_points and not args.plot_all_observation_points} "
                f"plot_mppi_debug={args.plot_mppi_debug and algo_cfg['controller']['mode'] == 'mppi'}"
            )
        if args.step_count > 0:
            demo_controller = build_demo_controller(
                args,
                cfg,
                reset_info.get('vehicle_names'),
                algo_cfg,
                projector=env.projector,
            )
            demo_controller.reset()
            controller_mode = algo_cfg["controller"]["mode"]
            if controller_mode == "actor":
                current_obs_dim = next(iter(obs_dict.values())).shape[0]
                expected_obs_dim = getattr(demo_controller, "expected_obs_dim", None)
                print(f"[DEMO] actor obs_dim check current={current_obs_dim} expected={expected_obs_dim}")
                if expected_obs_dim is not None and current_obs_dim != expected_obs_dim:
                    raise ValueError(
                        f"Actor checkpoint obs_dim mismatch: current={current_obs_dim}, expected={expected_obs_dim}"
                    )
                print(f"[DEMO] controller=FrontRearCooperativeController+Actor metadata={getattr(demo_controller, 'metadata', {})}")
            else:
                print(f"[DEMO] controller=FrontRearCooperativeController+Constant metadata={getattr(demo_controller, 'metadata', {})}")
            for step_idx in range(args.step_count):
                obs_dict, reward, terminated, truncated, info = env.step_with_controller(demo_controller)
                if tracking_recorder is not None:
                    tracking_recorder.add_step(step_idx, info)
                if args.show_log:
                    print(f"[DEMO] step={step_idx} done={terminated} truncated={truncated}")
                    print_info_summary(info)
                if args.print_tracking_debug:
                    print_tracking_debug(info)
                    print_actor_debug(info)
                if args.print_obs_debug:
                    print_obs_block_summary(obs_dict, cfg.obs, agent_index=args.inspect_agent_index)
                render_begin = None
                if args.plot_all_observation_points:
                    render_begin = time.perf_counter() if render_begin is None else render_begin
                    env.render_debug_markers(
                        plot_road=False,
                        plot_observation_points=True,
                        plot_z=args.plot_z,
                        road_thickness=3.0,
                        point_size=args.point_size,
                        duration=args.plot_duration,
                        is_persistent=True,
                        clear_existing=not args.plot_road,
                    )
                if args.plot_observation_points and not args.plot_all_observation_points:
                    render_begin = time.perf_counter() if render_begin is None else render_begin
                    env.plot_agent_observation_points(
                        agent_index=args.inspect_agent_index,
                        plot_z=args.plot_z,
                        point_size=args.point_size,
                        duration=args.plot_duration,
                        is_persistent=True,
                        clear_existing=not args.plot_road,
                    )
                if args.plot_marl_debug and controller_mode == "actor":
                    render_begin = time.perf_counter() if render_begin is None else render_begin
                    marl_debug_duration = compute_debug_marker_duration(args, cfg.control.dt)
                    env.render_marl_debug_markers(
                        getattr(demo_controller, "last_actor_debug_info", {}),
                        plot_z=-3.0,
                        duration=marl_debug_duration,
                        is_persistent=False,
                        clear_existing=False,
                    )
                if args.plot_mppi_debug and controller_mode == "mppi":
                    render_begin = time.perf_counter() if render_begin is None else render_begin
                    mppi_debug_duration = compute_debug_marker_duration(args, cfg.control.dt)
                    env.render_mppi_debug_markers(
                        getattr(demo_controller, "last_mppi_debug_info", {}),
                        plot_z=-3,
                        duration=mppi_debug_duration,
                        is_persistent=False,
                        clear_existing=False,
                    )
                if render_begin is not None:
                    render_elapsed = time.perf_counter() - render_begin
                    step_render_times.append(render_elapsed)
                    maybe_print_render_time(args.show_render_time, "step", step_idx, render_elapsed)
                is_done = any(bool(value) for value in terminated.values())
                is_truncated = any(bool(value) for value in truncated.values())
                if is_done or is_truncated:
                    done_reason = info.get("done_reason", "terminated" if is_done else "truncated")
                    terminal_vehicle_names = info.get("terminal_vehicle_names", [])
                    print(
                        f"[DEMO] ending run at step={step_idx} "
                        f"reason={done_reason} terminal_vehicle_names={terminal_vehicle_names}"
                    )
                    break
        if tracking_recorder is not None:
            log_path = tracking_recorder.save(filename=args.output_filename)
            print(f"[DEMO] tracking log saved to {log_path}")
        if args.show_render_time and (initial_render_times or step_render_times):
            if initial_render_times:
                initial_np = np.asarray(initial_render_times, dtype=np.float64)
                print(
                    "[RENDER] initial_summary "
                    f"count={initial_np.size} "
                    f"mean={initial_np.mean():.6f}s "
                    f"max={initial_np.max():.6f}s "
                    f"total={initial_np.sum():.6f}s"
                )
            if step_render_times:
                step_np = np.asarray(step_render_times, dtype=np.float64)
                print(
                    "[RENDER] step_summary "
                    f"count={step_np.size} "
                    f"mean={step_np.mean():.6f}s "
                    f"max={step_np.max():.6f}s "
                    f"total={step_np.sum():.6f}s"
                )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
