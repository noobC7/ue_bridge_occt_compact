import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROAD_TYPE_BY_ID = {
    0: "roundabout",
    1: "roundabout",
    2: "right_angle_turn",
    3: "right_angle_turn",
    4: "s_curve",
    5: "s_curve",
}

ROAD_FIXED_TRAJECTORY_ZOOM = {
    0: {"source_x": (-85.0, -60.0), "source_y": (60.0, 80.0), "inset_center": (-110.0, 90.0), "scale": 2.0},
    1: {"source_x": (-90.0, -60.0), "source_y": (60.0, 85.0), "inset_center": (-120.0, 90.0), "scale": 2.0},
    2: {"source_x": (45.0, 65.0), "source_y": (-25.0, -15.0), "inset_center": (-55.0, 15.0), "scale": 2.0},
    3: {"source_x": (50.0, 65.0), "source_y": (-45.0, -25.0), "inset_center": (95.0, -35.0), "scale": 2.0},
    4: [
        {"source_x": (65.0, 80.0), "source_y": (-35.0, -10.0), "inset_center": (40.0, -50.0), "scale": 2.0},
        {"source_x": (45.0, 65.0), "source_y": (-15.0, -5.0), "inset_center": (80.0, 10.0), "scale": 2.0},
    ],
    5: {"source_x": (40.0, 50.0), "source_y": (-30.0, 0.0), "inset_center": (70.0, -50.0), "scale": 2.0},
}

FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf"
font_prop_chinese = (
    fm.FontProperties(fname=FONT_PATH, size=7)
    if Path(FONT_PATH).exists()
    else fm.FontProperties(size=7)
)
font_size_label = 7
font_size_tick = 6
font_size_legend = 6
font_size_title = 8

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.titlepad"] = 4
plt.rcParams["axes.labelpad"] = 2
plt.rcParams["axes.grid"] = False

PLOT_METHOD_LINE_PALETTES = {
    "longitudinal_error": ["#8EEA84", "#2d8236", "#613995"],
    "lateral_error": ["#3e0466", "#d9576e", "#f9e826"],
    "hinge_status": ["#3e0466", "#d9576e", "#f9e826"],
    "speed": ["#08306b", "#2c81c9", "#9ac9e3"],
    "acceleration": ["#FD2D2D", "#f7af43", "#add929"],
    "steering_angle": ["#0dcd80", "#1FA0B6", "#1632bb"],
    "trajectory": ["#2d5a70", "#41b399", "#9fd799"],
}

_ROAD_BOUNDARY_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}


def load_log(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_out_dir(log_path: Union[Path, str], out_dir: Optional[str] = None) -> Path:
    if out_dir is not None:
        output_dir = Path(out_dir)
    else:
        output_dir = Path(log_path).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _style_cn_axes(
    ax,
    *,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: str = "步数",
    show_legend: bool = True,
) -> None:
    if title:
        ax.set_title(title, fontproperties=font_prop_chinese, fontsize=font_size_title)
    if y_label:
        ax.set_ylabel(y_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
    if x_label:
        ax.set_xlabel(x_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax.tick_params(
        axis="both",
        labelsize=font_size_tick,
        pad=1,
        direction="in",
        top=False,
        right=False,
        labelfontfamily="Times New Roman",
    )
    ax.margins(x=0)
    if show_legend:
        ax.legend(
            loc="best",
            fontsize=font_size_legend,
            prop=font_prop_chinese,
            handlelength=1.8,
            borderpad=0.2,
            labelspacing=0.2,
        )


def _save_pdf_figure(fig, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close(fig)
    return output_path


def _vehicle_display_name(vehicle_name: str) -> str:
    suffix = "".join(ch for ch in str(vehicle_name) if ch.isdigit())
    if suffix:
        return f"车辆{suffix}"
    return str(vehicle_name).replace("vehicle", "车辆")


def _palette(name: str, count: int) -> List[str]:
    palette = PLOT_METHOD_LINE_PALETTES.get(name, ["#1f77b4", "#ff7f0e", "#2ca02c"])
    if len(palette) >= count:
        return palette[:count]
    return [palette[idx % len(palette)] for idx in range(count)]


def resolve_log_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if (input_path / "tracking_log.json").exists():
        return sorted(input_path.glob("tracking_log*.json"))
    return sorted(input_path.glob("*/tracking_log*.json"))


def infer_run_metadata(log_path: Path, log_data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = log_data.get("metadata", {})
    run_name = log_path.parent.name
    if log_path.stem != "tracking_log":
        run_name = f"{run_name}/{log_path.stem}"
    method = metadata["method"]
    road_id = metadata["road_env_index"]
    road_type = ROAD_TYPE_BY_ID.get(int(road_id), "other") if road_id is not None else "unknown"
    road_metadata = metadata.get("road_metadata", {})
    return {
        "run_name": run_name,
        "method": str(method).strip().lower(),
        "road_id": int(road_id) if road_id is not None else None,
        "road_type": road_type,
        "map_name": road_metadata.get("map_name"),
        "path_ids": road_metadata.get("path_ids"),
        "road_s_max": road_metadata.get("s_max"),
        "vehicle_names": metadata["vehicles"],
    }


def _required_info_keys_for_method(method: str) -> List[str]:
    common_keys = [
        "s",
        "pose_map_xy",
        "projection_point_map",
        "closest_center_map",
        "distance_to_ref",
        "target_agent_s",
        "hinge_target_speed",
        "speed",
        "yaw_map",
        "agent_hinge_status",
        "hinge_ready_status",
        "occt_state",
        "controller_compute_time_ms",
    ]
    if method in {"marl", "mppi"}:
        return common_keys + ["actor_debug"]
    if method == "pid":
        return common_keys + ["controller_debug"]
    raise ValueError(f"Unsupported method '{method}' in log validation")


def validate_new_log_schema(log_path: Path, log_data: Dict[str, Any]) -> None:
    metadata = log_data.get("metadata", {})
    required_metadata_keys = ["method", "road_env_index", "vehicles", "algorithm_config"]
    missing_metadata = [key for key in required_metadata_keys if key not in metadata]
    if missing_metadata:
        raise KeyError(
            f"{log_path} is incompatible with the current analysis script. "
            f"Missing metadata keys: {missing_metadata}"
        )

    steps = _extract_nonnegative_steps(log_data)
    if not steps:
        raise ValueError(f"{log_path} does not contain any nonnegative step entries")

    method = str(metadata["method"]).strip().lower()
    sample_info = steps[0].get("info", {})
    required_info_keys = _required_info_keys_for_method(method)
    missing_info = [key for key in required_info_keys if key not in sample_info]
    if missing_info:
        raise KeyError(
            f"{log_path} is incompatible with the current analysis script. "
            f"Missing per-step info keys: {missing_info}"
        )

    middle_vehicle_names = _middle_vehicle_names(metadata["vehicles"])
    if method in {"marl", "mppi"}:
        missing_actor_debug = [name for name in middle_vehicle_names if name not in sample_info["actor_debug"]]
        if missing_actor_debug:
            raise KeyError(
                f"{log_path} is incompatible with the current analysis script. "
                f"Missing actor_debug for vehicles: {missing_actor_debug}"
            )
    if method == "pid":
        missing_controller_debug = [name for name in middle_vehicle_names if name not in sample_info["controller_debug"]]
        if missing_controller_debug:
            raise KeyError(
                f"{log_path} is incompatible with the current analysis script. "
                f"Missing controller_debug for vehicles: {missing_controller_debug}"
            )


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _safe_min(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.min(np.asarray(values, dtype=np.float64)))


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _iter_hinge_opportunities(hinge_ready_series: Sequence[bool], hinged_series: Sequence[bool]):
    ready = np.asarray(hinge_ready_series, dtype=np.bool_)
    hinged = np.asarray(hinged_series, dtype=np.bool_)
    if ready.shape != hinged.shape:
        raise ValueError("hinge_ready_series and hinged_series must have the same shape")
    step = 0
    while step < ready.shape[0]:
        if not bool(ready[step]):
            step += 1
            continue
        segment_start = step
        while step + 1 < ready.shape[0] and bool(ready[step + 1]):
            step += 1
        segment_end = step
        success_index = None
        for candidate in range(segment_start, segment_end + 1):
            if bool(hinged[candidate]):
                success_index = candidate
                break
        yield segment_start, segment_end, success_index
        step += 1


def _reconstruct_estimated_delta(act_delta: Sequence[Any], algorithm_config: Dict[str, Any]) -> List[float]:
    control_cfg = algorithm_config.get("control", {})
    dt = float(control_cfg.get("dt", 0.05))
    tau = float(control_cfg.get("steering_estimation_time_constant", 0.12))
    max_rate = float(control_cfg.get("steering_estimation_max_rate", 1.5707963268))
    max_angle = float(
        control_cfg.get(
            "steering_estimation_max_angle",
            control_cfg.get("max_steering_angle", 0.6108652382),
        )
    )
    tau = max(tau, 1e-6)
    delta_est = 0.0
    estimated = []
    for value in act_delta:
        if value is None:
            estimated.append(np.nan)
            continue
        target_delta = float(np.clip(float(value), -max_angle, max_angle))
        delta_dot_cmd = (target_delta - delta_est) / tau
        delta_dot = float(np.clip(delta_dot_cmd, -max_rate, max_rate))
        delta_est = float(np.clip(delta_est + delta_dot * dt, -max_angle, max_angle))
        estimated.append(delta_est)
    return estimated


def _extract_nonnegative_steps(log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    for item in log_data.get("steps", []):
        step_idx = int(item.get("step", -1))
        if step_idx < 0:
            continue
        steps.append(item)
    return steps


def _middle_vehicle_names(vehicle_names: Sequence[str]) -> List[str]:
    if len(vehicle_names) <= 2:
        return []
    return list(vehicle_names[1:-1])


def _extract_series(info_steps: List[Dict[str, Any]], key: str, vehicle_name: str) -> List[float]:
    values = []
    for item in info_steps:
        info = item.get("info", {})
        value = info.get(key, {}).get(vehicle_name)
        values.append(value)
    return values


def _compute_hinge_distance(info: Dict[str, Any], vehicle_names: Sequence[str], vehicle_name: str) -> float:
    hinge_distance_map = info.get("hinge_distance", {})
    logged_value = hinge_distance_map.get(vehicle_name)
    if logged_value is not None:
        return float(logged_value)

    if len(vehicle_names) <= 1:
        raise KeyError("cannot reconstruct hinge_distance with fewer than two vehicles")
    if vehicle_name not in vehicle_names:
        raise KeyError(f"vehicle '{vehicle_name}' is not present in metadata['vehicles']")

    pose_map = info.get("pose_map_xy", {})
    center_map = info.get("closest_center_map", {})
    front_vehicle = vehicle_names[0]
    rear_vehicle = vehicle_names[-1]
    missing_keys = []
    if vehicle_name not in pose_map or pose_map.get(vehicle_name) is None:
        missing_keys.append(f"pose_map_xy.{vehicle_name}")
    if front_vehicle not in center_map or center_map.get(front_vehicle) is None:
        missing_keys.append(f"closest_center_map.{front_vehicle}")
    if rear_vehicle not in center_map or center_map.get(rear_vehicle) is None:
        missing_keys.append(f"closest_center_map.{rear_vehicle}")
    if missing_keys:
        raise KeyError(
            "cannot reconstruct hinge_distance because required fields are missing: "
            + ", ".join(missing_keys)
        )

    ratio = float(vehicle_names.index(vehicle_name)) / float(len(vehicle_names) - 1)
    pose_xy = np.asarray(pose_map[vehicle_name], dtype=np.float32)
    front_center_xy = np.asarray(center_map[front_vehicle], dtype=np.float32)
    rear_center_xy = np.asarray(center_map[rear_vehicle], dtype=np.float32)
    hinge_target_xy = front_center_xy + ratio * (rear_center_xy - front_center_xy)
    return float(np.linalg.norm(pose_xy - hinge_target_xy))


def _valid_xy_points(x_values: Sequence[Any], y_values: Sequence[Any]) -> np.ndarray:
    points = []
    for x_val, y_val in zip(x_values, y_values):
        if x_val is None or y_val is None:
            continue
        x_float = float(x_val)
        y_float = float(y_val)
        if np.isnan(x_float) or np.isnan(y_float):
            continue
        points.append((x_float, y_float))
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _load_road_boundaries(log_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    metadata = log_data.get("metadata", {})
    algo_cfg = metadata.get("algorithm_config", {})
    map_cfg = algo_cfg.get("map", {})
    vehicle_cfg = algo_cfg.get("vehicle", {})
    road_metadata = metadata.get("road_metadata", {})
    map_dir = metadata.get("map_dir")
    selected_path_index = road_metadata.get("selected_path_index", metadata.get("road_env_index"))
    if map_dir is None or selected_path_index is None:
        raise KeyError("metadata must include map_dir and road_metadata.selected_path_index to plot road boundaries")

    cache_key = (
        str(map_dir),
        int(selected_path_index),
        float(map_cfg.get("sample_gap", 1.0)),
        float(map_cfg.get("min_lane_width", 2.1)),
        float(map_cfg.get("min_lane_len", 70.0)),
        float(map_cfg.get("max_ref_v", 20.0 / 3.6)),
        bool(map_cfg.get("is_constant_ref_v", False)),
        float(vehicle_cfg.get("rod_len", 1.0)),
        int(len(metadata.get("vehicles", []))),
    )
    if cache_key in _ROAD_BOUNDARY_CACHE:
        return _ROAD_BOUNDARY_CACHE[cache_key]

    try:
        import torch
        from ivs_python_example.occt_map import OcctCRMap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotting road boundaries requires torch and ivs_python_example.occt_map.OcctCRMap"
        ) from exc

    road = OcctCRMap(
        batch_dim=max(1, int(selected_path_index) + 1),
        device=torch.device("cpu"),
        cr_map_dir=str(map_dir),
        sample_gap=float(map_cfg.get("sample_gap", 1.0)),
        min_lane_width=float(map_cfg.get("min_lane_width", 2.1)),
        min_lane_len=float(map_cfg.get("min_lane_len", 70.0)),
        max_ref_v=float(map_cfg.get("max_ref_v", 20.0 / 3.6)),
        is_constant_ref_v=bool(map_cfg.get("is_constant_ref_v", False)),
        rod_len=float(vehicle_cfg.get("rod_len", 1.0)),
        n_agents=max(1, int(len(metadata.get("vehicles", [])))),
    )
    left_xy = np.asarray(road.get_road_left_pts().detach().cpu().numpy(), dtype=np.float32)
    right_xy = np.asarray(road.get_road_right_pts().detach().cpu().numpy(), dtype=np.float32)
    if left_xy.ndim == 3:
        left_xy = left_xy[int(selected_path_index)]
    if right_xy.ndim == 3:
        right_xy = right_xy[int(selected_path_index)]
    left_xy = left_xy[np.isfinite(left_xy).all(axis=1)]
    right_xy = right_xy[np.isfinite(right_xy).all(axis=1)]
    _ROAD_BOUNDARY_CACHE[cache_key] = (left_xy, right_xy)
    return left_xy, right_xy


def _data_rect_to_axes_bounds(
    x_min_data: float,
    x_max_data: float,
    y_min_data: float,
    y_max_data: float,
    x_limits: Sequence[float],
    y_limits: Sequence[float],
) -> List[float]:
    x_min, x_max = float(x_limits[0]), float(x_limits[1])
    y_min, y_max = float(y_limits[0]), float(y_limits[1])
    if x_max <= x_min or y_max <= y_min:
        return [0.0, 0.0, 1.0, 1.0]
    left = (float(x_min_data) - x_min) / max(x_max - x_min, 1e-6)
    right = (float(x_max_data) - x_min) / max(x_max - x_min, 1e-6)
    bottom = (float(y_min_data) - y_min) / max(y_max - y_min, 1e-6)
    top = (float(y_max_data) - y_min) / max(y_max - y_min, 1e-6)
    left = float(np.clip(left, 0.0, 1.0))
    right = float(np.clip(right, 0.0, 1.0))
    bottom = float(np.clip(bottom, 0.0, 1.0))
    top = float(np.clip(top, 0.0, 1.0))
    return [left, bottom, max(right - left, 1e-6), max(top - bottom, 1e-6)]


def _data_point_to_axes_fraction(
    point_x: float,
    point_y: float,
    x_limits: Sequence[float],
    y_limits: Sequence[float],
) -> List[float]:
    x_min, x_max = float(x_limits[0]), float(x_limits[1])
    y_min, y_max = float(y_limits[0]), float(y_limits[1])
    if x_max <= x_min or y_max <= y_min:
        return [0.5, 0.5]
    x_norm = (float(point_x) - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (float(point_y) - y_min) / max(y_max - y_min, 1e-6)
    return [float(np.clip(x_norm, 0.0, 1.0)), float(np.clip(y_norm, 0.0, 1.0))]


def _resolve_fixed_trajectory_zoom_spec(road_id: Optional[int], swap_xy: bool) -> List[Dict[str, float]]:
    if road_id is None or int(road_id) not in ROAD_FIXED_TRAJECTORY_ZOOM:
        return []
    raw_spec = ROAD_FIXED_TRAJECTORY_ZOOM[int(road_id)]
    specs = raw_spec if isinstance(raw_spec, list) else [raw_spec]
    resolved_specs: List[Dict[str, float]] = []
    for spec in specs:
        source_x = spec["source_x"]
        source_y = spec["source_y"]
        inset_center = spec["inset_center"]
        if swap_xy:
            resolved_specs.append(
                {
                    "x_min": float(source_y[0]),
                    "x_max": float(source_y[1]),
                    "y_min": float(source_x[0]),
                    "y_max": float(source_x[1]),
                    "inset_center_x": float(inset_center[1]),
                    "inset_center_y": float(inset_center[0]),
                    "scale": float(spec["scale"]),
                }
            )
        else:
            resolved_specs.append(
                {
                    "x_min": float(source_x[0]),
                    "x_max": float(source_x[1]),
                    "y_min": float(source_y[0]),
                    "y_max": float(source_y[1]),
                    "inset_center_x": float(inset_center[0]),
                    "inset_center_y": float(inset_center[1]),
                    "scale": float(spec["scale"]),
                }
            )
    return resolved_specs


def _build_fixed_inset_bounds(
    zoom_spec: Dict[str, float],
    x_limits: Sequence[float],
    y_limits: Sequence[float],
) -> List[float]:
    source_bounds = _data_rect_to_axes_bounds(
        zoom_spec["x_min"],
        zoom_spec["x_max"],
        zoom_spec["y_min"],
        zoom_spec["y_max"],
        x_limits,
        y_limits,
    )
    center_norm = _data_point_to_axes_fraction(
        zoom_spec["inset_center_x"],
        zoom_spec["inset_center_y"],
        x_limits,
        y_limits,
    )
    inset_w = min(0.55, max(source_bounds[2] * zoom_spec["scale"], 0.16))
    inset_h = min(0.55, max(source_bounds[3] * zoom_spec["scale"], 0.16))
    margin = 0.03
    max_left = max(margin, 1.0 - margin - inset_w)
    max_bottom = max(margin, 1.0 - margin - inset_h)
    left = float(np.clip(center_norm[0] - 0.5 * inset_w, margin, max_left))
    bottom = float(np.clip(center_norm[1] - 0.5 * inset_h, margin, max_bottom))
    return [left, bottom, inset_w, inset_h]


def _draw_zoom_connectors(
    fig,
    main_ax,
    inset_ax,
    x_min_data: float,
    x_max_data: float,
    y_min_data: float,
    y_max_data: float,
    inset_bounds_axes: Sequence[float],
    color: str = "#7a7a7a",
) -> None:
    x0 = float(x_min_data)
    x1 = float(x_max_data)
    y0 = float(y_min_data)
    y1 = float(y_max_data)

    zoom_rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        linewidth=0.6,
        edgecolor=color,
        zorder=3,
    )
    main_ax.add_patch(zoom_rect)


def _compute_run_metrics(log_path: Path, log_data: Dict[str, Any]) -> Dict[str, Any]:
    metadata = infer_run_metadata(log_path, log_data)
    steps = _extract_nonnegative_steps(log_data)
    vehicle_names = metadata["vehicle_names"]
    middle_vehicle_names = _middle_vehicle_names(vehicle_names)
    road_s_max = metadata.get("road_s_max")
    if road_s_max is None and steps:
        road_s_max = steps[0].get("info", {}).get("road_s_max")

    controller_compute_time_ms = [
        float(item.get("info", {}).get("controller_compute_time_ms"))
        for item in steps
        if item.get("info", {}).get("controller_compute_time_ms") is not None
    ]

    s_error_values = []
    speed_values = []
    accel_values = []
    jerk_values = []
    steering_rate_values = []
    ttc_values = []
    hinge_ratio_values = []
    hinge_ready_ratio_values = []
    occt_ratio_values = []
    hinge_time_values = []
    hinge_speed_diff_values = []
    hinge_count = 0

    last_speed_by_vehicle: Dict[str, float] = {}
    last_acc_by_vehicle: Dict[str, float] = {}
    dt = 0.05
    if steps:
        controller_metadata = steps[0].get("info", {}).get("controller_metadata", {})
        if isinstance(controller_metadata, dict):
            maybe_dt = controller_metadata.get("dt")
            if maybe_dt is not None:
                dt = float(maybe_dt)

    terminal_event_count = 0
    for item in steps:
        info = item.get("info", {})
        s_map = info.get("s", {})
        speed_map = info.get("speed", {})
        agent_hinge_status_map = info.get("agent_hinge_status", {})
        hinge_ready_status_map = info.get("hinge_ready_status", {})
        occt_state_map = info.get("occt_state", {})

        if "done_reason" in info:
            terminal_event_count += 1

        for vehicle_name in middle_vehicle_names:
            current_s = s_map.get(vehicle_name)
            target_s = info.get("target_agent_s", {}).get(vehicle_name)
            if current_s is None or target_s is None:
                continue
            s_error_values.append(abs(float(target_s) - float(current_s)))

        if middle_vehicle_names:
            hinge_ratio_values.append(
                float(np.mean([float(bool(agent_hinge_status_map.get(name, False))) for name in middle_vehicle_names]))
            )
            hinge_ready_ratio_values.append(
                float(np.mean([float(bool(hinge_ready_status_map.get(name, False))) for name in middle_vehicle_names]))
            )
            occt_ratio_values.append(
                float(np.mean([float(bool(occt_state_map.get(name, False))) for name in middle_vehicle_names]))
            )

        for vehicle_name in middle_vehicle_names:
            speed = speed_map.get(vehicle_name)
            if speed is None:
                continue
            speed = float(speed)
            speed_values.append(speed)
            if vehicle_name in last_speed_by_vehicle:
                acc = (speed - last_speed_by_vehicle[vehicle_name]) / dt
                accel_values.append(abs(acc))
                if vehicle_name in last_acc_by_vehicle:
                    jerk_values.append(abs((acc - last_acc_by_vehicle[vehicle_name]) / dt))
                last_acc_by_vehicle[vehicle_name] = acc
            last_speed_by_vehicle[vehicle_name] = speed

        for idx, vehicle_name in enumerate(middle_vehicle_names, start=1):
            front_vehicle = vehicle_names[idx - 1]
            ego_s = s_map.get(vehicle_name)
            ego_v = speed_map.get(vehicle_name)
            front_s = s_map.get(front_vehicle)
            front_v = speed_map.get(front_vehicle)
            if None in (ego_s, ego_v, front_s, front_v):
                continue
            distance = float(front_s) - float(ego_s)
            closing_speed = float(ego_v) - float(front_v)
            if distance > 0.0 and closing_speed > 0.0:
                ttc_values.append(distance / max(closing_speed, 1e-6))

    info_steps = _extract_nonnegative_steps(log_data)
    method = metadata["method"]
    for vehicle_name in middle_vehicle_names:
        if method == "pid":
            common = extract_pid_series(log_data, vehicle_name)
        else:
            common = extract_actor_series(log_data, vehicle_name)
        hinge_target_speed_series = _extract_series(info_steps, "hinge_target_speed", vehicle_name)
        est_delta_series = common.get("est_delta", [])
        last_est_delta = None
        for steering in est_delta_series:
            if steering is None or np.isnan(steering):
                continue
            steering = float(steering)
            if last_est_delta is not None:
                steering_rate_values.append(abs((steering - last_est_delta) / dt) * 180.0 / np.pi)
            last_est_delta = steering
        for segment_start, _, success_index in _iter_hinge_opportunities(common["hinge_ready"], common["hinged"]):
            if success_index is None:
                continue
            hinge_count += 1
            hinge_time_values.append((success_index - segment_start + 1) * dt)
            if success_index < len(common["speed"]) and success_index < len(hinge_target_speed_series):
                actual_speed = common["speed"][success_index]
                target_speed = hinge_target_speed_series[success_index]
                if actual_speed is not None and target_speed is not None:
                    hinge_speed_diff_values.append(abs(float(actual_speed) - float(target_speed)))

    final_s_front = float(steps[-1].get("info", {}).get("s", {}).get(vehicle_names[0], np.nan)) if steps and vehicle_names else float("nan")
    final_s_rear = float(steps[-1].get("info", {}).get("s", {}).get(vehicle_names[-1], np.nan)) if steps and vehicle_names else float("nan")
    final_progress_ratio = (
        float(np.mean([steps[-1].get("info", {}).get("s", {}).get(name, 0.0) for name in middle_vehicle_names])) / float(road_s_max)
        if steps and middle_vehicle_names and road_s_max not in (None, 0)
        else float("nan")
    )

    return {
        "run_name": metadata["run_name"],
        "method": metadata["method"],
        "road_id": metadata["road_id"],
        "road_type": metadata["road_type"],
        "map_name": metadata["map_name"],
        "path_ids": metadata["path_ids"],
        "steps_logged": len(steps),
        "terminal_event_count": terminal_event_count,
        "road_s_max": road_s_max,
        "final_s_front": final_s_front,
        "final_s_rear": final_s_rear,
        "final_progress_ratio": final_progress_ratio,
        "s_error_mean": _safe_mean(s_error_values),
        "s_error_std": _safe_std(s_error_values),
        "s_error_max": _safe_max(s_error_values),
        "ttc_global_min": _safe_min(ttc_values),
        "ttc_mean": _safe_mean(ttc_values),
        "speed_mean": _safe_mean(speed_values),
        "speed_std": _safe_std(speed_values),
        "acc_mean": _safe_mean(accel_values),
        "acc_std": _safe_std(accel_values),
        "acc_max": _safe_max(accel_values),
        "jerk_mean": _safe_mean(jerk_values),
        "jerk_std": _safe_std(jerk_values),
        "jerk_max": _safe_max(jerk_values),
        "ste_rate_mean": _safe_mean(steering_rate_values),
        "ste_rate_std": _safe_std(steering_rate_values),
        "ste_rate_max": _safe_max(steering_rate_values),
        "hinge_time": _safe_mean(hinge_time_values),
        "hinge_time_std": _safe_std(hinge_time_values),
        "hinge_count": hinge_count,
        "hinge_spe_diff": _safe_mean(hinge_speed_diff_values),
        "hinge_spe_diff_std": _safe_std(hinge_speed_diff_values),
        "hinge_ratio_mean": _safe_mean(hinge_ratio_values),
        "hinge_ready_ratio_mean": _safe_mean(hinge_ready_ratio_values),
        "occt_ratio_mean": _safe_mean(occt_ratio_values),
        "controller_compute_time_ms_mean": _safe_mean(controller_compute_time_ms),
        "controller_compute_time_ms_std": _safe_std(controller_compute_time_ms),
        "controller_compute_time_ms_max": _safe_max(controller_compute_time_ms),
        "controller_compute_time_ms_total": float(np.nansum(controller_compute_time_ms)) if controller_compute_time_ms else float("nan"),
    }


def _group_run_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _aggregate_run_metrics(rows, group_keys=("method", "road_id", "road_type"))


def _aggregate_run_metrics(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    numeric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"run_name", "method", "road_id", "road_type", "map_name", "path_ids"}
    ] if rows else []
    summary_rows: List[Dict[str, Any]] = []

    def _sort_key(item):
        group_tuple = item[0]
        method = str(group_tuple[group_keys.index("method")]) if "method" in group_keys else ""
        road_id = group_tuple[group_keys.index("road_id")] if "road_id" in group_keys else None
        road_type = str(group_tuple[group_keys.index("road_type")]) if "road_type" in group_keys else ""
        return (method, road_type, road_id if road_id is not None else -1)

    for group_tuple, group_rows in sorted(grouped.items(), key=_sort_key):
        summary = {key: value for key, value in zip(group_keys, group_tuple)}
        summary["run_count"] = len(group_rows)
        map_names = sorted({str(row.get("map_name")) for row in group_rows if row.get("map_name")})
        road_ids = sorted({int(row["road_id"]) for row in group_rows if row.get("road_id") is not None})
        if map_names:
            summary["map_name"] = map_names[0] if len(map_names) == 1 else ",".join(map_names)
        if road_ids:
            summary["road_ids"] = road_ids
        for key in numeric_keys:
            values = [row.get(key) for row in group_rows]
            numeric_values = [float(value) for value in values if value is not None and not np.isnan(value)]
            if key.endswith("_std"):
                summary[f"{key}_mean"] = _safe_mean(numeric_values)
            else:
                summary[f"{key}_avg"] = _safe_mean(numeric_values)
                summary[f"{key}_std"] = _safe_std(numeric_values)
        summary_rows.append(summary)
    return summary_rows


def write_csv(output_path: Path, rows: List[Dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["empty"])
        return output_path

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})
    return output_path


def extract_vehicle_common_series(log_data: Dict[str, Any], vehicle_name: str) -> Dict[str, List[Any]]:
    steps = []
    speed = []
    s_vals = []
    target_s_vals = []
    yaw_vals = []
    pose_x = []
    pose_y = []
    projection_point = []
    hinge_ready = []
    hinged = []
    occt_state = []
    compute_ms = []
    hinge_target_speed = []
    for item in _extract_nonnegative_steps(log_data):
        info = item.get("info", {})
        steps.append(int(item.get("step", 0)))
        speed.append(info.get("speed", {}).get(vehicle_name))
        s_vals.append(info.get("s", {}).get(vehicle_name))
        target_s_vals.append(info.get("target_agent_s", {}).get(vehicle_name))
        yaw_vals.append(info.get("yaw_map", {}).get(vehicle_name))
        pose = info.get("pose_map_xy", {}).get(vehicle_name)
        pose_x.append(None if pose is None else pose[0])
        pose_y.append(None if pose is None else pose[1])
        projection_point.append(info.get("projection_point_map", {}).get(vehicle_name))
        hinge_ready.append(info.get("hinge_ready_status", {}).get(vehicle_name))
        hinged.append(info.get("agent_hinge_status", {}).get(vehicle_name))
        occt_state.append(info.get("occt_state", {}).get(vehicle_name))
        compute_ms.append(info.get("controller_compute_time_ms"))
        hinge_target_speed.append(info.get("hinge_target_speed", {}).get(vehicle_name))
    return {
        "steps": steps,
        "speed": speed,
        "s": s_vals,
        "target_s": target_s_vals,
        "yaw": yaw_vals,
        "pose_x": pose_x,
        "pose_y": pose_y,
        "projection_point": projection_point,
        "hinge_ready": hinge_ready,
        "hinged": hinged,
        "occt_state": occt_state,
        "compute_ms": compute_ms,
        "hinge_target_speed": hinge_target_speed,
    }


def extract_actor_series(log_data: Dict[str, Any], vehicle_name: str) -> Dict[str, List[Any]]:
    series = extract_vehicle_common_series(log_data, vehicle_name)
    act_acc = []
    meas_acc = []
    act_delta = []
    for item in _extract_nonnegative_steps(log_data):
        debug = item.get("info", {}).get("actor_debug", {}).get(vehicle_name, {})
        act_acc.append(debug.get("acceleration_mps2"))
        meas_acc.append(debug.get("measured_acc_long"))
        act_delta.append(debug.get("front_wheel_angle_rad"))
    est_delta = _reconstruct_estimated_delta(
        act_delta,
        log_data["metadata"]["algorithm_config"],
    )
    series.update(
        {
            "act_acc": act_acc,
            "meas_acc": meas_acc,
            "act_delta": act_delta,
            "est_delta": est_delta,
        }
    )
    return series


def extract_pid_series(log_data: Dict[str, Any], vehicle_name: str) -> Dict[str, List[Any]]:
    series = extract_vehicle_common_series(log_data, vehicle_name)
    cmd_delta = []
    for item in _extract_nonnegative_steps(log_data):
        debug = item.get("info", {}).get("controller_debug", {}).get(vehicle_name, {})
        cmd_delta.append(debug.get("delta_des"))
    est_delta = _reconstruct_estimated_delta(
        cmd_delta,
        log_data["metadata"]["algorithm_config"],
    )
    series.update(
        {
            "cmd_delta": cmd_delta,
            "est_delta": est_delta,
        }
    )
    return series


def plot_actor_like_series(vehicle_name: str, series: Dict[str, List[Any]], out_dir: Path, method: str) -> List[Path]:
    if not series["steps"]:
        return []
    saved = []
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 4.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(series["steps"], series["act_acc"], label="target_acc")
    ax.plot(series["steps"], series["meas_acc"], label="measured_acc")
    _style_cn_axes(ax, title=None, y_label="加速度 (m/s²)")

    ax = axes[0, 1]
    ax.plot(series["steps"], series["act_delta"], label="policy_delta")
    ax.plot(series["steps"], series["est_delta"], label="estimated_delta")
    _style_cn_axes(ax, title=None, y_label="转角 (rad)")

    ax = axes[1, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    _style_cn_axes(ax, title=None, y_label="速度 (m/s)")

    ax = axes[1, 1]
    ax.plot(series["steps"], series["yaw"], label="yaw")
    _style_cn_axes(ax, title=None, y_label="航向角 (rad)")

    ax = axes[2, 0]
    ax.plot(series["steps"], series["s"], label="s")
    _style_cn_axes(ax, title=None, y_label="弧长位置 (m)")

    ax = axes[2, 1]
    ax.plot(series["pose_x"], series["pose_y"], label="trajectory")
    ax.set_aspect("equal", adjustable="box")
    _style_cn_axes(ax, title=None, x_label="x (m)", y_label="y (m)")

    saved.append(_save_pdf_figure(fig, out_dir / f"{vehicle_name}_{method}_timeseries.pdf"))
    return saved


def plot_pid_series(vehicle_name: str, series: Dict[str, List[Any]], out_dir: Path) -> List[Path]:
    if not series["steps"]:
        return []
    saved = []
    fig, axes = plt.subplots(3, 2, figsize=(6.2, 4.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    _style_cn_axes(ax, title=None, y_label="速度 (m/s)")

    ax = axes[0, 1]
    ax.plot(series["steps"], series["s"], label="s")
    _style_cn_axes(ax, title=None, y_label="弧长位置 (m)")

    ax = axes[1, 0]
    ax.plot(series["steps"], series["cmd_delta"], label="command_delta")
    ax.plot(series["steps"], series["est_delta"], label="estimated_delta")
    _style_cn_axes(ax, title=None, y_label="转角 (rad)")

    ax = axes[1, 1]
    ax.step(series["steps"], np.asarray(series["hinge_ready"], dtype=np.float32), where="post", label="hinge_ready")
    ax.step(series["steps"], np.asarray(series["hinged"], dtype=np.float32), where="post", label="hinged")
    ax.step(series["steps"], np.asarray(series["occt_state"], dtype=np.float32), where="post", label="occt_state")
    ax.set_ylim(-0.15, 1.15)
    _style_cn_axes(ax, title=None, y_label="状态")

    ax = axes[2, 0]
    ax.plot(series["steps"], series["yaw"], label="yaw")
    _style_cn_axes(ax, title=None, y_label="航向角 (rad)")

    ax = axes[2, 1]
    ax.plot(series["pose_x"], series["pose_y"], label="trajectory")
    ax.set_aspect("equal", adjustable="box")
    _style_cn_axes(ax, title=None, x_label="x (m)", y_label="y (m)")

    saved.append(_save_pdf_figure(fig, out_dir / f"{vehicle_name}_pid_timeseries.pdf"))
    return saved


def plot_hinge_series(log_data: Dict[str, Any], vehicle_names: Sequence[str], out_dir: Path) -> List[Path]:
    middle_vehicle_names = _middle_vehicle_names(vehicle_names)
    if not middle_vehicle_names:
        return []
    steps = []
    series = {
        "hinge_ready_status": {name: [] for name in middle_vehicle_names},
        "completed_status": {name: [] for name in middle_vehicle_names},
        "hinge_distance": {name: [] for name in middle_vehicle_names},
    }
    raw_occt_state = {name: [] for name in middle_vehicle_names}
    for item in _extract_nonnegative_steps(log_data):
        info = item.get("info", {})
        if not all(
            name in info.get("hinge_ready_status", {})
            and name in info.get("occt_state", {})
            for name in middle_vehicle_names
        ):
            continue
        steps.append(int(item["step"]))
        for name in middle_vehicle_names:
            hinge_ready = bool(info["hinge_ready_status"][name])
            occt_state = bool(info["occt_state"][name])
            series["hinge_ready_status"][name].append(hinge_ready)
            raw_occt_state[name].append(occt_state)
            try:
                hinge_distance = _compute_hinge_distance(info, vehicle_names, name)
            except KeyError as exc:
                raise KeyError(
                    f"step {item.get('step')} cannot provide hinge_distance for {name}: {exc}"
                ) from exc
            series["hinge_distance"][name].append(hinge_distance)
    if not steps:
        return []

    for name in middle_vehicle_names:
        ready = np.asarray(series["hinge_ready_status"][name], dtype=np.bool_)
        occt = np.asarray(raw_occt_state[name], dtype=np.bool_)
        completed = np.zeros_like(ready, dtype=np.bool_)
        success_indices = np.flatnonzero(ready & occt)
        completion_index = int(success_indices[0]) if success_indices.size > 0 else None
        if completion_index is not None:
            completed[completion_index:] = True
        series["completed_status"][name] = completed.tolist()

    saved = []
    fig, axes = plt.subplots(len(middle_vehicle_names), 1, figsize=(3.2, 2.2), constrained_layout=True, sharex=True)
    if len(middle_vehicle_names) == 1:
        axes = [axes]
    legend_handles = None
    legend_labels = None
    right_axis_zero_ratio = 0.15 / 1.30
    left_axis_negative_ratio = right_axis_zero_ratio / (1.0 - right_axis_zero_ratio)
    for axis, vehicle_name in zip(axes, middle_vehicle_names):
        palette = _palette("hinge_status", 2)
        ready = np.asarray(series["hinge_ready_status"][vehicle_name], dtype=np.int32)
        completed = np.asarray(series["completed_status"][vehicle_name], dtype=np.int32)
        hinge_distance = np.asarray(series["hinge_distance"][vehicle_name], dtype=np.float64)
        hinge_distance = hinge_distance.copy()
        hinge_distance[completed.astype(np.bool_)] = 0.0
        axis_state = axis.twinx()

        in_ready = False
        span_start = None
        for step_value, is_ready in zip(steps, ready):
            if is_ready and not in_ready:
                in_ready = True
                span_start = step_value
            elif not is_ready and in_ready:
                axis_state.fill_betweenx([0.0, 1.0], span_start, step_value, color=palette[0], alpha=0.12, zorder=0)
                in_ready = False
                span_start = None
        if in_ready and span_start is not None:
            axis_state.fill_betweenx([0.0, 1.0], span_start, steps[-1], color=palette[0], alpha=0.12, zorder=0)

        line_distance = axis.plot(
            steps,
            hinge_distance,
            label="铰接点距离",
            linewidth=1.0,
            color="#1f77b4",
            zorder=1,
        )
        line_completed = axis_state.step(
            steps,
            completed,
            where="post",
            label="铰接完成",
            linewidth=1.4,
            color=palette[1],
            zorder=2,
        )
        axis_state.set_ylim(-0.15, 1.15)
        axis_state.set_yticks([0, 1])
        axis_state.tick_params(axis="y", colors=palette[1], labelsize=font_size_tick, direction="in", pad=1)
        hinge_distance_upper = float(np.nanmax(hinge_distance)) if hinge_distance.size > 0 else 0.0
        hinge_distance_upper = max(hinge_distance_upper * 1.05, 0.05)
        hinge_distance_lower = -left_axis_negative_ratio * hinge_distance_upper
        axis.set_ylim(hinge_distance_lower, hinge_distance_upper)
        axis.set_yticks([0.0])
        axis.tick_params(axis="y", colors="#1f77b4", labelsize=font_size_tick, direction="in", pad=1)
        axis.spines["left"].set_color("#1f77b4")
        axis_state.spines["right"].set_color(palette[1])
        _style_cn_axes(
            axis,
            title=None,
            y_label=_vehicle_display_name(vehicle_name),
            x_label="步数" if axis is axes[-1] else None,
            show_legend=False,
        )
        axis.yaxis.label.set_color("black")
        axis_state.set_ylabel("")
        if legend_handles is None:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor=palette[0], edgecolor="none", alpha=0.12, label="可铰接"),
                line_distance[0],
                line_completed[0],
            ]
            legend_labels = ["可铰接", "铰接点距离", "铰接完成"]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        fontsize=font_size_legend,
        prop=font_prop_chinese,
        bbox_to_anchor=(0.5, 1.1),
    )
    saved.append(_save_pdf_figure(fig, out_dir / "hinge_state_timeline.pdf"))
    return saved


def plot_compute_time(log_data: Dict[str, Any], out_dir: Path, run_name: str) -> List[Path]:
    steps = []
    compute_ms = []
    for item in _extract_nonnegative_steps(log_data):
        value = item.get("info", {}).get("controller_compute_time_ms")
        if value is None:
            continue
        steps.append(int(item["step"]))
        compute_ms.append(float(value))
    if not steps:
        return []
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    ax.plot(steps, compute_ms)
    _style_cn_axes(ax, title=None, y_label="计算耗时 (ms)", x_label="步数", show_legend=False)
    return [_save_pdf_figure(fig, out_dir / "controller_compute_time.pdf")]


def plot_platoon_error_curves(log_data: Dict[str, Any], out_dir: Path, run_name: str, vehicle_names: Sequence[str]) -> List[Path]:
    middle_vehicle_names = _middle_vehicle_names(vehicle_names)
    if not middle_vehicle_names:
        return []
    step_items = _extract_nonnegative_steps(log_data)
    if not step_items:
        return []
    series_by_vehicle = {
        vehicle_name: extract_vehicle_common_series(log_data, vehicle_name)
        for vehicle_name in middle_vehicle_names
    }
    steps = series_by_vehicle[middle_vehicle_names[0]]["steps"]

    long_colors = _palette("longitudinal_error", len(middle_vehicle_names))
    lat_colors = _palette("lateral_error", len(middle_vehicle_names))
    saved = []

    def _shade_hinge_intervals(ax, hinge_ready_series: Sequence[Any], color: str) -> None:
        in_segment = False
        start_step = None
        for step_value, is_hinge_ready in zip(steps, hinge_ready_series):
            if bool(is_hinge_ready) and not in_segment:
                in_segment = True
                start_step = step_value
            elif not bool(is_hinge_ready) and in_segment:
                ax.axvspan(start_step, step_value, color=color, alpha=0.08)
                in_segment = False
                start_step = None
        if in_segment and start_step is not None:
            ax.axvspan(start_step, steps[-1], color=color, alpha=0.08)

    longitudinal_fig, longitudinal_ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    lateral_fig, lateral_ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

    for color_long, color_lat, vehicle_name in zip(long_colors, lat_colors, middle_vehicle_names):
        series = series_by_vehicle[vehicle_name]
        longitudinal_error = []
        lateral_error = []
        vehicle_index = vehicle_names.index(vehicle_name)
        for step_pos, (current_s, target_s, pose_x, pose_y, proj) in enumerate(
            zip(
                series["s"],
                series["target_s"],
                series["pose_x"],
                series["pose_y"],
                series["projection_point"],
            )
        ):
            if current_s is None or target_s is None:
                longitudinal_error.append(np.nan)
            else:
                longitudinal_error.append(float(target_s) - float(current_s))
            info = step_items[step_pos].get("info", {})
            distance_to_ref = info.get("distance_to_ref", {}).get(vehicle_name)
            if distance_to_ref is not None:
                lateral_error.append(float(distance_to_ref))
            else:
                lateral_error.append(np.nan)

        longitudinal_ax.plot(steps, longitudinal_error, color=color_long, linewidth=1.6, label=_vehicle_display_name(vehicle_name))
        lateral_ax.plot(steps, lateral_error, color=color_lat, linewidth=1.6, label=_vehicle_display_name(vehicle_name))
        _shade_hinge_intervals(longitudinal_ax, series["hinge_ready"], color_long)
        _shade_hinge_intervals(lateral_ax, series["hinge_ready"], color_lat)

    _style_cn_axes(longitudinal_ax, title=None, y_label="目标弧长误差 (m)", x_label="步数", show_legend=True)
    _style_cn_axes(lateral_ax, title=None, y_label="横向误差 (m)", x_label="步数", show_legend=True)
    saved.append(_save_pdf_figure(longitudinal_fig, out_dir / "platoon_longitudinal_error.pdf"))
    saved.append(_save_pdf_figure(lateral_fig, out_dir / "platoon_lateral_error.pdf"))

    return saved


def plot_controller_compute_time_boxplot(run_rows: List[Dict[str, Any]], output_root: Path) -> List[Path]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in run_rows:
        value = row.get("controller_compute_time_ms_mean")
        if value is None or np.isnan(value):
            continue
        grouped[str(row["method"])].append(float(value))
    if not grouped:
        return []

    methods = sorted(grouped.keys())
    data = [grouped[method] for method in methods]
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.2), constrained_layout=True)
    box = ax.boxplot(data, tick_labels=methods, showfliers=True, patch_artist=True)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_yscale("log")
    _style_cn_axes(ax, title=None, y_label="控制计算时间 (ms，对数坐标)", x_label="", show_legend=False)
    return [_save_pdf_figure(fig, output_root / "controller_compute_time_boxplot.pdf")]


def plot_group_follower_series(
    log_data: Dict[str, Any],
    out_dir: Path,
    run_name: str,
    method: str,
    vehicle_names: Sequence[str],
    road_id: Optional[int],
) -> List[Path]:
    middle_vehicle_names = _middle_vehicle_names(vehicle_names)
    if not middle_vehicle_names:
        return []
    saved = []
    legend_title = None
    if method in {"marl", "mppi"}:
        series_by_vehicle = {name: extract_actor_series(log_data, name) for name in middle_vehicle_names}
    elif method == "pid":
        series_by_vehicle = {name: extract_pid_series(log_data, name) for name in middle_vehicle_names}
    else:
        series_by_vehicle = {name: extract_vehicle_common_series(log_data, name) for name in middle_vehicle_names}

    # trajectory
    swap_xy = road_id in {3, 4, 5}
    fig, ax = plt.subplots(1, 1, figsize=(3.6 if swap_xy else 3.0, 2.2), constrained_layout=True)
    all_x = []
    all_y = []
    trajectory_points: Dict[str, np.ndarray] = {}
    fig_w = 3.6 if swap_xy else 3.0
    fig_h = 2.2
    trajectory_colors = _palette("trajectory", len(middle_vehicle_names))
    left_boundary_xy, right_boundary_xy = _load_road_boundaries(log_data)
    left_boundary_plot = left_boundary_xy[:, [1, 0]] if swap_xy else left_boundary_xy
    right_boundary_plot = right_boundary_xy[:, [1, 0]] if swap_xy else right_boundary_xy
    if left_boundary_plot.size > 0:
        ax.plot(
            left_boundary_plot[:, 0],
            left_boundary_plot[:, 1],
            color="#1f77ff",
            linewidth=0.75,
            alpha=0.90,
            label="左边界",
            zorder=0,
        )
        all_x.extend(left_boundary_plot[:, 0].astype(np.float64).tolist())
        all_y.extend(left_boundary_plot[:, 1].astype(np.float64).tolist())
    if right_boundary_plot.size > 0:
        ax.plot(
            right_boundary_plot[:, 0],
            right_boundary_plot[:, 1],
            color="#d62728",
            linewidth=0.75,
            alpha=0.90,
            label="右边界",
            zorder=0,
        )
        all_x.extend(right_boundary_plot[:, 0].astype(np.float64).tolist())
        all_y.extend(right_boundary_plot[:, 1].astype(np.float64).tolist())
    for color, vehicle_name in zip(trajectory_colors, middle_vehicle_names):
        s = series_by_vehicle[vehicle_name]
        x_vals = s["pose_y"] if swap_xy else s["pose_x"]
        y_vals = s["pose_x"] if swap_xy else s["pose_y"]
        points = _valid_xy_points(x_vals, y_vals)
        trajectory_points[vehicle_name] = points
        if points.size == 0:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            label=_vehicle_display_name(vehicle_name),
            color=color,
            linewidth=0.60,
            alpha=0.92,
            zorder=1,
        )
        all_x.extend(points[:, 0].astype(np.float64).tolist())
        all_y.extend(points[:, 1].astype(np.float64).tolist())
    ax.set_aspect("equal", adjustable="box")
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        range_x = max(max_x - min_x, 1e-6)
        range_y = max(max_y - min_y, 1e-6)
        target_ratio = fig_w / fig_h
        if range_x / range_y < target_ratio:
            range_x = range_y * target_ratio
        else:
            range_y = range_x / target_ratio
        pad_x = max(range_x * 0.05, 0.5)
        pad_y = max(range_y * 0.05, 0.5)
        ax.set_xlim(center_x - 0.5 * range_x - pad_x, center_x + 0.5 * range_x + pad_x)
        ax.set_ylim(center_y - 0.5 * range_y - pad_y, center_y + 0.5 * range_y + pad_y)

        zoom_specs = _resolve_fixed_trajectory_zoom_spec(road_id, swap_xy)
        for zoom_spec in zoom_specs:
            inset_bounds = _build_fixed_inset_bounds(zoom_spec, ax.get_xlim(), ax.get_ylim())
            inset_ax = ax.inset_axes(inset_bounds)
            inset_ax.set_zorder(8.0)
            inset_ax.set_facecolor("white")
            inset_ax.patch.set_alpha(1.0)
            if left_boundary_plot.size > 0:
                inset_ax.plot(
                    left_boundary_plot[:, 0],
                    left_boundary_plot[:, 1],
                    color="#1f77ff",
                    linewidth=0.75,
                    alpha=0.90,
                    zorder=0,
                )
            if right_boundary_plot.size > 0:
                inset_ax.plot(
                    right_boundary_plot[:, 0],
                    right_boundary_plot[:, 1],
                    color="#d62728",
                    linewidth=0.75,
                    alpha=0.90,
                    zorder=0,
                )
            for color, vehicle_name in zip(trajectory_colors, middle_vehicle_names):
                points = trajectory_points.get(vehicle_name, np.zeros((0, 2), dtype=np.float32))
                if points.size == 0:
                    continue
                inset_ax.plot(points[:, 0], points[:, 1], color=color, linewidth=0.75, alpha=0.95, zorder=1)
            inset_ax.set_xlim(zoom_spec["x_min"], zoom_spec["x_max"])
            inset_ax.set_ylim(zoom_spec["y_min"], zoom_spec["y_max"])
            inset_ax.set_aspect("equal", adjustable="box")
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.tick_params(
                axis="both",
                which="both",
                left=False,
                right=False,
                bottom=False,
                top=False,
                labelleft=False,
                labelbottom=False,
            )
            for spine in inset_ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#666666")
            _draw_zoom_connectors(
                fig,
                ax,
                inset_ax,
                x_min_data=zoom_spec["x_min"],
                x_max_data=zoom_spec["x_max"],
                y_min_data=zoom_spec["y_min"],
                y_max_data=zoom_spec["y_max"],
                inset_bounds_axes=inset_bounds,
                color="#7a7a7a",
            )
    _style_cn_axes(
        ax,
        title=None,
        x_label="y (m)" if swap_xy else "x (m)",
        y_label="x (m)" if swap_xy else "y (m)",
        show_legend=True,
    )
    saved.append(_save_pdf_figure(fig, out_dir / "group_trajectory.pdf"))

    # acceleration / target_acc
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    for color, vehicle_name in zip(_palette("acceleration", len(middle_vehicle_names)), middle_vehicle_names):
        s = series_by_vehicle[vehicle_name]
        if method in {"marl", "mppi"}:
            values = s["act_acc"]
        else:
            speed = np.asarray([np.nan if v is None else float(v) for v in s["speed"]], dtype=np.float64)
            dt = float(log_data["metadata"]["algorithm_config"]["control"]["dt"])
            values = np.concatenate([[np.nan], np.diff(speed) / dt]).tolist()
        ax.plot(s["steps"], values, label=_vehicle_display_name(vehicle_name), color=color, linewidth=1.2)
    _style_cn_axes(ax, title=None, y_label="目标加速度 (m/s²)", x_label="步数", show_legend=True)
    saved.append(_save_pdf_figure(fig, out_dir / "group_target_acc.pdf"))

    # speed
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    for color, vehicle_name in zip(_palette("speed", len(middle_vehicle_names)), middle_vehicle_names):
        s = series_by_vehicle[vehicle_name]
        ax.plot(s["steps"], s["speed"], label=_vehicle_display_name(vehicle_name), color=color, linewidth=1.2)
    _style_cn_axes(ax, title=None, y_label="速度 (m/s)", x_label="步数", show_legend=True)
    saved.append(_save_pdf_figure(fig, out_dir / "group_speed.pdf"))

    # steering / policy delta
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    for color, vehicle_name in zip(_palette("steering_angle", len(middle_vehicle_names)), middle_vehicle_names):
        s = series_by_vehicle[vehicle_name]
        values = s["act_delta"] if method in {"marl", "mppi"} else s["cmd_delta"]
        ax.plot(s["steps"], values, label=_vehicle_display_name(vehicle_name), color=color, linewidth=1.2)
    _style_cn_axes(ax, title=None, y_label="目标转角 (rad)", x_label="步数", show_legend=True)
    saved.append(_save_pdf_figure(fig, out_dir / "group_policy_delta.pdf"))
    return saved


def generate_plots_for_log(log_path: Path, out_dir: Path, vehicles: Optional[Sequence[str]] = None) -> List[Path]:
    log_data = load_log(log_path)
    validate_new_log_schema(log_path, log_data)
    metadata = infer_run_metadata(log_path, log_data)
    vehicle_names = list(vehicles) if vehicles else metadata["vehicle_names"]
    middle_vehicle_names = _middle_vehicle_names(vehicle_names)
    saved_files: List[Path] = []
    method = metadata["method"]

    if method in {"marl", "mppi"}:
        target_vehicle_names = middle_vehicle_names if middle_vehicle_names else vehicle_names
        for vehicle_name in target_vehicle_names:
            series = extract_actor_series(log_data, vehicle_name)
            saved_files.extend(plot_actor_like_series(vehicle_name, series, out_dir, method=method))
    elif method == "pid":
        target_vehicle_names = middle_vehicle_names if middle_vehicle_names else vehicle_names
        for vehicle_name in target_vehicle_names:
            series = extract_pid_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))
    else:
        for vehicle_name in vehicle_names:
            series = extract_vehicle_common_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))

    saved_files.extend(plot_hinge_series(log_data, vehicle_names, out_dir))
    saved_files.extend(plot_compute_time(log_data, out_dir, metadata["run_name"]))
    saved_files.extend(plot_platoon_error_curves(log_data, out_dir, metadata["run_name"], vehicle_names))
    saved_files.extend(plot_group_follower_series(log_data, out_dir, metadata["run_name"], method, vehicle_names, metadata["road_id"]))
    return saved_files


def generate_csv_reports(log_paths: Sequence[Path], output_root: Path) -> List[Path]:
    run_rows = []
    for log_path in log_paths:
        log_data = load_log(log_path)
        validate_new_log_schema(log_path, log_data)
        run_rows.append(_compute_run_metrics(log_path, log_data))
    summary_rows = _group_run_metrics(run_rows)
    run_csv = write_csv(output_root / "tracking_metrics_runs.csv", run_rows)
    summary_csv = write_csv(output_root / "tracking_metrics_summary.csv", summary_rows)
    saved_paths = [run_csv, summary_csv]

    scenario_type_to_filename = {
        "roundabout": "tracking_metrics_roundabout.csv",
        "right_angle_turn": "tracking_metrics_right_angle_turn.csv",
        "s_curve": "tracking_metrics_s_curve.csv",
    }
    for road_type, filename in scenario_type_to_filename.items():
        scenario_rows = [row for row in run_rows if row.get("road_type") == road_type]
        scenario_summary_rows = _aggregate_run_metrics(scenario_rows, group_keys=("method", "road_type"))
        saved_paths.append(write_csv(output_root / filename, scenario_summary_rows))

    overall_rows = _aggregate_run_metrics(run_rows, group_keys=("method",))
    for row in overall_rows:
        row["road_type"] = "all"
    saved_paths.append(write_csv(output_root / "tracking_metrics_overall.csv", overall_rows))
    return saved_paths


def build_parser():
    parser = argparse.ArgumentParser(description="Generate tracking CSV reports and plots for MARL / PID / MPPI logs")
    parser.add_argument("--log-file", required=True, help="Path to a tracking_log.json file, a single run folder, or a root log folder")
    parser.add_argument("--out-dir", default=None, help="Optional override for single-log plot output")
    parser.add_argument("--vehicles", nargs="*", default=None)
    parser.add_argument("--generate-plots", action="store_true", help="Generate plots for every resolved log")
    parser.add_argument("--csv-only", action="store_true", help="Only generate CSV reports, skip plotting")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.log_file).resolve()
    log_paths = resolve_log_paths(input_path)
    if not log_paths:
        raise FileNotFoundError(f"No tracking_log.json files found under {input_path}")

    if input_path.is_file():
        output_root = input_path.parent
    elif (input_path / "tracking_log.json").exists():
        output_root = input_path
    else:
        output_root = input_path

    csv_paths = generate_csv_reports(log_paths, output_root)
    for csv_path in csv_paths:
        print(f"[CSV] saved {csv_path}")

    should_generate_plots = not args.csv_only and (args.generate_plots or input_path.is_file() or (input_path / "tracking_log.json").exists())
    if should_generate_plots:
        all_saved_plots: List[Path] = []
        for log_path in log_paths:
            plot_out_dir = ensure_out_dir(log_path, args.out_dir if len(log_paths) == 1 else None)
            all_saved_plots.extend(generate_plots_for_log(log_path, plot_out_dir, vehicles=args.vehicles))
        if input_path.is_dir() and len(log_paths) > 1:
            run_rows = []
            for log_path in log_paths:
                log_data = load_log(log_path)
                validate_new_log_schema(log_path, log_data)
                run_rows.append(_compute_run_metrics(log_path, log_data))
            all_saved_plots.extend(plot_controller_compute_time_boxplot(run_rows, output_root))
        print(f"[PLOT] saved {len(all_saved_plots)} figures")
        for path in all_saved_plots:
            print(path)


if __name__ == "__main__":
    main()
