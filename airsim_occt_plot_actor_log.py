import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


ROAD_TYPE_BY_ID = {
    0: "roundabout",
    1: "roundabout",
    2: "right_angle_turn",
    3: "right_angle_turn",
    4: "s_curve",
    5: "s_curve",
}

METHOD_NAME_ALIASES = {
    "actor": "marl",
}


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
    method = metadata.get("method")
    road_id = metadata.get("road_env_index")
    if method is None or road_id is None:
        parts = run_name.split("_")
        for part in parts:
            if part in {"marl", "pid", "mppi", "constant"}:
                method = part
            if part.startswith("road"):
                try:
                    road_id = int(part.replace("road", ""))
                except ValueError:
                    pass
    road_type = ROAD_TYPE_BY_ID.get(int(road_id), "other") if road_id is not None else "unknown"
    road_metadata = metadata.get("road_metadata", {})
    normalized_method = METHOD_NAME_ALIASES.get(str(method).strip().lower(), method or "unknown")
    return {
        "run_name": run_name,
        "method": normalized_method,
        "road_id": int(road_id) if road_id is not None else None,
        "road_type": road_type,
        "map_name": road_metadata.get("map_name"),
        "path_ids": road_metadata.get("path_ids"),
        "road_s_max": road_metadata.get("s_max"),
        "vehicle_names": metadata.get("vehicles") or infer_vehicle_names(log_data),
    }


def infer_vehicle_names(log_data: Dict[str, Any]) -> List[str]:
    steps = log_data.get("steps", [])
    for item in steps:
        info = item.get("info", {})
        if "s" in info and isinstance(info["s"], dict):
            return sorted(info["s"].keys())
        if "pose_map_xy" in info and isinstance(info["pose_map_xy"], dict):
            return sorted(info["pose_map_xy"].keys())
    return []


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

    last_speed_by_vehicle: Dict[str, float] = {}
    last_acc_by_vehicle: Dict[str, float] = {}
    last_steering_by_vehicle: Dict[str, float] = {}
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
        steering_feedback_map = info.get("steering_feedback", {})
        agent_hinge_status_map = info.get("agent_hinge_status", {})
        hinge_ready_status_map = info.get("hinge_ready_status", {})
        occt_state_map = info.get("occt_state", {})

        if "done_reason" in info:
            terminal_event_count += 1

        if vehicle_names:
            s_front = float(s_map.get(vehicle_names[0], 0.0))
            s_rear = float(s_map.get(vehicle_names[-1], 0.0))
            if len(vehicle_names) > 1:
                desired_gap = (s_front - s_rear) / max(len(vehicle_names) - 1, 1)
                for idx, vehicle_name in enumerate(middle_vehicle_names, start=1):
                    current_s = float(s_map.get(vehicle_name, 0.0))
                    target_s = s_front - desired_gap * idx
                    s_error_values.append(abs(target_s - current_s))

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

            steering = steering_feedback_map.get(vehicle_name)
            if steering is not None:
                steering = float(steering)
                if vehicle_name in last_steering_by_vehicle:
                    rate_deg = abs((steering - last_steering_by_vehicle[vehicle_name]) / dt) * 180.0 / np.pi
                    steering_rate_values.append(rate_deg)
                last_steering_by_vehicle[vehicle_name] = steering

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
        "hinge_ratio_mean": _safe_mean(hinge_ratio_values),
        "hinge_ready_ratio_mean": _safe_mean(hinge_ready_ratio_values),
        "occt_ratio_mean": _safe_mean(occt_ratio_values),
        "controller_compute_time_ms_mean": _safe_mean(controller_compute_time_ms),
        "controller_compute_time_ms_std": _safe_std(controller_compute_time_ms),
        "controller_compute_time_ms_max": _safe_max(controller_compute_time_ms),
        "controller_compute_time_ms_total": float(np.nansum(controller_compute_time_ms)) if controller_compute_time_ms else float("nan"),
    }


def _group_run_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["road_id"], row["road_type"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    numeric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"run_name", "method", "road_id", "road_type", "map_name", "path_ids"}
    ] if rows else []
    for (method, road_id, road_type), group_rows in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1] if item[0][1] is not None else -1)):
        summary = {
            "method": method,
            "road_id": road_id,
            "road_type": road_type,
            "run_count": len(group_rows),
            "map_name": group_rows[0].get("map_name"),
        }
        for key in numeric_keys:
            values = [row.get(key) for row in group_rows]
            numeric_values = [float(value) for value in values if value is not None and not np.isnan(value)]
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
    yaw_vals = []
    est_delta = []
    target_delta = []
    pose_x = []
    pose_y = []
    hinge_ready = []
    hinged = []
    occt_state = []
    compute_ms = []
    for item in _extract_nonnegative_steps(log_data):
        info = item.get("info", {})
        steps.append(int(item.get("step", 0)))
        speed.append(info.get("speed", {}).get(vehicle_name))
        s_vals.append(info.get("s", {}).get(vehicle_name))
        yaw_vals.append(info.get("yaw_map", {}).get(vehicle_name))
        est_delta.append(info.get("steering_feedback", {}).get(vehicle_name))
        target_delta.append(info.get("steering_target_rad", {}).get(vehicle_name))
        pose = info.get("pose_map_xy", {}).get(vehicle_name)
        pose_x.append(None if pose is None else pose[0])
        pose_y.append(None if pose is None else pose[1])
        hinge_ready.append(info.get("hinge_ready_status", {}).get(vehicle_name))
        hinged.append(info.get("agent_hinge_status", {}).get(vehicle_name))
        occt_state.append(info.get("occt_state", {}).get(vehicle_name))
        compute_ms.append(info.get("controller_compute_time_ms"))
    return {
        "steps": steps,
        "speed": speed,
        "s": s_vals,
        "yaw": yaw_vals,
        "est_delta": est_delta,
        "target_delta": target_delta,
        "pose_x": pose_x,
        "pose_y": pose_y,
        "hinge_ready": hinge_ready,
        "hinged": hinged,
        "occt_state": occt_state,
        "compute_ms": compute_ms,
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
    series.update(
        {
            "act_acc": act_acc,
            "meas_acc": meas_acc,
            "act_delta": act_delta,
        }
    )
    return series


def plot_actor_like_series(vehicle_name: str, series: Dict[str, List[Any]], out_dir: Path, method: str) -> List[Path]:
    import matplotlib.pyplot as plt

    if not series["steps"]:
        return []
    saved = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.plot(series["steps"], series["act_acc"], label="target_acc")
    ax.plot(series["steps"], series["meas_acc"], label="measured_acc")
    ax.set_title(f"{vehicle_name} {method} acceleration")
    ax.set_ylabel("m/s^2")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(series["steps"], series["act_delta"], label="policy_delta")
    ax.plot(series["steps"], series["target_delta"], label="actuator_target_delta")
    ax.plot(series["steps"], series["est_delta"], label="estimated_delta")
    ax.set_title(f"{vehicle_name} steering")
    ax.set_ylabel("rad")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    ax.set_title(f"{vehicle_name} speed")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(series["steps"], series["yaw"], label="yaw")
    ax.set_title(f"{vehicle_name} yaw")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(series["steps"], series["s"], label="s")
    ax.set_title(f"{vehicle_name} longitudinal progress")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(series["pose_x"], series["pose_y"], label="trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{vehicle_name} trajectory")
    ax.legend()

    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_{method}_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_pid_series(vehicle_name: str, series: Dict[str, List[Any]], out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    if not series["steps"]:
        return []
    saved = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    ax.set_title(f"{vehicle_name} speed")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(series["steps"], series["s"], label="s")
    ax.set_title(f"{vehicle_name} longitudinal progress")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(series["steps"], series["target_delta"], label="target_delta")
    ax.plot(series["steps"], series["est_delta"], label="estimated_delta")
    ax.set_title(f"{vehicle_name} steering tracking")
    ax.legend()

    ax = axes[1, 1]
    ax.step(series["steps"], np.asarray(series["hinge_ready"], dtype=np.float32), where="post", label="hinge_ready")
    ax.step(series["steps"], np.asarray(series["hinged"], dtype=np.float32), where="post", label="hinged")
    ax.step(series["steps"], np.asarray(series["occt_state"], dtype=np.float32), where="post", label="occt_state")
    ax.set_ylim(-0.15, 1.15)
    ax.legend()
    ax.set_title(f"{vehicle_name} hinge / occt state")

    ax = axes[2, 0]
    ax.plot(series["steps"], series["yaw"], label="yaw")
    ax.set_title(f"{vehicle_name} yaw")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(series["pose_x"], series["pose_y"], label="trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{vehicle_name} trajectory")
    ax.legend()

    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_pid_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_hinge_series(log_data: Dict[str, Any], vehicle_names: Sequence[str], out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    steps = []
    series = {
        "agent_hinge_status": {name: [] for name in vehicle_names},
        "hinge_ready_status": {name: [] for name in vehicle_names},
        "occt_state": {name: [] for name in vehicle_names},
    }
    for item in _extract_nonnegative_steps(log_data):
        info = item.get("info", {})
        if not all(
            name in info.get("agent_hinge_status", {})
            and name in info.get("hinge_ready_status", {})
            and name in info.get("occt_state", {})
            for name in vehicle_names
        ):
            continue
        steps.append(int(item["step"]))
        for name in vehicle_names:
            series["agent_hinge_status"][name].append(bool(info["agent_hinge_status"][name]))
            series["hinge_ready_status"][name].append(bool(info["hinge_ready_status"][name]))
            series["occt_state"][name].append(bool(info["occt_state"][name]))
    if not steps:
        return []

    saved = []
    fig, axes = plt.subplots(len(vehicle_names), 1, figsize=(12, 2.0 * len(vehicle_names)), sharex=True)
    if len(vehicle_names) == 1:
        axes = [axes]
    for axis, vehicle_name in zip(axes, vehicle_names):
        axis.step(steps, np.asarray(series["hinge_ready_status"][vehicle_name], dtype=np.int32), where="post", label="hinge_ready")
        axis.step(steps, np.asarray(series["agent_hinge_status"][vehicle_name], dtype=np.int32), where="post", label="hinged")
        axis.step(steps, np.asarray(series["occt_state"][vehicle_name], dtype=np.int32), where="post", label="occt_state")
        axis.set_ylim(-0.15, 1.15)
        axis.set_yticks([0, 1])
        axis.set_title(vehicle_name)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    path = out_dir / "hinge_state_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_compute_time(log_data: Dict[str, Any], out_dir: Path, run_name: str) -> List[Path]:
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, compute_ms)
    ax.set_title(f"{run_name} controller compute time")
    ax.set_xlabel("step")
    ax.set_ylabel("ms")
    fig.tight_layout()
    path = out_dir / "controller_compute_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def generate_plots_for_log(log_path: Path, out_dir: Path, vehicles: Optional[Sequence[str]] = None) -> List[Path]:
    log_data = load_log(log_path)
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
            series = extract_vehicle_common_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))
    else:
        for vehicle_name in vehicle_names:
            series = extract_vehicle_common_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))

    saved_files.extend(plot_hinge_series(log_data, vehicle_names, out_dir))
    saved_files.extend(plot_compute_time(log_data, out_dir, metadata["run_name"]))
    return saved_files


def generate_csv_reports(log_paths: Sequence[Path], output_root: Path) -> List[Path]:
    run_rows = [_compute_run_metrics(log_path, load_log(log_path)) for log_path in log_paths]
    summary_rows = _group_run_metrics(run_rows)
    run_csv = write_csv(output_root / "tracking_metrics_runs.csv", run_rows)
    summary_csv = write_csv(output_root / "tracking_metrics_summary.csv", summary_rows)
    return [run_csv, summary_csv]


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
        print(f"[PLOT] saved {len(all_saved_plots)} figures")
        for path in all_saved_plots:
            print(path)


if __name__ == "__main__":
    main()
