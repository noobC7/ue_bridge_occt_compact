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


def validate_new_log_schema(log_path: Path, log_data: Dict[str, Any]) -> None:
    metadata = log_data.get("metadata", {})
    required_metadata_keys = ["method", "road_env_index", "vehicles", "algorithm_config"]
    missing_metadata = [key for key in required_metadata_keys if key not in metadata]
    if missing_metadata:
        raise KeyError(
            f"{log_path} is not compatible with the current analysis script. "
            f"Missing metadata keys: {missing_metadata}"
        )

    steps = [item for item in log_data.get("steps", []) if int(item.get("step", -1)) >= 0]
    if not steps:
        raise ValueError(f"{log_path} does not contain any nonnegative step entries")

    sample_info = steps[0].get("info", {})
    required_info_keys = [
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
    missing_info = [key for key in required_info_keys if key not in sample_info]
    if missing_info:
        raise KeyError(
            f"{log_path} is not compatible with the current analysis script. "
            f"Missing per-step info keys: {missing_info}"
        )

    method = str(metadata["method"]).strip().lower()
    vehicles = metadata["vehicles"]
    if method in {"marl", "mppi"}:
        if "actor_debug" not in sample_info:
            raise KeyError(f"{log_path} missing actor_debug for method '{method}'")
        missing_actor_debug = [vehicle for vehicle in vehicles[1:-1] if vehicle not in sample_info["actor_debug"]]
        if missing_actor_debug:
            raise KeyError(f"{log_path} missing actor_debug entries for vehicles: {missing_actor_debug}")
    elif method == "pid":
        if "controller_debug" not in sample_info:
            raise KeyError(f"{log_path} missing controller_debug for method '{method}'")
    else:
        raise ValueError(
            f"{log_path} uses unsupported method '{metadata['method']}'. "
            "Expected one of: marl, pid, mppi."
        )


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
        terminal_vehicle_names = []
        if metadata["vehicles"]:
            terminal_vehicle_names = [metadata["vehicles"][0], metadata["vehicles"][-1]]
        missing_controller_debug = [name for name in terminal_vehicle_names if name not in sample_info["controller_debug"]]
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

    info_steps = _extract_nonnegative_steps(log_data)
    for vehicle_name in middle_vehicle_names:
        common = extract_vehicle_common_series(log_data, vehicle_name)
        hinge_target_speed_series = _extract_series(info_steps, "hinge_target_speed", vehicle_name)
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
    ax.plot(series["steps"], series["cmd_delta"], label="command_delta")
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


def plot_platoon_error_curves(log_data: Dict[str, Any], out_dir: Path, run_name: str, vehicle_names: Sequence[str]) -> List[Path]:
    import matplotlib.pyplot as plt

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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
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

    longitudinal_fig, longitudinal_ax = plt.subplots(figsize=(12, 4.5))
    lateral_fig, lateral_ax = plt.subplots(figsize=(12, 4.5))

    for color, vehicle_name in zip(colors, middle_vehicle_names):
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
            if current_s is None:
                longitudinal_error.append(np.nan)
            else:
                if target_s is None:
                    info = step_items[step_pos].get("info", {})
                    s_map = info.get("s", {})
                    front_s = s_map.get(vehicle_names[0])
                    rear_s = s_map.get(vehicle_names[-1])
                    if front_s is None or rear_s is None:
                        longitudinal_error.append(np.nan)
                    else:
                        desired_gap = (float(front_s) - float(rear_s)) / max(len(vehicle_names) - 1, 1)
                        target_s_value = float(front_s) - desired_gap * vehicle_index
                        longitudinal_error.append(target_s_value - float(current_s))
                else:
                    longitudinal_error.append(float(target_s) - float(current_s))
            info = step_items[step_pos].get("info", {})
            distance_to_ref = info.get("distance_to_ref", {}).get(vehicle_name)
            closest_center = info.get("closest_center_map", {}).get(vehicle_name)
            if distance_to_ref is not None:
                lateral_error.append(float(distance_to_ref))
            elif pose_x is None or pose_y is None:
                lateral_error.append(np.nan)
            elif closest_center is not None:
                lateral_error.append(
                    float(
                        np.linalg.norm(
                            np.asarray([pose_x, pose_y], dtype=np.float32)
                            - np.asarray(closest_center, dtype=np.float32)
                        )
                    )
                )
            else:
                lateral_error.append(np.nan)

        longitudinal_ax.plot(steps, longitudinal_error, color=color, linewidth=1.6, label=vehicle_name)
        lateral_ax.plot(steps, lateral_error, color=color, linewidth=1.6, label=vehicle_name)
        _shade_hinge_intervals(longitudinal_ax, series["hinge_ready"], color)
        _shade_hinge_intervals(lateral_ax, series["hinge_ready"], color)

    longitudinal_ax.set_title(f"{run_name} platoon longitudinal error")
    longitudinal_ax.set_xlabel("step")
    longitudinal_ax.set_ylabel("target_s - current_s (m)")
    longitudinal_ax.grid(alpha=0.25)
    longitudinal_ax.legend()
    longitudinal_fig.tight_layout()
    longitudinal_path = out_dir / "platoon_longitudinal_error.png"
    longitudinal_fig.savefig(longitudinal_path, dpi=160)
    plt.close(longitudinal_fig)
    saved.append(longitudinal_path)

    lateral_ax.set_title(f"{run_name} platoon lateral error")
    lateral_ax.set_xlabel("step")
    lateral_ax.set_ylabel("distance to ref (m)")
    lateral_ax.grid(alpha=0.25)
    lateral_ax.legend()
    lateral_fig.tight_layout()
    lateral_path = out_dir / "platoon_lateral_error.png"
    lateral_fig.savefig(lateral_path, dpi=160)
    plt.close(lateral_fig)
    saved.append(lateral_path)

    return saved


def plot_controller_compute_time_boxplot(run_rows: List[Dict[str, Any]], output_root: Path) -> List[Path]:
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=methods, showfliers=True)
    ax.set_title("Controller Compute Time Across All Scenarios")
    ax.set_ylabel("compute time (ms)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_root / "controller_compute_time_boxplot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


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
            series = extract_vehicle_common_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))
    else:
        for vehicle_name in vehicle_names:
            series = extract_vehicle_common_series(log_data, vehicle_name)
            saved_files.extend(plot_pid_series(vehicle_name, series, out_dir))

    saved_files.extend(plot_hinge_series(log_data, vehicle_names, out_dir))
    saved_files.extend(plot_compute_time(log_data, out_dir, metadata["run_name"]))
    saved_files.extend(plot_platoon_error_curves(log_data, out_dir, metadata["run_name"], vehicle_names))
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
