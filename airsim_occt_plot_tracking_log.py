import argparse
import json
from pathlib import Path


def load_log(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def ensure_out_dir(log_path, out_dir=None):
    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        out_dir = Path(log_path).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_vehicle_series(log_data, vehicle_name):
    steps = []
    s_cur = []
    s_tgt = []
    e_lat = []
    e_v = []
    v_cur = []
    v_ref = []
    steer = []
    delta_des = []
    throttle = []
    brake = []
    pose_x = []
    pose_y = []
    target_x = []
    target_y = []
    for item in log_data.get("steps", []):
        info = item.get("info", {})
        debug = info.get("controller_debug", {}).get(vehicle_name)
        if debug is None:
            continue
        steps.append(item["step"])
        s_cur.append(debug.get("current_s"))
        s_tgt.append(debug.get("target_s"))
        e_lat.append(debug.get("lateral_error"))
        e_v.append(debug.get("speed_error"))
        v_cur.append(debug.get("current_speed"))
        v_ref.append(debug.get("reference_speed"))
        steer.append(debug.get("steering_cmd"))
        delta_des.append(debug.get("delta_des"))
        throttle.append(debug.get("throttle_cmd"))
        brake.append(debug.get("brake_cmd"))
        pose = info.get("pose_map_xy", {}).get(vehicle_name)
        pose_x.append(None if pose is None else pose[0])
        pose_y.append(None if pose is None else pose[1])
        target_pt = debug.get("target_point_map")
        if target_pt is None:
            target_x.append(None)
            target_y.append(None)
        else:
            target_x.append(target_pt[0])
            target_y.append(target_pt[1])
    return {
        "steps": steps,
        "s_cur": s_cur,
        "s_tgt": s_tgt,
        "e_lat": e_lat,
        "e_v": e_v,
        "v_cur": v_cur,
        "v_ref": v_ref,
        "steer": steer,
        "delta_des": delta_des,
        "throttle": throttle,
        "brake": brake,
        "pose_x": pose_x,
        "pose_y": pose_y,
        "target_x": target_x,
        "target_y": target_y,
    }


def plot_vehicle_series(vehicle_name, series, out_dir):
    import matplotlib.pyplot as plt

    if not series["steps"]:
        return []
    saved = []

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.plot(series["steps"], series["s_cur"], label="s_current")
    ax.plot(series["steps"], series["s_tgt"], label="s_target")
    ax.set_title(f"{vehicle_name} s tracking")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(series["steps"], series["e_lat"])
    ax.set_title(f"{vehicle_name} lateral error")

    ax = axes[1, 0]
    ax.plot(series["steps"], series["v_cur"], label="v_current")
    ax.plot(series["steps"], series["v_ref"], label="v_ref")
    ax.set_title(f"{vehicle_name} speed")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(series["steps"], series["steer"], label="steer_cmd")
    ax.plot(series["steps"], series["delta_des"], label="delta_des")
    ax.set_title(f"{vehicle_name} steering")
    ax.legend()

    ax = axes[2, 0]
    ax.plot(series["steps"], series["throttle"], label="throttle")
    ax.plot(series["steps"], series["brake"], label="brake")
    ax.set_title(f"{vehicle_name} longitudinal command")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(series["steps"], series["e_v"])
    ax.set_title(f"{vehicle_name} speed error")

    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_tracking_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(series["pose_x"], series["pose_y"], label="vehicle_pose")
    if any(v is not None for v in series["target_x"]):
        ax.plot(series["target_x"], series["target_y"], label="target_point")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{vehicle_name} map trajectory")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_trajectory_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Plot front/rear tracking logs")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    log_data = load_log(args.log_file)
    out_dir = ensure_out_dir(args.log_file, args.out_dir)

    controller_debug_keys = []
    for step in log_data.get("steps", []):
        controller_debug_keys.extend(step.get("info", {}).get("controller_debug", {}).keys())
    vehicle_names = sorted(set(controller_debug_keys))

    saved_files = []
    for vehicle_name in vehicle_names:
        series = extract_vehicle_series(log_data, vehicle_name)
        saved_files.extend(plot_vehicle_series(vehicle_name, series, out_dir))

    print(f"[PLOT] saved {len(saved_files)} figures to {out_dir}")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()
