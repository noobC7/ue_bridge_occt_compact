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


def extract_actor_series(log_data, vehicle_name):
    steps = []
    act_acc = []
    meas_acc = []
    act_delta = []
    cmd_throttle = []
    cmd_brake = []
    cmd_steer = []
    speed = []
    s_vals = []
    yaw_vals = []
    pose_x = []
    pose_y = []
    for item in log_data.get("steps", []):
        info = item.get("info", {})
        debug = info.get("actor_debug", {}).get(vehicle_name)
        if debug is None:
            continue
        steps.append(item.get("step"))
        act_acc.append(debug.get("acceleration_mps2"))
        meas_acc.append(debug.get("measured_acc_long"))
        act_delta.append(debug.get("front_wheel_angle_rad"))
        cmd_throttle.append(debug.get("throttle_cmd"))
        cmd_brake.append(debug.get("brake_cmd"))
        cmd_steer.append(debug.get("steering_cmd"))
        speed.append(debug.get("current_speed"))
        s_vals.append(info.get("s", {}).get(vehicle_name))
        yaw_vals.append(info.get("yaw_map", {}).get(vehicle_name))
        pose = info.get("pose_map_xy", {}).get(vehicle_name)
        pose_x.append(None if pose is None else pose[0])
        pose_y.append(None if pose is None else pose[1])
    return {
        "steps": steps,
        "act_acc": act_acc,
        "meas_acc": meas_acc,
        "act_delta": act_delta,
        "cmd_throttle": cmd_throttle,
        "cmd_brake": cmd_brake,
        "cmd_steer": cmd_steer,
        "speed": speed,
        "s": s_vals,
        "yaw": yaw_vals,
        "pose_x": pose_x,
        "pose_y": pose_y,
    }


def plot_actor_series(vehicle_name, series, out_dir):
    import matplotlib.pyplot as plt

    if not series["steps"]:
        return []
    saved = []

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax = axes[0, 0]
    ax.plot(series["steps"], series["act_acc"], label="target_acc")
    ax.plot(series["steps"], series["meas_acc"], label="measured_acc")
    ax.set_title(f"{vehicle_name} actor vs measured acceleration")
    ax.set_ylabel("m/s^2")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(series["steps"], series["act_delta"])
    ax.set_title(f"{vehicle_name} actor front wheel angle")
    ax.set_ylabel("rad")

    ax = axes[1, 0]
    ax.plot(series["steps"], series["cmd_throttle"], label="throttle")
    ax.plot(series["steps"], series["cmd_brake"], label="brake")
    ax.set_title(f"{vehicle_name} low-level longitudinal command")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(series["steps"], series["cmd_steer"])
    ax.set_title(f"{vehicle_name} low-level steering command")

    ax = axes[2, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    #ax.plot(series["steps"], series["s"], label="s")
    ax.set_title(f"{vehicle_name} speed")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(series["steps"], series["yaw"])
    ax.set_title(f"{vehicle_name} yaw")

    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_actor_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(series["pose_x"], series["pose_y"], label="vehicle_pose")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{vehicle_name} actor-controlled trajectory")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_actor_trajectory_map.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Plot actor control logs")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--vehicles", nargs="*", default=None)
    args = parser.parse_args()

    log_data = load_log(args.log_file)
    out_dir = ensure_out_dir(args.log_file, args.out_dir)

    actor_debug_keys = []
    for step in log_data.get("steps", []):
        actor_debug_keys.extend(step.get("info", {}).get("actor_debug", {}).keys())
    vehicle_names = sorted(set(actor_debug_keys))
    if args.vehicles:
        requested = set(args.vehicles)
        vehicle_names = [name for name in vehicle_names if name in requested]

    saved_files = []
    for vehicle_name in vehicle_names:
        series = extract_actor_series(log_data, vehicle_name)
        saved_files.extend(plot_actor_series(vehicle_name, series, out_dir))

    print(f"[ACTOR_PLOT] saved {len(saved_files)} figures to {out_dir}")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()
