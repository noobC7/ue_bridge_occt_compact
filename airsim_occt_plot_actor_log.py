import argparse
import json
from pathlib import Path

import numpy as np


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
    est_delta = []
    target_delta = []
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
        est_delta.append(info.get("steering_feedback", {}).get(vehicle_name))
        target_delta.append(info.get("steering_target_rad", {}).get(vehicle_name))
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
        "est_delta": est_delta,
        "target_delta": target_delta,
        "speed": speed,
        "s": s_vals,
        "yaw": yaw_vals,
        "pose_x": pose_x,
        "pose_y": pose_y,
    }


def extract_logged_hinge_series(log_data, vehicle_names):
    series = {
        "steps": [],
        "agent_hinge_status": {name: [] for name in vehicle_names},
        "hinge_ready_status": {name: [] for name in vehicle_names},
        "occt_state": {name: [] for name in vehicle_names},
    }
    for item in log_data.get("steps", []):
        info = item.get("info", {})
        occt_state = info.get("occt_state", {})
        agent_hinge_status = info.get("agent_hinge_status", {})
        hinge_ready_status = info.get("hinge_ready_status", {})
        if not all(
            name in occt_state and name in agent_hinge_status and name in hinge_ready_status
            for name in vehicle_names
        ):
            continue
        series["steps"].append(int(item["step"]))
        for name in vehicle_names:
            series["occt_state"][name].append(bool(occt_state[name]))
            series["agent_hinge_status"][name].append(bool(agent_hinge_status[name]))
            series["hinge_ready_status"][name].append(bool(hinge_ready_status[name]))
    return series


def has_logged_hinge_series(log_data, vehicle_names):
    steps = log_data.get("steps", [])
    if not steps:
        return False
    info = steps[0].get("info", {})
    required_keys = ("occt_state", "agent_hinge_status", "hinge_ready_status")
    if not all(key in info for key in required_keys):
        return False
    return all(name in info["occt_state"] for name in vehicle_names)


def get_logged_vehicle_names(log_data):
    metadata = log_data.get("metadata", {})
    vehicle_names = metadata.get("vehicles", [])
    if vehicle_names:
        return list(vehicle_names)

    actor_debug_keys = []
    for step in log_data.get("steps", []):
        actor_debug_keys.extend(step.get("info", {}).get("actor_debug", {}).keys())
    return sorted(set(actor_debug_keys))


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
    ax.plot(series["steps"], series["act_delta"], label="actor_target_delta")
    if any(value is not None for value in series["target_delta"]):
        ax.plot(series["steps"], series["target_delta"], label="actuator_target_delta")
    if any(value is not None for value in series["est_delta"]):
        ax.plot(series["steps"], series["est_delta"], label="estimated_delta")
    ax.set_title(f"{vehicle_name} front wheel angle")
    ax.set_ylabel("rad")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(series["steps"], series["speed"], label="speed")
    ax.set_title(f"{vehicle_name} speed")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(series["steps"], series["yaw"])
    ax.set_title(f"{vehicle_name} yaw")

    ax = axes[2, 0]
    ax.plot(series["steps"], series["s"], label="s")
    ax.set_title(f"{vehicle_name} longitudinal progress")
    ax.legend()

    ax = axes[2, 1]
    ax.plot(series["pose_x"], series["pose_y"], label="vehicle_pose")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{vehicle_name} trajectory")
    ax.legend()

    fig.tight_layout()
    path = out_dir / f"{vehicle_name}_actor_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_hinge_series(series, out_dir: Path):
    import matplotlib.pyplot as plt

    steps = series["steps"]
    vehicle_names = list(series["agent_hinge_status"].keys())
    if not steps:
        return []

    saved = []

    fig, axes = plt.subplots(len(vehicle_names), 1, figsize=(12, 2.0 * len(vehicle_names)), sharex=True)
    if len(vehicle_names) == 1:
        axes = [axes]
    for axis, vehicle_name in zip(axes, vehicle_names):
        axis.step(
            steps,
            np.asarray(series["hinge_ready_status"][vehicle_name], dtype=np.int32),
            where="post",
            label="hinge_ready",
        )
        axis.step(
            steps,
            np.asarray(series["agent_hinge_status"][vehicle_name], dtype=np.int32),
            where="post",
            label="agent_hinged",
        )
        axis.step(
            steps,
            np.asarray(series["occt_state"][vehicle_name], dtype=np.int32),
            where="post",
            label="occt_state",
            linestyle="--",
        )
        axis.set_ylim(-0.15, 1.15)
        axis.set_yticks([0, 1])
        axis.set_title(vehicle_name)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("step")
    fig.suptitle("Hinge / OCCT state timeline")
    fig.tight_layout()
    path = out_dir / "hinge_state_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    matrices = [
        ("hinge_ready_status", "Hinge Ready"),
        ("agent_hinge_status", "Agent Hinged"),
        ("occt_state", "OCCT State"),
    ]
    for axis, (key, title) in zip(axes, matrices):
        mat = np.asarray([series[key][name] for name in vehicle_names], dtype=np.float32)
        im = axis.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_yticks(range(len(vehicle_names)))
        axis.set_yticklabels(vehicle_names)
        fig.colorbar(im, ax=axis, fraction=0.015, pad=0.02)
    axes[-1].set_xticks(np.linspace(0, len(steps) - 1, num=min(8, len(steps)), dtype=int))
    axes[-1].set_xticklabels([str(steps[idx]) for idx in axes[-1].get_xticks().astype(int)])
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    path = out_dir / "hinge_state_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def build_parser():
    parser = argparse.ArgumentParser(description="Plot actor and hinge / OCCT states from tracking logs")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--vehicles", nargs="*", default=None)
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=["actor", "hinge", "all"],
        default=["all"],
        help="Select which plot groups to generate; default is all.",
    )
    return parser


def run(args):
    log_data = load_log(args.log_file)
    out_dir = ensure_out_dir(args.log_file, args.out_dir)
    requested_modes = set(args.modes)
    if "all" in requested_modes:
        requested_modes = {"actor", "hinge"}

    vehicle_names = get_logged_vehicle_names(log_data)
    if args.vehicles:
        requested = set(args.vehicles)
        vehicle_names = [name for name in vehicle_names if name in requested]

    saved_files = []

    if "actor" in requested_modes:
        for vehicle_name in vehicle_names:
            series = extract_actor_series(log_data, vehicle_name)
            saved_files.extend(plot_actor_series(vehicle_name, series, out_dir))

    if "hinge" in requested_modes:
        if has_logged_hinge_series(log_data, vehicle_names):
            print("[PLOT] using logged hinge / OCCT states directly")
            hinge_series = extract_logged_hinge_series(log_data, vehicle_names)
            saved_files.extend(plot_hinge_series(hinge_series, out_dir))
        else:
            raise RuntimeError(
                "This log does not contain occt_state / agent_hinge_status / hinge_ready_status. "
                "Old logs are not supported by the merged plot script. Please rerun the simulation to generate a new log."
            )

    print(f"[PLOT] saved {len(saved_files)} figures to {out_dir}")
    for path in saved_files:
        print(path)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
