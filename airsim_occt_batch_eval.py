import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_METHOD_TO_CONFIG = {
    "marl": "configs/algorithm/default.yaml",
    "pid": "configs/algorithm/pid_baseline.yaml",
    "mppi": "configs/algorithm/mppi_baseline.yaml",
}

DEFAULT_MAP_DIR = "/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path"


def build_parser():
    parser = argparse.ArgumentParser(description="Batch-evaluate MARL / PID / MPPI on multiple roads")
    parser.add_argument("--methods", nargs="*", default=["pid", "mppi", "marl"])
    parser.add_argument("--roads", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--map-dir", default=DEFAULT_MAP_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicles", nargs="*", default=None)
    parser.add_argument("--step-count", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plot-road", dest="plot_road", action="store_true")
    parser.add_argument("--no-plot-road", dest="plot_road", action="store_false")
    parser.add_argument("--plot-marl-debug", action="store_true")
    parser.add_argument("--plot-mppi-debug", action="store_true")
    parser.add_argument("--show-log", action="store_true")
    parser.add_argument("--print-obs-debug", action="store_true")
    parser.add_argument("--print-tracking-debug", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.set_defaults(plot_road=True)
    return parser


def build_output_suffix(method: str, road_index: int, repeat_index: int, repeats: int) -> str:
    suffix = f"{method}_road{road_index}"
    return suffix


def build_run_dir(output_root: Path, method: str, road_index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"tracking_{timestamp}_{method}_road{road_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_command(args, method: str, road_index: int, repeat_index: int, run_dir: Path) -> list[str]:
    if method not in DEFAULT_METHOD_TO_CONFIG:
        raise KeyError(f"Unsupported method '{method}'. Supported methods: {sorted(DEFAULT_METHOD_TO_CONFIG)}")
    cmd = [
        args.python,
        "airsim_occt_env_demo.py",
        "--algo-config",
        DEFAULT_METHOD_TO_CONFIG[method],
        "--map-dir",
        args.map_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--road-env-index",
        str(road_index),
        "--step-count",
        str(args.step_count),
        "--output-suffix",
        build_output_suffix(method, road_index, repeat_index, args.repeats),
        "--use-output-dir-as-run-dir",
        "--output-dir",
        str(run_dir),
        "--output-filename",
        "tracking_log.json" if args.repeats == 1 else f"tracking_log_{repeat_index}.json",
    ]
    if args.vehicles:
        cmd.extend(["--vehicles", *args.vehicles])
    if not args.plot_road:
        cmd.append("--no-plot-road")
    if args.plot_marl_debug:
        cmd.append("--plot-marl-debug")
    if args.plot_mppi_debug:
        cmd.append("--plot-mppi-debug")
    if args.show_log:
        cmd.append("--show-log")
    if args.print_obs_debug:
        cmd.append("--print-obs-debug")
    if args.print_tracking_debug:
        cmd.append("--print-tracking-debug")
    return cmd


def main():
    parser = build_parser()
    args = parser.parse_args()
    output_root = Path(args.output_dir) if args.output_dir is not None else Path.cwd() / "airsim_occt_tracking_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    failures = []
    total_runs = len(args.methods) * len(args.roads) * max(int(args.repeats), 1)
    run_counter = 0
    for method in args.methods:
        for road_index in args.roads:
            run_dir = build_run_dir(output_root, method, road_index)
            for repeat_index in range(max(int(args.repeats), 1)):
                run_counter += 1
                cmd = build_command(
                    args,
                    method=method,
                    road_index=road_index,
                    repeat_index=repeat_index,
                    run_dir=run_dir,
                )
                print(f"\n[BATCH] run={run_counter}/{total_runs} method={method} road={road_index} repeat={repeat_index}")
                print("[BATCH] command:", " ".join(cmd))
                result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
                if result.returncode != 0:
                    failures.append((method, road_index, repeat_index, result.returncode))
                    print(
                        f"[BATCH] FAILED method={method} road={road_index} "
                        f"repeat={repeat_index} returncode={result.returncode}"
                    )
                    if not args.continue_on_error:
                        raise SystemExit(result.returncode)
                else:
                    print(f"[BATCH] OK method={method} road={road_index} repeat={repeat_index}")
                if run_counter < total_runs:
                    time.sleep(1.0)

    if failures:
        print("\n[BATCH] summary: failures detected")
        for method, road_index, repeat_index, returncode in failures:
            print(
                f"  method={method} road={road_index} repeat={repeat_index} "
                f"returncode={returncode}"
            )
        raise SystemExit(1)

    print("\n[BATCH] summary: all runs completed successfully")


if __name__ == "__main__":
    main()
