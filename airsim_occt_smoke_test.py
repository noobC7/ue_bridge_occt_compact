import argparse
import time
from typing import List

import numpy as np

import setup_vsim

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_config import EnvConfig, VehicleConfig


def build_env_config(host: str, port: int, vehicle_names: List[str]) -> EnvConfig:
    vehicle_cfgs = [
        VehicleConfig(
            vehicle_name=name,
            length=3.82,
            width=1.5,
            l_f=1.17,
            l_r=1.15,
        )
        for name in vehicle_names
    ]
    return EnvConfig(
        host=host,
        port=port,
        vehicle_configs=vehicle_cfgs,
    )


def format_state_line(state) -> str:
    pose = np.round(state.pose_world_xy, 3).tolist()
    vel = np.round(state.vel_world_xy, 3).tolist()
    acc = np.round(state.acc_world_xy, 3).tolist()
    gps = None if state.gps_lat_lon_alt is None else np.round(state.gps_lat_lon_alt, 6).tolist()
    return (
        f"vehicle={state.vehicle_name} "
        f"t={state.timestamp:.3f} "
        f"pose_xy={pose} yaw={state.yaw_world:.3f} z={state.z_world:.3f} "
        f"vel_xy={vel} acc_xy={acc} yaw_rate={state.yaw_rate:.3f} "
        f"gps={gps}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for UE AirSim state reception")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument(
        "--vehicles",
        nargs="*",
        default=None,
        help="Vehicle names in UE/AirSim, e.g. vehicle0 vehicle1; omitted means auto-discover",
    )
    parser.add_argument("--count", type=int, default=3, help="Number of polling rounds")
    parser.add_argument("--interval", type=float, default=0.5, help="Sleep between polling rounds")
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Enable API control for listed vehicles before polling",
    )
    args = parser.parse_args()

    cfg = build_env_config(args.host, args.port, args.vehicles or [])
    io = AirSimIO(cfg)

    print("[SMOKE] Connecting to AirSim...")
    io.connect()

    vehicle_names = args.vehicles or io.list_vehicles()
    if not vehicle_names:
        raise RuntimeError("No vehicles returned by AirSim listVehicles()")
    print(f"[SMOKE] vehicles={vehicle_names}")

    if args.enable_api:
        print(f"[SMOKE] Enabling API control for: {vehicle_names}")
        io.enable_api(vehicle_names)

    print("[SMOKE] Polling vehicle states...")
    for step in range(args.count):
        print(f"[SMOKE] round={step}")
        states = io.read_all(vehicle_names)
        for state in states:
            print(format_state_line(state))
        if step != args.count - 1:
            time.sleep(args.interval)

    print("[SMOKE] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
