import argparse
from pathlib import Path

import numpy as np

import setup_vsim

from airsim_occt_airsim_io import AirSimIO
from airsim_occt_config import EnvConfig, MapConfig
from airsim_occt_env_demo import build_road, build_transform
from airsim_occt_plotting import (
    build_start_aligned_world_to_map,
    get_selected_road_metadata,
    plot_selected_road_in_airsim,
)


def build_plot_env_config(args) -> EnvConfig:
    cfg = EnvConfig(
        host=args.host,
        port=args.port,
        map_cfg=MapConfig(
            cr_map_dir=args.map_dir,
            sample_gap=args.map_sample_gap,
            min_lane_width=args.min_lane_width,
            min_lane_len=args.min_lane_len,
            max_ref_v=args.max_ref_v,
            is_constant_ref_v=args.is_constant_ref_v,
        ),
        road_env_index=args.road_env_index,
    )
    cfg.obs.n_agents = args.n_agents
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description='Plot current selected Occt road in AirSim')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=41451)
    parser.add_argument('--map-dir', required=True)
    parser.add_argument('--transform-file', default=None)
    parser.add_argument('--no-align-road-start', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--road-env-index', type=int, default=0)
    parser.add_argument('--n-agents', type=int, default=3)
    parser.add_argument('--rod-len', type=float, default=None)
    parser.add_argument('--map-sample-gap', type=float, default=1.0)
    parser.add_argument('--min-lane-width', type=float, default=2.1)
    parser.add_argument('--min-lane-len', type=float, default=70.0)
    parser.add_argument('--max-ref-v', type=float, default=20.0 / 3.6)
    parser.add_argument('--is-constant-ref-v', action='store_true')
    parser.add_argument('--plot-z', type=float, default=0.0)
    parser.add_argument('--thickness', type=float, default=3.0)
    parser.add_argument('--center-dash-stride', type=int, default=3)
    parser.add_argument('--center-gap-stride', type=int, default=2)
    parser.add_argument('--start-point-size', type=float, default=15.0)
    parser.add_argument('--duration', type=float, default=-1.0)
    parser.add_argument('--no-persistent', action='store_true')
    parser.add_argument('--no-clear', action='store_true')
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(f'map dir not found: {map_dir}')

    cfg = build_plot_env_config(args)
    print('[PLOT] Building road...')
    road = build_road(cfg, args)
    flip_world_y = False
    if args.transform_file:
        transform = build_transform(args)
    elif not args.no_align_road_start:
        transform = build_start_aligned_world_to_map(road, road_env_index=args.road_env_index)
        flip_world_y = True
        print('[PLOT] using start-aligned transform with final world-y flip: road start -> UE origin, road forward -> +X')
    else:
        transform = build_transform(args)
    metadata = get_selected_road_metadata(road, road_env_index=args.road_env_index)
    print(
        f"[PLOT] selected_path_index={metadata['selected_path_index']} "
        f"map_name={metadata['map_name']} path_ids={metadata['path_ids']} "
        f"s_max={metadata['s_max']:.3f} num_center_pts={metadata['num_center_pts']}"
    )

    io = AirSimIO(cfg)
    print('[PLOT] Connecting to AirSim...')
    io.connect()
    metadata = plot_selected_road_in_airsim(
        io=io,
        road=road,
        world_to_map=transform,
        road_env_index=args.road_env_index,
        plot_z=args.plot_z,
        thickness=args.thickness,
        center_dash_stride=args.center_dash_stride,
        center_gap_stride=args.center_gap_stride,
        duration=args.duration,
        is_persistent=not args.no_persistent,
        clear_existing=not args.no_clear,
        start_point_size=args.start_point_size,
        flip_world_y=flip_world_y,
    )
    print(
        f"[PLOT] plotted center/left/right for path_index={metadata['selected_path_index']} "
        f"map_name={metadata['map_name']}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
