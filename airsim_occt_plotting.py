from typing import Dict

import numpy as np

from airsim_occt_schema import Transform2D


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_selected_path_index(road, road_env_index: int = 0) -> int:
    if hasattr(road, 'batch_id'):
        batch_id = _to_numpy(road.batch_id).reshape(-1)
        if road_env_index >= len(batch_id):
            raise IndexError(f'road_env_index {road_env_index} out of range for batch_id length {len(batch_id)}')
        return int(batch_id[road_env_index])
    return int(road_env_index)


def get_selected_road_metadata(road, road_env_index: int = 0) -> Dict:
    path_index = get_selected_path_index(road, road_env_index)
    item = road.path_library[path_index]
    s = _to_numpy(item['s']).reshape(-1)
    return {
        'selected_path_index': path_index,
        'map_name': item.get('map_name'),
        'path_ids': item.get('path_ids'),
        's_max': float(s[-1]) if len(s) else 0.0,
        'num_center_pts': int(len(_to_numpy(item['center_vertices']))),
    }


def extract_selected_road_polylines(road, road_env_index: int = 0) -> Dict[str, np.ndarray]:
    path_index = get_selected_path_index(road, road_env_index)
    item = road.path_library[path_index]
    return {
        'center': _to_numpy(item['center_vertices']).astype(np.float32),
        'left': _to_numpy(item['left_vertices']).astype(np.float32),
        'right': _to_numpy(item['right_vertices']).astype(np.float32),
    }


def map_points_to_world(points_map_xy: np.ndarray, world_to_map: Transform2D) -> np.ndarray:
    return world_to_map.inverse_apply_point(points_map_xy).astype(np.float32)


def build_start_aligned_world_to_map(road, road_env_index: int = 0) -> Transform2D:
    polylines_map = extract_selected_road_polylines(road, road_env_index=road_env_index)
    center = polylines_map["center"]
    if center.shape[0] < 2:
        raise ValueError("selected road must have at least two center points for start alignment")
    start_xy = center[0]
    forward_vec = center[1] - center[0]
    heading = float(np.arctan2(forward_vec[1], forward_vec[0]))
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    rot = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
    return Transform2D(mat=rot, bias=start_xy.astype(np.float32))


def build_world_plot_lines(
    road,
    world_to_map: Transform2D,
    road_env_index: int = 0,
    plot_z: float = 0.0,
    flip_world_y: bool = False,
) -> Dict[str, np.ndarray]:
    polylines_map = extract_selected_road_polylines(road, road_env_index=road_env_index)
    outputs = {}
    for key, pts_map in polylines_map.items():
        pts_world_xy = map_points_to_world(pts_map, world_to_map)
        if flip_world_y:
            pts_world_xy = pts_world_xy.copy()
            pts_world_xy[:, 1] *= -1.0
        z_col = np.full((pts_world_xy.shape[0], 1), float(plot_z), dtype=np.float32)
        outputs[key] = np.concatenate([pts_world_xy, z_col], axis=1)
    return outputs


def build_world_plot_points(
    points_map_xy: np.ndarray,
    world_to_map: Transform2D,
    plot_z: float = 0.0,
    flip_world_y: bool = False,
) -> np.ndarray:
    pts_world_xy = map_points_to_world(points_map_xy, world_to_map)
    if flip_world_y:
        pts_world_xy = pts_world_xy.copy()
        pts_world_xy[:, 1] *= -1.0
    z_col = np.full((pts_world_xy.shape[0], 1), float(plot_z), dtype=np.float32)
    return np.concatenate([pts_world_xy, z_col], axis=1)


def plot_agent_observation_points_in_airsim(
    io,
    projection,
    world_to_map: Transform2D,
    plot_z: float = 0.0,
    point_size: float = 12.0,
    duration: float = -1.0,
    is_persistent: bool = True,
    clear_existing: bool = False,
    flip_world_y: bool = False,
) -> None:
    if clear_existing:
        io.flush_persistent_markers()
    center_pts_world = build_world_plot_points(
        projection.short_term_ref[:, :2],
        world_to_map=world_to_map,
        plot_z=plot_z,
        flip_world_y=flip_world_y,
    )
    left_pts_world = build_world_plot_points(
        projection.left_boundary_pts,
        world_to_map=world_to_map,
        plot_z=plot_z,
        flip_world_y=flip_world_y,
    )
    right_pts_world = build_world_plot_points(
        projection.right_boundary_pts,
        world_to_map=world_to_map,
        plot_z=plot_z,
        flip_world_y=flip_world_y,
    )
    io.plot_points_world(
        left_pts_world,
        color_rgba=[0.0, 0.0, 1.0, 1.0],
        size=point_size,
        duration=duration,
        is_persistent=is_persistent,
    )
    io.plot_points_world(
        right_pts_world,
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        size=point_size,
        duration=duration,
        is_persistent=is_persistent,
    )
    io.plot_points_world(
        center_pts_world,
        color_rgba=[0.0, 0.0, 0.0, 1.0],
        size=point_size,
        duration=duration,
        is_persistent=is_persistent,
    )


def plot_all_agent_observation_points_in_airsim(
    io,
    projections,
    world_to_map: Transform2D,
    plot_z: float = 0.0,
    point_size: float = 12.0,
    duration: float = -1.0,
    is_persistent: bool = True,
    clear_existing: bool = False,
    flip_world_y: bool = False,
    skip_agent_indices=None,
) -> None:
    if clear_existing:
        io.flush_persistent_markers()
    skip_agent_indices = set(skip_agent_indices or [])
    for agent_index, projection in enumerate(projections):
        if agent_index in skip_agent_indices:
            continue
        plot_agent_observation_points_in_airsim(
            io=io,
            projection=projection,
            world_to_map=world_to_map,
            plot_z=plot_z,
            point_size=point_size,
            duration=duration,
            is_persistent=is_persistent,
            clear_existing=False,
            flip_world_y=flip_world_y,
        )


def build_dashed_line_list(points_xyz: np.ndarray, dash_stride: int = 3, gap_stride: int = 2) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape [N,3], got {pts.shape}")
    if pts.shape[0] < 2:
        return pts.copy()
    line_pairs = []
    step = max(1, int(dash_stride + gap_stride))
    dash_stride = max(1, int(dash_stride))
    for start_idx in range(0, pts.shape[0] - 1, step):
        end_idx = min(start_idx + dash_stride, pts.shape[0] - 1)
        if end_idx > start_idx:
            line_pairs.append(pts[start_idx])
            line_pairs.append(pts[end_idx])
    if not line_pairs:
        line_pairs.extend([pts[0], pts[-1]])
    return np.asarray(line_pairs, dtype=np.float32)


def plot_selected_road_in_airsim(
    io,
    road,
    world_to_map: Transform2D,
    road_env_index: int = 0,
    plot_z: float = 0.0,
    thickness: float = 3.0,
    center_dash_stride: int = 3,
    center_gap_stride: int = 2,
    duration: float = -1.0,
    is_persistent: bool = True,
    clear_existing: bool = True,
    start_point_size: float = 15.0,
    flip_world_y: bool = False,
) -> Dict:
    if clear_existing:
        io.flush_persistent_markers()
    metadata = get_selected_road_metadata(road, road_env_index=road_env_index)
    world_lines = build_world_plot_lines(
        road=road,
        world_to_map=world_to_map,
        road_env_index=road_env_index,
        plot_z=plot_z,
        flip_world_y=flip_world_y,
    )
    io.plot_line_strip_world(
        world_lines["left"],
        color_rgba=[0.0, 0.0, 1.0, 1.0],
        thickness=thickness,
        duration=duration,
        is_persistent=is_persistent,
    )
    io.plot_line_strip_world(
        world_lines["right"],
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        thickness=thickness,
        duration=duration,
        is_persistent=is_persistent,
    )
    center_line_list = build_dashed_line_list(
        world_lines["center"],
        dash_stride=center_dash_stride,
        gap_stride=center_gap_stride,
    )
    io.plot_line_list_world(
        center_line_list,
        color_rgba=[0.0, 0.0, 0.0, 1.0],
        thickness=thickness,
        duration=duration,
        is_persistent=is_persistent,
    )
    io.plot_points_world(
        world_lines["center"][:1],
        color_rgba=[1.0, 1.0, 0.0, 1.0],
        size=start_point_size,
        duration=duration,
        is_persistent=is_persistent,
    )
    return metadata
