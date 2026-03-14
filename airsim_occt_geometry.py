from typing import Optional, Tuple, Union

import numpy as np


def wrap_angle(angle: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float32)


def transform_points_global_to_local(
    ego_xy: np.ndarray,
    ego_yaw: float,
    points_xy: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    delta = pts - np.asarray(ego_xy, dtype=np.float32)
    rot = rotation_matrix(-ego_yaw)
    return delta @ rot.T


def heading_to_local_velocity(speed: float, rel_yaw: float) -> np.ndarray:
    return np.asarray([speed * np.cos(rel_yaw), speed * np.sin(rel_yaw)], dtype=np.float32)


def vehicle_rectangle_vertices(
    center_xy: np.ndarray,
    yaw: float,
    length: float,
    width: float,
) -> np.ndarray:
    half_l = 0.5 * length
    half_w = 0.5 * width
    local = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    rot = rotation_matrix(yaw)
    return (local @ rot.T) + np.asarray(center_xy, dtype=np.float32)


def project_point_to_segment(
    point_xy: np.ndarray,
    seg_a_xy: np.ndarray,
    seg_b_xy: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    point = np.asarray(point_xy, dtype=np.float32)
    seg_a = np.asarray(seg_a_xy, dtype=np.float32)
    seg_b = np.asarray(seg_b_xy, dtype=np.float32)
    seg = seg_b - seg_a
    seg_norm_sq = float(np.dot(seg, seg))
    if seg_norm_sq <= 1e-8:
        closest = seg_a
        t = 0.0
    else:
        t = float(np.clip(np.dot(point - seg_a, seg) / seg_norm_sq, 0.0, 1.0))
        closest = seg_a + t * seg
    dist = float(np.linalg.norm(point - closest))
    return closest, t, dist


def project_point_to_polyline(
    point_xy: np.ndarray,
    polyline_xy: np.ndarray,
    polyline_s: np.ndarray,
) -> Tuple[np.ndarray, float, float, int]:
    best_point = np.asarray(polyline_xy[0], dtype=np.float32)
    best_s = float(polyline_s[0])
    best_dist = float("inf")
    best_idx = 0
    for idx in range(len(polyline_xy) - 1):
        seg_a = polyline_xy[idx]
        seg_b = polyline_xy[idx + 1]
        closest, t, dist = project_point_to_segment(point_xy, seg_a, seg_b)
        if dist < best_dist:
            ds = float(polyline_s[idx + 1] - polyline_s[idx])
            best_point = closest
            best_s = float(polyline_s[idx] + t * ds)
            best_dist = dist
            best_idx = idx
    return best_point, best_s, best_dist, best_idx


def distance_point_to_polyline(
    point_xy: np.ndarray,
    polyline_xy: np.ndarray,
    polyline_s: Optional[np.ndarray] = None,
) -> float:
    if polyline_s is None:
        polyline_s = build_arc_length(polyline_xy)
    _, _, dist, _ = project_point_to_polyline(point_xy, polyline_xy, polyline_s)
    return dist


def build_arc_length(polyline_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(polyline_xy, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((0,), dtype=np.float32)
    diffs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(diffs, axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)])


def interpolate_polyline_by_s(polyline_s: np.ndarray, polyline_val: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    s = np.asarray(polyline_s, dtype=np.float32)
    v = np.asarray(polyline_val, dtype=np.float32)
    q = np.asarray(query_s, dtype=np.float32)
    if v.ndim == 1:
        return np.interp(q, s, v).astype(np.float32)
    out = [np.interp(q, s, v[:, dim]) for dim in range(v.shape[1])]
    return np.stack(out, axis=-1).astype(np.float32)
