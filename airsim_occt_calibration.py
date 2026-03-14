import json

import numpy as np

from airsim_occt_schema import Transform2D


class AlignmentCalibrator:
    def solve_from_correspondences(
        self,
        world_points_xy: np.ndarray,
        map_points_xy: np.ndarray,
        allow_reflection: bool = True,
    ) -> Transform2D:
        world_pts = np.asarray(world_points_xy, dtype=np.float32)
        map_pts = np.asarray(map_points_xy, dtype=np.float32)
        if world_pts.shape != map_pts.shape or world_pts.ndim != 2 or world_pts.shape[1] != 2:
            raise ValueError("world_points_xy and map_points_xy must both be shaped as [N, 2]")
        design = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1), dtype=np.float32)], axis=1)
        coeff, _, _, _ = np.linalg.lstsq(design, map_pts, rcond=None)
        mat = coeff[:2, :].T.astype(np.float32)
        bias = coeff[2, :].astype(np.float32)
        if not allow_reflection and np.linalg.det(mat) < 0.0:
            raise ValueError("estimated transform includes reflection but allow_reflection is False")
        return Transform2D(mat=mat, bias=bias)

    def save(self, transform: Transform2D, path: str) -> None:
        payload = {
            "mat": np.asarray(transform.mat, dtype=np.float32).tolist(),
            "bias": np.asarray(transform.bias, dtype=np.float32).tolist(),
        }
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

    def load(self, path: str) -> Transform2D:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        return Transform2D(
            mat=np.asarray(payload["mat"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )

