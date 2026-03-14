from typing import Dict

import numpy as np


class OracleParityTester:
    def compare_single_frame(self, train_snapshot: Dict, deploy_snapshot: Dict, atol: float = 1e-5) -> Dict:
        train_obs = np.asarray(train_snapshot["obs"], dtype=np.float32)
        deploy_obs = np.asarray(deploy_snapshot["obs"], dtype=np.float32)
        if train_obs.shape != deploy_obs.shape:
            return {
                "pass": False,
                "shape_match": False,
                "train_shape": tuple(train_obs.shape),
                "deploy_shape": tuple(deploy_obs.shape),
            }
        diff = np.abs(train_obs - deploy_obs)
        return {
            "pass": bool(np.allclose(train_obs, deploy_obs, atol=atol)),
            "shape_match": True,
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        }

