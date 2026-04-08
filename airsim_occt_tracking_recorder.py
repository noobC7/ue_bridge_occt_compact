import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {k: _to_builtin(v) for k, v in vars(value).items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def make_output_dir(base_dir=None, prefix="tracking_run", suffix=None):
    base = Path(base_dir) if base_dir else Path.cwd() / "airsim_occt_tracking_outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    if suffix:
        dir_name = f"{dir_name}_{suffix}"
    out_dir = base / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class TrackingLogRecorder:
    def __init__(self, output_dir, metadata):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = _to_builtin(metadata)
        self.steps = []

    def add_step(self, step_idx, info):
        self.steps.append({
            "step": int(step_idx),
            "info": _to_builtin(info),
        })

    def save(self, filename="tracking_log.json"):
        payload = {
            "metadata": self.metadata,
            "steps": self.steps,
        }
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)
        return path
