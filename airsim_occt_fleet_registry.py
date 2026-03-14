from typing import List, Tuple

from airsim_occt_config import VehicleConfig
from airsim_occt_schema import NeighborIndices


class FleetRegistry:
    def __init__(self, vehicle_configs: List[VehicleConfig]) -> None:
        self._vehicle_configs = list(vehicle_configs)
        self._name_to_index = {
            cfg.vehicle_name: index for index, cfg in enumerate(self._vehicle_configs)
        }

    @property
    def n_agents(self) -> int:
        return len(self._vehicle_configs)

    @property
    def vehicle_names(self) -> List[str]:
        return [cfg.vehicle_name for cfg in self._vehicle_configs]

    def ordered_pairs(self) -> List[Tuple[int, str]]:
        return [(index, cfg.vehicle_name) for index, cfg in enumerate(self._vehicle_configs)]

    def index_of(self, vehicle_name: str) -> int:
        return self._name_to_index[vehicle_name]

    def config_of(self, agent_index: int) -> VehicleConfig:
        return self._vehicle_configs[agent_index]

    def neighbors_of(self, agent_index: int) -> NeighborIndices:
        front = agent_index - 1 if agent_index > 0 else None
        rear = agent_index + 1 if agent_index < self.n_agents - 1 else None
        return NeighborIndices(front=front, rear=rear)

