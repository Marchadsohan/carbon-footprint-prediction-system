# src/cloud_connectors/base_connector.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseCloudConnector(ABC):
    """Common interface for all cloud providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def list_regions(self) -> List[str]:
        pass

    @abstractmethod
    def fetch_usage_metrics(
        self,
        start_time,
        end_time,
        granularity: str = "hour"
    ) -> Any:
        """Return CPU, memory, energy, cost, etc., in a common schema."""
        pass

    @abstractmethod
    def fetch_carbon_intensity(self, region: str) -> float:
        """Return gCO2/kWh or similar for a region."""
        pass

    @abstractmethod
    def estimate_emissions(self, usage_record: Dict[str, Any]) -> float:
        """Compute kgCO2 for a single usage record."""
        pass
