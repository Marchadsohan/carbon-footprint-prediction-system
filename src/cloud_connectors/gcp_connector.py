from typing import Dict, Any, List
from .base_connector import BaseCloudConnector

class GCPConnector(BaseCloudConnector):

    # Carbon intensity by GCP region (kg CO2/kWh)
    REGION_CARBON_INTENSITY = {
        'us-central1':      0.000479,
        'us-east1':         0.000500,
        'us-east4':         0.000364,
        'us-west1':         0.000117,  # Cleanest - hydro/wind
        'us-west2':         0.000248,
        'europe-west1':     0.000149,  # Belgium - very clean
        'europe-west2':     0.000231,
        'europe-west3':     0.000319,
        'europe-west4':     0.000390,
        'asia-east1':       0.000542,
        'asia-southeast1':  0.000493,
        'asia-northeast1':  0.000506,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.project_id = config.get('project_id')
        self.credentials_path = config.get('credentials_path')

        # Set credentials if path provided
        if self.credentials_path:
            import os
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path

    def list_regions(self) -> List[str]:
        return list(self.REGION_CARBON_INTENSITY.keys())

    def fetch_usage_metrics(self, start_time, end_time, granularity: str = "hour"):
        """Fetch CPU metrics from GCP Cloud Monitoring."""
        try:
            from google.cloud import monitoring_v3
            from google.protobuf.timestamp_pb2 import Timestamp

            client = monitoring_v3.MetricServiceClient()
            project_name = f"projects/{self.project_id}"

            instance_id = self.config.get('instance_id', '')
            region = self.config.get('default_region', 'us-central1')

            # GCP metric filter for CPU
            filter_str = (
                'metric.type="compute.googleapis.com/instance/cpu/utilization" '
                f'AND resource.labels.instance_id="{instance_id}"'
            )

            # Time interval
            interval = monitoring_v3.TimeInterval()
            start_ts = Timestamp()
            start_ts.FromDatetime(start_time)
            end_ts = Timestamp()
            end_ts.FromDatetime(end_time)
            interval.start_time = start_ts
            interval.end_time = end_ts

            results = client.list_time_series(
                request={
                    "name": project_name,
                    "filter": filter_str,
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            usage_records = []
            for series in results:
                for point in series.points:
                    cpu_value = point.value.double_value * 100  # Convert to %
                    usage_records.append({
                        "timestamp": point.interval.end_time.isoformat(),
                        "region": region,
                        "service": "Compute Engine",
                        "instance_type": self.config.get('machine_type', 'e2-micro'),
                        "cpu_utilization": cpu_value,
                        "memory_utilization": 0.0,   # Needs Cloud Monitoring agent
                        "kwh": self._estimate_kwh(cpu_value),
                        "cost": 0.0,
                    })

            return usage_records

        except Exception as e:
            return []

    def _estimate_kwh(self, cpu_percent: float) -> float:
        """Estimate kWh from CPU utilization (GCP e2-micro baseline)."""
        base_watts = 2.0   # e2-micro idle
        max_watts  = 8.0
        actual_watts = base_watts + (max_watts - base_watts) * (cpu_percent / 100.0)
        return (actual_watts / 1000.0) * (5 / 60.0)  # 5 min window

    def fetch_carbon_intensity(self, region: str) -> float:
        return self.REGION_CARBON_INTENSITY.get(region, 0.000400) * 1000  # Convert to gCO2/kWh

    def estimate_emissions(self, usage_record: Dict[str, Any]) -> float:
        kwh = usage_record.get('kwh', 0.0)
        intensity = self.REGION_CARBON_INTENSITY.get(
            usage_record.get('region', 'us-central1'), 0.000400
        )
        return kwh * intensity  # kgCO2
