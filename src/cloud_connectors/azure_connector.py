from typing import Dict, Any, List
from .base_connector import BaseCloudConnector

class AzureConnector(BaseCloudConnector):

    REGION_CARBON_INTENSITY = {
        'eastus':           0.000386,
        'eastus2':          0.000386,
        'westus':           0.000247,
        'westus2':          0.000136,  # Cleanest - hydro/wind
        'centralus':        0.000479,
        'northeurope':      0.000316,  # Ireland
        'westeurope':       0.000390,  # Netherlands
        'uksouth':          0.000231,
        'ukwest':           0.000231,
        'southeastasia':    0.000493,  # Singapore
        'eastasia':         0.000542,  # Hong Kong
        'centralindia':     0.000708,  # India
        'southindia':       0.000708,
    }

    VM_POWER = {
        'Standard_B1s':   5,
        'Standard_B1ms':  10,
        'Standard_B2s':   20,
        'Standard_B2ms':  40,
        'Standard_D2s_v3': 85,
        'Standard_D4s_v3': 170,
        'Standard_F2s_v2': 70,
        'Standard_F4s_v2': 140,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.subscription_id = config.get('azure_subscription_id')
        self.tenant_id       = config.get('azure_tenant_id')
        self.client_id       = config.get('azure_client_id')
        self.client_secret   = config.get('azure_client_secret')
        self.resource_group  = config.get('azure_resource_group', '')
        self._credential     = None

    def _get_credential(self):
        """Lazy load Azure credential."""
        if not self._credential:
            from azure.identity import ClientSecretCredential
            self._credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        return self._credential

    def list_regions(self) -> List[str]:
        return list(self.REGION_CARBON_INTENSITY.keys())

    def fetch_usage_metrics(self, start_time, end_time, granularity: str = "hour"):
        """Fetch CPU metrics from Azure Monitor."""
        try:
            from azure.monitor.query import MetricsQueryClient
            import datetime

            credential   = self._get_credential()
            client       = MetricsQueryClient(credential)
            resource_id  = self.config.get('instance_id', '')
            region       = self.config.get('default_region', 'eastus')

            if not resource_id:
                return []

            response = client.query_resource(
                resource_uri=resource_id,
                metric_names=["Percentage CPU"],
                timespan=(start_time, end_time),
                granularity=datetime.timedelta(hours=1),
                aggregations=["Average"]
            )

            usage_records = []
            for metric in response.metrics:
                for ts in metric.timeseries:
                    for point in ts.data:
                        if point.average is not None:
                            cpu = point.average
                            usage_records.append({
                                "timestamp": point.timestamp.isoformat(),
                                "region": region,
                                "service": "Azure VM",
                                "instance_type": self.config.get('vm_size', 'Standard_B1s'),
                                "cpu_utilization": cpu,
                                "memory_utilization": 0.0,
                                "kwh": self._estimate_kwh(cpu, self.config.get('vm_size', 'Standard_B1s')),
                                "cost": 0.0,
                            })

            return usage_records

        except Exception as e:
            return []

    def _estimate_kwh(self, cpu_percent: float, vm_size: str) -> float:
        base_watts   = self.VM_POWER.get(vm_size, 20)
        actual_watts = base_watts * (0.5 + 0.5 * (cpu_percent / 100.0))
        return (actual_watts / 1000.0) * (5 / 60.0)

    def fetch_carbon_intensity(self, region: str) -> float:
        return self.REGION_CARBON_INTENSITY.get(region, 0.000400) * 1000

    def estimate_emissions(self, usage_record: Dict[str, Any]) -> float:
        kwh       = usage_record.get('kwh', 0.0)
        intensity = self.REGION_CARBON_INTENSITY.get(
            usage_record.get('region', 'eastus'), 0.000400
        )
        return kwh * intensity
