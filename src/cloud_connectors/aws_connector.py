"""
AWS Cloud Connector
Handles: connection, metric collection, carbon calculation, data saving
"""

import os
import boto3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from .base_connector import BaseCloudConnector

logger = logging.getLogger(__name__)


class AWSConnector(BaseCloudConnector):

    REGION_CARBON_INTENSITY = {
        'us-east-1':      0.000379,
        'us-east-2':      0.000744,
        'us-west-1':      0.000251,
        'us-west-2':      0.000114,
        'eu-west-1':      0.000316,
        'eu-central-1':   0.000338,
        'ap-southeast-1': 0.000493,
        'ap-northeast-1': 0.000506,
        'ca-central-1':   0.000130,
        'ap-south-1':     0.000708,
    }

    INSTANCE_POWER = {
        't2.micro': 5,   't2.small': 10,  't2.medium': 20,
        't2.large': 40,  't2.xlarge': 80,
        't3.micro': 5,   't3.small': 10,  't3.medium': 20,
        't3.large': 40,  't3.xlarge': 80,
        'm5.large': 50,  'm5.xlarge': 100, 'm5.2xlarge': 200,
        'c5.large': 60,  'c5.xlarge': 120, 'c5.2xlarge': 240,
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = boto3.Session(
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            region_name=config.get("default_region", "us-east-1"),
        )
        self.ec2 = self.session.client("ec2")
        self.cloudwatch = self.session.client("cloudwatch")
        logger.info(f"AWS connector ready | region: {config.get('default_region')}")

    # ─── BaseConnector interface ──────────────────────────────────────────

    def list_regions(self) -> List[str]:
        resp = self.ec2.describe_regions(AllRegions=False)
        return [r["RegionName"] for r in resp["Regions"]]

    def fetch_usage_metrics(self, start_time, end_time, granularity: str = "hour") -> List[Dict]:
        """Fetch real CPU metrics from CloudWatch."""
        instance_id = self.config.get("instance_id", "")
        if not instance_id:
            return []

        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/EC2",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=["Average", "Maximum"],
            )

            records = []
            region = self.config.get("default_region", "us-east-1")
            for dp in sorted(response.get("Datapoints", []), key=lambda x: x["Timestamp"]):
                cpu = dp["Average"]
                records.append({
                    "timestamp":            dp["Timestamp"].isoformat(),
                    "region":               region,
                    "service":              "EC2",
                    "instance_id":          instance_id,
                    "instance_type":        self.config.get("instance_type", "t3.micro"),
                    "cpu_utilization":      cpu,
                    "cpu_utilization_max":  dp.get("Maximum", cpu),
                    "memory_utilization":   0.0,
                    "kwh":                  self._estimate_kwh(cpu, self.config.get("instance_type", "t3.micro")),
                    "cost":                 0.0,
                    "cloud_provider":       "AWS",
                })
            return records

        except Exception as e:
            logger.error(f"CloudWatch error: {e}")
            return []

    def fetch_carbon_intensity(self, region: str) -> float:
        return self.REGION_CARBON_INTENSITY.get(region, 0.000400) * 1000  # gCO2/kWh

    def estimate_emissions(self, usage_record: Dict[str, Any]) -> float:
        kwh = usage_record.get("kwh", 0.0)
        intensity = self.REGION_CARBON_INTENSITY.get(
            usage_record.get("region", "us-east-1"), 0.000400
        )
        return kwh * intensity  # kgCO2

    # ─── Collection helpers ───────────────────────────────────────────────

    def get_running_instances(self) -> List[Dict]:
        """Get all running EC2 instances."""
        try:
            response = self.ec2.describe_instances(
                Filters=[{"Name": "instance-state-name", "Values": ["running"]}]
            )
            instances = []
            for r in response["Reservations"]:
                for i in r["Instances"]:
                    name = next(
                        (t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"),
                        "unnamed"
                    )
                    instances.append({
                        "instance_id":        i["InstanceId"],
                        "instance_type":      i["InstanceType"],
                        "instance_name":      name,
                        "region":             self.config.get("default_region"),
                        "availability_zone":  i["Placement"]["AvailabilityZone"],
                        "state":              i["State"]["Name"],
                        "cloud_provider":     "AWS",
                    })
            logger.info(f"Found {len(instances)} running EC2 instances")
            return instances
        except Exception as e:
            logger.error(f"EC2 list error: {e}")
            return []

    def collect_and_calculate(self) -> pd.DataFrame:
        """
        Main method: collect metrics + calculate carbon for all running instances.
        Returns a DataFrame ready for XGBoost / dashboard.
        """
        instances = self.get_running_instances()

        # Fallback if no running instance
        if not instances:
            logger.warning("No running instances — using fallback data")
            instances = [{
                "instance_id":       self.config.get("instance_id", "i-unknown"),
                "instance_type":     "t3.micro",
                "instance_name":     "fallback",
                "region":            self.config.get("default_region", "us-east-1"),
                "availability_zone": "us-east-1a",
                "state":             "stopped",
                "cloud_provider":    "AWS",
            }]

        rows = []
        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        for inst in instances:
            self.config["instance_id"]   = inst["instance_id"]
            self.config["instance_type"] = inst["instance_type"]

            metrics = self.fetch_usage_metrics(start_time, end_time)

            if not metrics:
                # Use default CPU when instance stopped/no data
                cpu = 6.8
                metrics = [{
                    "timestamp":           datetime.utcnow().isoformat(),
                    "region":              inst["region"],
                    "service":             "EC2",
                    "instance_id":         inst["instance_id"],
                    "instance_type":       inst["instance_type"],
                    "cpu_utilization":     cpu,
                    "cpu_utilization_max": cpu,
                    "memory_utilization":  0.0,
                    "kwh":                 self._estimate_kwh(cpu, inst["instance_type"]),
                    "cost":                0.0,
                    "cloud_provider":      "AWS",
                }]

            for record in metrics:
                record.update({
                    "instance_name":     inst["instance_name"],
                    "availability_zone": inst["availability_zone"],
                    "carbon_kg":         self.estimate_emissions(record),
                    "carbon_kg_per_day": self.estimate_emissions(record) * 288,
                })
            rows.extend(metrics)

        df = pd.DataFrame(rows)
        logger.info(f"AWS: collected {len(df)} records")
        return df

    # ─── Private helpers ──────────────────────────────────────────────────

    def _estimate_kwh(self, cpu_percent: float, instance_type: str) -> float:
        base_watts   = self.INSTANCE_POWER.get(instance_type, 20)
        actual_watts = base_watts * (0.5 + 0.5 * (cpu_percent / 100.0))
        return (actual_watts / 1000.0) * (5 / 60.0)  # 5 min window

    def save_data(self, df: pd.DataFrame, output_dir: str = "data/realtime") -> str:
        """Save collected data to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{output_dir}/aws_carbon_{ts}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved: {filepath}")
        return filepath
