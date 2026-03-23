import os
from typing import List
from dotenv import load_dotenv
from .base_connector import BaseCloudConnector
from .aws_connector import AWSConnector
from .gcp_connector import GCPConnector
from .azure_connector import AzureConnector

load_dotenv()

def get_all_configured_connectors() -> List[BaseCloudConnector]:
    """
    Auto-detects clouds from .env.
    User only fills .env — everything else is automatic.
    """
    connectors = []

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        connectors.append(AWSConnector({
            "provider":              "aws",
            "aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "default_region":        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "instance_id":           os.getenv("EC2_INSTANCE_ID", ""),
        }))
        print("[Factory] AWS loaded")

    if os.getenv("GCP_PROJECT_ID"):
        connectors.append(GCPConnector({
            "provider":         "gcp",
            "project_id":       os.getenv("GCP_PROJECT_ID"),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "default_region":   os.getenv("GCP_DEFAULT_REGION", "us-central1"),
            "instance_id":      os.getenv("GCE_INSTANCE_ID", ""),
            "machine_type":     os.getenv("GCE_MACHINE_TYPE", "e2-micro"),
        }))
        print("[Factory] GCP loaded")

    if os.getenv("AZURE_CLIENT_ID"):
        connectors.append(AzureConnector({
            "provider":              "azure",
            "azure_subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
            "azure_tenant_id":       os.getenv("AZURE_TENANT_ID"),
            "azure_client_id":       os.getenv("AZURE_CLIENT_ID"),
            "azure_client_secret":   os.getenv("AZURE_CLIENT_SECRET"),
            "azure_resource_group":  os.getenv("AZURE_RESOURCE_GROUP", ""),
            "default_region":        os.getenv("AZURE_DEFAULT_REGION", "eastus"),
            "instance_id":           os.getenv("AZURE_VM_RESOURCE_ID", ""),
            "vm_size":               os.getenv("AZURE_VM_SIZE", "Standard_B1s"),
        }))
        print("[Factory] Azure loaded")

    if not connectors:
        print("[Factory] WARNING: No credentials found in .env")

    return connectors
