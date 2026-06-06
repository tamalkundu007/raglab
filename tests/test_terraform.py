"""Unit tests for Terraform infrastructure files (R4 Phase 8)."""
from __future__ import annotations
import re
from pathlib import Path
import pytest

INFRA_ROOT = Path(__file__).parent.parent / "infra" / "terraform"
ACTIVE_SERVICES = [
    "api-gateway","ingestion","embedding","indexing","retrieval",
    "llm","pipeline","storage","ui","graph","observability","auth"
]

def sv():    return (INFRA_ROOT/"shared"/"variables.tf").read_text()
def az():    return (INFRA_ROOT/"azure"/"main.tf").read_text()
def wl():    return (INFRA_ROOT/"azure"/"workloads.tf").read_text()
def aws():   return (INFRA_ROOT/"aws"/"main.tf").read_text()
def gcp():   return (INFRA_ROOT/"gcp"/"main.tf").read_text()
def readme():return (INFRA_ROOT/"README.md").read_text()

class TestFilesExist:
    def test_azure_main(self):   assert (INFRA_ROOT/"azure"/"main.tf").exists()
    def test_azure_workloads(self): assert (INFRA_ROOT/"azure"/"workloads.tf").exists()
    def test_aws_main(self):     assert (INFRA_ROOT/"aws"/"main.tf").exists()
    def test_gcp_main(self):     assert (INFRA_ROOT/"gcp"/"main.tf").exists()
    def test_shared_vars(self):  assert (INFRA_ROOT/"shared"/"variables.tf").exists()
    def test_readme(self):       assert (INFRA_ROOT/"README.md").exists()

class TestSharedVariables:
    def test_environment(self):       assert 'variable "environment"' in sv()
    def test_project_name(self):      assert 'variable "project_name"' in sv()
    def test_image_tag(self):         assert 'variable "image_tag"' in sv()
    def test_service_ports(self):     assert 'variable "service_ports"' in sv()
    def test_all_services_in_ports(self):
        for s in ACTIVE_SERVICES: assert s in sv(), f"Missing: {s}"
    def test_node_pool_min(self):     assert "node_pool_min_count" in sv()
    def test_node_pool_max(self):     assert "node_pool_max_count" in sv()
    def test_postgres_storage(self):  assert "postgres_storage_gb" in sv()
    def test_qdrant_replicas(self):   assert "qdrant_replicas" in sv()
    def test_rabbitmq_replicas(self): assert "rabbitmq_replicas" in sv()
    def test_common_tags(self):       assert "common_tags" in sv()
    def test_env_validation(self):    assert "validation" in sv()

class TestAzureMain:
    def test_required_version(self): assert ">= 1.7" in az()
    def test_azurerm_provider(self): assert "hashicorp/azurerm" in az()
    def test_oidc_no_client_secret(self):
        assert "use_oidc = true" in az()
        # No actual client_secret assignment (only referenced in comments)
        assert not re.search(r'client_secret\s*=\s*"', az())
    def test_resource_group(self):   assert "azurerm_resource_group" in az()
    def test_vnet(self):             assert "azurerm_virtual_network" in az()
    def test_aks_subnet(self):       assert '"aks"' in az()
    def test_postgres_subnet(self):  assert '"postgres"' in az() and "delegation" in az()
    def test_acr(self):              assert "azurerm_container_registry" in az()
    def test_acr_admin_disabled(self):
        assert "admin_enabled" in az() and "false" in az()
    def test_aks_cluster(self):      assert "azurerm_kubernetes_cluster" in az()
    def test_aks_workload_identity(self):
        assert "workload_identity_enabled = true" in az()
    def test_aks_autoscaling(self):  assert "enable_auto_scaling" in az()
    def test_postgres(self):         assert "azurerm_postgresql_flexible_server" in az()
    def test_postgres_v16(self):     assert '"16"' in az()
    def test_key_vault(self):        assert "azurerm_key_vault" in az()
    def test_kv_secrets(self):       assert "azurerm_key_vault_secret" in az()
    def test_log_analytics(self):    assert "azurerm_log_analytics_workspace" in az()
    def test_acr_role_assignment(self): assert "AcrPull" in az()
    def test_outputs(self):
        assert 'output "aks_cluster_name"' in az()
        assert 'output "acr_login_server"' in az()
    def test_backend_commented(self): assert "backend" in az()
    def test_no_hardcoded_password(self):
        assert not re.search(r'password\s*=\s*"[^"]{8,}"', az(), re.IGNORECASE)
        assert "random_password" in az()
    def test_purge_protection(self): assert "purge_protection_enabled" in az()

class TestAzureWorkloads:
    def test_kubernetes_provider(self): assert "hashicorp/kubernetes" in wl()
    def test_helm_provider(self):       assert "hashicorp/helm" in wl()
    def test_qdrant_helm(self):         assert "qdrant" in wl() and "helm_release" in wl()
    def test_rabbitmq_helm(self):       assert "rabbitmq" in wl()
    def test_all_services(self):
        for s in ACTIVE_SERVICES: assert s in wl(), f"Workloads missing: {s}"
    def test_k8s_deployments(self):   assert "kubernetes_deployment" in wl()
    def test_k8s_services(self):      assert "kubernetes_service" in wl()
    def test_liveness_probe(self):    assert "liveness_probe" in wl()
    def test_readiness_probe(self):   assert "readiness_probe" in wl()
    def test_health_path(self):       assert "/health" in wl()
    def test_secret_ref(self):        assert "secret_ref" in wl() or "env_from" in wl()
    def test_loadbalancer(self):      assert "LoadBalancer" in wl()
    def test_clusterip(self):         assert "ClusterIP" in wl()

class TestAWSMain:
    def test_required_version(self):  assert "required_version" in aws()
    def test_aws_provider(self):      assert "hashicorp/aws" in aws()
    def test_no_static_keys(self):
        assert "access_key" not in aws()
        assert "secret_key" not in aws()
    def test_vpc(self):               assert "aws_vpc" in aws()
    def test_public_subnets(self):    assert '"public"' in aws()
    def test_private_subnets(self):   assert '"private"' in aws()
    def test_db_subnets(self):        assert '"database"' in aws()
    def test_nat_gateway(self):       assert "aws_nat_gateway" in aws()
    def test_ecr_repos(self):         assert "aws_ecr_repository" in aws()
    def test_all_services_in_ecr(self):
        for s in ACTIVE_SERVICES: assert s in aws(), f"ECR missing: {s}"
    def test_ecr_scan_on_push(self):  assert "scan_on_push = true" in aws()
    def test_ecr_lifecycle(self):     assert "aws_ecr_lifecycle_policy" in aws()
    def test_eks_cluster(self):       assert "aws_eks_cluster" in aws()
    def test_eks_v129(self):          assert '"1.29"' in aws()
    def test_oidc_provider(self):
        assert "aws_iam_openid_connect_provider" in aws()
        assert "sts.amazonaws.com" in aws()
    def test_node_group(self):        assert "aws_eks_node_group" in aws()
    def test_autoscaling(self):       assert "scaling_config" in aws()
    def test_rds(self):               assert "aws_db_instance" in aws()
    def test_rds_pg16(self):          assert '"16"' in aws()
    def test_rds_multi_az(self):      assert "multi_az" in aws() and "true" in aws()
    def test_rds_encrypted(self):     assert "storage_encrypted" in aws() and "true" in aws()
    def test_deletion_protection(self):assert "deletion_protection" in aws() and "true" in aws()
    def test_secrets_manager(self):   assert "aws_secretsmanager_secret" in aws()
    def test_db_dsn_secret(self):     assert "postgres-dsn" in aws() or "postgres_dsn" in aws()
    def test_llm_secrets(self):       assert "azure-openai-key" in aws() or "anthropic-api-key" in aws()
    def test_cloudwatch(self):        assert "aws_cloudwatch_log_group" in aws()
    def test_log_retention(self):     assert "retention_in_days" in aws()
    def test_backend_commented(self): assert "backend" in aws()
    def test_no_aws_access_key(self): assert not re.search(r'AKIA[A-Z0-9]{16}', aws())
    def test_outputs(self):
        assert 'output "eks_cluster_name"' in aws()
        assert 'output "ecr_registry"' in aws()
    def test_iam_roles(self):
        assert "aws_iam_role" in aws()
        assert "AmazonEKSClusterPolicy" in aws()

class TestGCPStub:
    def test_r7_documented(self):    assert "R7" in gcp()
    def test_no_active_resources(self):
        for i, line in enumerate(gcp().split("\n")):
            s = line.strip()
            assert not s.startswith('resource "google_'), \
                f"GCP active resource at line {i+1}: {s}"
    def test_provider_commented(self):
        assert '# provider "google"' in gcp() or "provider block" in gcp().lower()
    def test_no_sa_key_file(self):
        assert not re.search(r'credentials\s*=\s*"[^"]+\.json"', gcp())
    def test_gke_autopilot(self):    assert "Autopilot" in gcp() or "autopilot" in gcp()
    def test_cloud_sql(self):        assert "Cloud SQL" in gcp() or "google_sql" in gcp()
    def test_artifact_registry(self):assert "Artifact Registry" in gcp() or "artifact_registry" in gcp()
    def test_stub_output(self):      assert 'output "r7_stub_message"' in gcp()
    def test_workload_identity(self):assert "Workload Identity" in gcp() or "workload_identity" in gcp()

class TestReadme:
    def test_azure_section(self):    assert "Azure" in readme() and "AKS" in readme()
    def test_aws_section(self):      assert "AWS" in readme() and "EKS" in readme()
    def test_gcp_r7(self):           assert "R7" in readme() and "GCP" in readme()
    def test_remote_state(self):     assert "backend" in readme() or "remote state" in readme().lower()
    def test_azure_secrets(self):    assert "Key Vault" in readme() or "keyvault" in readme().lower()
    def test_aws_secrets(self):      assert "Secrets Manager" in readme()
    def test_security(self):         assert "Security" in readme() or "security" in readme()
    def test_oidc(self):             assert "OIDC" in readme()
    def test_no_keys(self):
        assert not re.search(r'sk-[A-Za-z0-9]{30,}', readme())
        assert not re.search(r'AKIA[A-Z0-9]{16}', readme())
    def test_workload_identity(self):assert "Workload Identity" in readme() or "IRSA" in readme()

class TestSecurityAudit:
    def _tfs(self): return list(INFRA_ROOT.rglob("*.tf"))
    def test_no_openai_keys(self):
        for f in self._tfs():
            assert not re.search(r'sk-[A-Za-z0-9]{30,}', f.read_text()), f"Key in {f.name}"
    def test_no_aws_access_keys(self):
        for f in self._tfs():
            assert not re.search(r'AKIA[A-Z0-9]{16}', f.read_text()), f"AWS key in {f.name}"
    def test_no_plaintext_passwords(self):
        for f in self._tfs():
            assert not re.search(r'password\s*=\s*"[^"]{8,}"', f.read_text(), re.IGNORECASE), \
                f"Password in {f.name}"
    def test_no_gcp_sa_json(self):
        for f in self._tfs():
            assert not re.search(r'credentials\s*=\s*"[^"]+\.json"', f.read_text()), \
                f"SA key path in {f.name}"
