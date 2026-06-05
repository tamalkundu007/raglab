"""
Unit tests for CI/CD pipeline configuration files.

Tests validate:
- YAML syntax and structure of all workflow files
- Required jobs present with correct names
- Security: OIDC auth used, no hardcoded secrets
- Service matrix completeness (all 9 active services covered)
- GCP workflow is correctly disabled (no on: triggers)
- Branch targeting correct
- Concurrency controls present
- Bicep template structure
- Terraform structure
- Setup guide completeness
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
DEPLOY_DIR = REPO_ROOT / "deploy"
DOCS_DIR = REPO_ROOT / "docs"

ACTIVE_SERVICES = [
    "api-gateway", "ingestion", "embedding", "indexing",
    "retrieval", "llm", "pipeline", "storage", "ui",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def read_file(path: Path) -> str:
    return path.read_text()


def ci():      return load_yaml(WORKFLOWS_DIR / "ci.yml")
def cd_azure():return load_yaml(WORKFLOWS_DIR / "cd-azure.yml")
def cd_aws():  return load_yaml(WORKFLOWS_DIR / "cd-aws.yml")
def cd_gcp():  return load_yaml(WORKFLOWS_DIR / "cd-gcp.yml")
def bicep():   return read_file(DEPLOY_DIR / "azure" / "main.bicep")
def tf():      return read_file(DEPLOY_DIR / "aws" / "main.tf")
def guide():   return read_file(DOCS_DIR / "CI_CD_SETUP.md")


# ═══════════════════════════════════════════════════════════════════════════════
# File existence
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilesExist:
    def test_ci_yml_exists(self):
        assert (WORKFLOWS_DIR / "ci.yml").exists()

    def test_cd_azure_yml_exists(self):
        assert (WORKFLOWS_DIR / "cd-azure.yml").exists()

    def test_cd_aws_yml_exists(self):
        assert (WORKFLOWS_DIR / "cd-aws.yml").exists()

    def test_cd_gcp_yml_exists(self):
        assert (WORKFLOWS_DIR / "cd-gcp.yml").exists()

    def test_azure_deploy_dir_exists(self):
        assert (DEPLOY_DIR / "azure").is_dir()

    def test_aws_deploy_dir_exists(self):
        assert (DEPLOY_DIR / "aws").is_dir()

    def test_gcp_deploy_dir_exists(self):
        assert (DEPLOY_DIR / "gcp").is_dir()

    def test_azure_bicep_exists(self):
        assert (DEPLOY_DIR / "azure" / "main.bicep").exists()

    def test_aws_terraform_exists(self):
        assert (DEPLOY_DIR / "aws" / "main.tf").exists()

    def test_gcp_readme_exists(self):
        assert (DEPLOY_DIR / "gcp" / "README.md").exists()

    def test_cicd_setup_guide_exists(self):
        assert (DOCS_DIR / "CI_CD_SETUP.md").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# CI workflow
# ═══════════════════════════════════════════════════════════════════════════════

class TestCIWorkflow:
    def test_name_is_ci(self):
        assert ci().get("name") == "CI"

    def test_triggers_push_main(self):
        branches = ci()["on"]["push"]["branches"]
        assert "main" in branches

    def test_triggers_push_develop(self):
        branches = ci()["on"]["push"]["branches"]
        assert "develop" in branches

    def test_triggers_pull_request(self):
        assert "pull_request" in ci()["on"]

    def test_concurrency_cancel_in_progress(self):
        assert ci()["concurrency"]["cancel-in-progress"] is True

    def test_lint_job_present(self):
        assert "lint" in ci()["jobs"]

    def test_test_job_present(self):
        assert "test" in ci()["jobs"]

    def test_build_check_job_present(self):
        assert "build-check" in ci()["jobs"]

    def test_test_needs_lint(self):
        needs = ci()["jobs"]["test"].get("needs", [])
        assert "lint" in (needs if isinstance(needs, list) else [needs])

    def test_build_check_needs_test(self):
        needs = ci()["jobs"]["build-check"].get("needs", [])
        assert "test" in (needs if isinstance(needs, list) else [needs])

    def test_ruff_in_lint_steps(self):
        steps = ci()["jobs"]["lint"]["steps"]
        combined = " ".join(str(s.get("run", "") + s.get("name", "")) for s in steps)
        assert "ruff" in combined.lower()

    def test_pytest_in_test_steps(self):
        steps = ci()["jobs"]["test"]["steps"]
        runs = " ".join(str(s.get("run", "")) for s in steps)
        assert "pytest" in runs

    def test_junit_xml_configured(self):
        steps = ci()["jobs"]["test"]["steps"]
        runs = " ".join(str(s.get("run", "")) for s in steps)
        assert "junitxml" in runs or "junit" in runs.lower()

    def test_coverage_configured(self):
        steps = ci()["jobs"]["test"]["steps"]
        runs = " ".join(str(s.get("run", "")) for s in steps)
        assert "cov" in runs

    def test_build_matrix_has_all_services(self):
        matrix = ci()["jobs"]["build-check"]["strategy"]["matrix"]["service"]
        for svc in ACTIVE_SERVICES:
            assert svc in matrix, f"CI build matrix missing: {svc!r}"

    def test_python_312_in_matrix(self):
        matrix = ci()["jobs"]["test"]["strategy"]["matrix"]
        assert "3.12" in matrix.get("python-version", [])

    def test_no_hardcoded_secrets(self):
        content = read_file(WORKFLOWS_DIR / "ci.yml")
        assert not re.search(r'sk-[A-Za-z0-9]{20,}', content)
        assert not re.search(r'ghp_[A-Za-z0-9]{20,}', content)


# ═══════════════════════════════════════════════════════════════════════════════
# Azure CD workflow
# ═══════════════════════════════════════════════════════════════════════════════

class TestAzureCDWorkflow:
    def test_name_contains_azure(self):
        assert "Azure" in cd_azure().get("name", "")

    def test_triggers_push_main_only(self):
        branches = cd_azure()["on"]["push"]["branches"]
        assert "main" in branches
        assert "develop" not in branches

    def test_workflow_dispatch_present(self):
        assert "workflow_dispatch" in cd_azure()["on"]

    def test_oidc_id_token_write(self):
        assert cd_azure()["permissions"]["id-token"] == "write"

    def test_contents_read_permission(self):
        assert cd_azure()["permissions"]["contents"] == "read"

    def test_no_cancel_in_progress(self):
        assert cd_azure()["concurrency"]["cancel-in-progress"] is False

    def test_detect_changes_job(self):
        assert "detect-changes" in cd_azure()["jobs"]

    def test_build_push_job(self):
        assert "build-push" in cd_azure()["jobs"]

    def test_deploy_job(self):
        assert "deploy-azure" in cd_azure()["jobs"]

    def test_azure_login_oidc_in_build(self):
        steps = cd_azure()["jobs"]["build-push"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("azure/login" in u for u in uses)

    def test_azure_login_oidc_in_deploy(self):
        steps = cd_azure()["jobs"]["deploy-azure"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("azure/login" in u for u in uses)

    def test_secrets_referenced(self):
        content = read_file(WORKFLOWS_DIR / "cd-azure.yml")
        assert "secrets.ACR_LOGIN_SERVER" in content
        assert "secrets.AZURE_CLIENT_ID" in content
        assert "secrets.AZURE_TENANT_ID" in content

    def test_build_matrix_all_services(self):
        matrix = cd_azure()["jobs"]["build-push"]["strategy"]["matrix"]["service"]
        for svc in ACTIVE_SERVICES:
            assert svc in matrix, f"Azure CD matrix missing: {svc!r}"

    def test_docker_build_push_action_used(self):
        steps = cd_azure()["jobs"]["build-push"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("docker/build-push-action" in u for u in uses)

    def test_github_sha_as_image_tag(self):
        assert "github.sha" in read_file(WORKFLOWS_DIR / "cd-azure.yml")

    def test_bicep_in_deploy(self):
        steps = cd_azure()["jobs"]["deploy-azure"]["steps"]
        runs = " ".join(str(s.get("run", "")) for s in steps)
        assert "bicep" in runs.lower() or "deployment group" in runs.lower()

    def test_health_check_in_deploy(self):
        steps = cd_azure()["jobs"]["deploy-azure"]["steps"]
        names = [s.get("name", "").lower() for s in steps]
        assert any("health" in n for n in names)

    def test_production_environment_set(self):
        env = str(cd_azure()["jobs"]["deploy-azure"].get("environment", ""))
        assert "production" in env.lower() or "azure" in env.lower()

    def test_no_hardcoded_api_keys(self):
        content = read_file(WORKFLOWS_DIR / "cd-azure.yml")
        assert not re.search(r'sk-[A-Za-z0-9]{20,}', content)


# ═══════════════════════════════════════════════════════════════════════════════
# AWS CD workflow
# ═══════════════════════════════════════════════════════════════════════════════

class TestAWSCDWorkflow:
    def test_name_contains_aws(self):
        assert "AWS" in cd_aws().get("name", "")

    def test_triggers_push_main_only(self):
        branches = cd_aws()["on"]["push"]["branches"]
        assert "main" in branches
        assert "develop" not in branches

    def test_oidc_id_token_write(self):
        assert cd_aws()["permissions"]["id-token"] == "write"

    def test_no_cancel_in_progress(self):
        assert cd_aws()["concurrency"]["cancel-in-progress"] is False

    def test_build_push_job(self):
        assert "build-push" in cd_aws()["jobs"]

    def test_deploy_job(self):
        assert "deploy-aws" in cd_aws()["jobs"]

    def test_oidc_configure_aws_credentials(self):
        steps = cd_aws()["jobs"]["build-push"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("configure-aws-credentials" in u for u in uses)

    def test_ecr_login_present(self):
        steps = cd_aws()["jobs"]["build-push"]["steps"]
        uses = [s.get("uses", "") for s in steps]
        assert any("ecr-login" in u or "amazon-ecr" in u for u in uses)

    def test_no_static_aws_keys(self):
        content = read_file(WORKFLOWS_DIR / "cd-aws.yml")
        assert "AWS_ACCESS_KEY_ID:" not in content
        assert "AWS_SECRET_ACCESS_KEY:" not in content

    def test_role_arn_via_secrets(self):
        assert "secrets.AWS_ROLE_ARN" in read_file(WORKFLOWS_DIR / "cd-aws.yml")

    def test_build_matrix_all_services(self):
        matrix = cd_aws()["jobs"]["build-push"]["strategy"]["matrix"]["service"]
        for svc in ACTIVE_SERVICES:
            assert svc in matrix, f"AWS CD matrix missing: {svc!r}"

    def test_github_sha_as_image_tag(self):
        assert "github.sha" in read_file(WORKFLOWS_DIR / "cd-aws.yml")

    def test_ecs_update_in_deploy(self):
        steps = cd_aws()["jobs"]["deploy-aws"]["steps"]
        runs = " ".join(str(s.get("run", "")) for s in steps)
        assert "ecs" in runs.lower()

    def test_wait_stabilisation_in_deploy(self):
        steps = cd_aws()["jobs"]["deploy-aws"]["steps"]
        names = [s.get("name", "").lower() for s in steps]
        assert any("wait" in n or "stabil" in n for n in names)

    def test_health_check_in_deploy(self):
        steps = cd_aws()["jobs"]["deploy-aws"]["steps"]
        names = [s.get("name", "").lower() for s in steps]
        assert any("health" in n for n in names)

    def test_secrets_manager_referenced(self):
        content = read_file(WORKFLOWS_DIR / "cd-aws.yml")
        assert "secretsmanager" in content.lower() or "Secrets Manager" in content


# ═══════════════════════════════════════════════════════════════════════════════
# GCP stub
# ═══════════════════════════════════════════════════════════════════════════════

class TestGCPStubWorkflow:
    def test_no_active_triggers(self):
        """GCP workflow must NOT have on: triggers until R7."""
        wf = cd_gcp()
        on = wf.get("on")
        assert on is None or on == {} or on is False, \
            f"GCP has active triggers: {on!r} — must be disabled until R7"

    def test_documents_r7(self):
        assert "R7" in read_file(WORKFLOWS_DIR / "cd-gcp.yml")

    def test_stub_job_with_if_false(self):
        jobs = cd_gcp().get("jobs", {})
        for job_name, job in jobs.items():
            if "stub" in job_name.lower():
                assert job.get("if") == False or str(job.get("if")).lower() == "false"

    def test_documents_cloud_run(self):
        assert "Cloud Run" in read_file(WORKFLOWS_DIR / "cd-gcp.yml")

    def test_uses_workload_identity_not_static_key(self):
        content = read_file(WORKFLOWS_DIR / "cd-gcp.yml")
        assert "workload_identity_provider" in content
        assert "GCP_SERVICE_ACCOUNT_KEY" not in content


# ═══════════════════════════════════════════════════════════════════════════════
# Azure Bicep
# ═══════════════════════════════════════════════════════════════════════════════

class TestAzureBicep:
    def test_has_services_definition(self):
        assert "services" in bicep()

    def test_covers_all_active_services(self):
        b = bicep()
        for svc in ACTIVE_SERVICES:
            assert svc in b, f"Bicep missing: {svc!r}"

    def test_uses_key_vault_for_secrets(self):
        assert "keyVaultUrl" in bicep() or "vault.azure.net" in bicep()

    def test_no_hardcoded_api_keys(self):
        assert not re.search(r'sk-[A-Za-z0-9]{20,}', bicep())

    def test_liveness_probe_configured(self):
        assert "Liveness" in bicep()

    def test_readiness_probe_configured(self):
        assert "Readiness" in bicep()

    def test_scaling_rules_defined(self):
        assert "scale" in bicep()

    def test_gateway_output_defined(self):
        assert "output" in bicep() and ("gateway" in bicep().lower() or "FQDN" in bicep())

    def test_container_apps_resource(self):
        assert "containerApps" in bicep() or "Container" in bicep()

    def test_secret_ref_for_credentials(self):
        assert "secretRef" in bicep()


# ═══════════════════════════════════════════════════════════════════════════════
# AWS Terraform
# ═══════════════════════════════════════════════════════════════════════════════

class TestAWSTerraform:
    def test_required_version_present(self):
        assert "required_version" in tf()

    def test_aws_provider_configured(self):
        assert 'hashicorp/aws' in tf()

    def test_ecs_cluster_resource(self):
        assert "aws_ecs_cluster" in tf()

    def test_fargate_configured(self):
        assert "FARGATE" in tf()

    def test_iam_role_for_task_execution(self):
        assert "aws_iam_role" in tf()
        assert "ecs-tasks.amazonaws.com" in tf()

    def test_secrets_manager_for_api_keys(self):
        assert "secretsmanager" in tf().lower()

    def test_no_hardcoded_aws_access_keys(self):
        assert not re.search(r'AKIA[A-Z0-9]{16}', tf())

    def test_all_services_present(self):
        t = tf()
        for svc in ACTIVE_SERVICES:
            assert svc in t, f"Terraform missing: {svc!r}"

    def test_cloudwatch_log_group(self):
        assert "aws_cloudwatch_log_group" in tf()

    def test_ecs_task_definitions(self):
        assert "aws_ecs_task_definition" in tf()

    def test_health_check_configured(self):
        assert "healthCheck" in tf() or "health_check" in tf()

    def test_awsvpc_network_mode(self):
        assert "awsvpc" in tf()

    def test_outputs_defined(self):
        assert "output" in tf()


# ═══════════════════════════════════════════════════════════════════════════════
# Setup guide
# ═══════════════════════════════════════════════════════════════════════════════

class TestCICDSetupGuide:
    def test_azure_secrets_documented(self):
        g = guide()
        assert "AZURE_CLIENT_ID" in g
        assert "ACR_LOGIN_SERVER" in g

    def test_aws_secrets_documented(self):
        g = guide()
        assert "AWS_ROLE_ARN" in g
        assert "ECR_REGISTRY" in g

    def test_oidc_documented_not_static_keys(self):
        g = guide()
        assert "OIDC" in g
        assert "no" in g.lower()

    def test_azure_key_vault_or_secrets_manager(self):
        g = guide()
        assert "Key Vault" in g or "Secrets Manager" in g

    def test_azure_oidc_setup_commands(self):
        g = guide()
        assert "federated-credential" in g

    def test_aws_oidc_setup_commands(self):
        g = guide()
        assert "create-open-id-connect-provider" in g

    def test_gcp_r7_documented(self):
        assert "R7" in guide()

    def test_branch_strategy_documented(self):
        g = guide()
        assert "main" in g and "develop" in g

    def test_monitoring_section_present(self):
        g = guide()
        assert "Monitor" in g or "logs" in g.lower()

    def test_security_section_present(self):
        g = guide()
        assert "Security" in g or "security" in g


# ═══════════════════════════════════════════════════════════════════════════════
# GCP deploy README
# ═══════════════════════════════════════════════════════════════════════════════

class TestGCPDeployReadme:
    @pytest.fixture
    def gcp_readme(self):
        return read_file(DEPLOY_DIR / "gcp" / "README.md")

    def test_documents_r7(self, gcp_readme):
        assert "R7" in gcp_readme

    def test_documents_cloud_run(self, gcp_readme):
        assert "Cloud Run" in gcp_readme

    def test_documents_workload_identity(self, gcp_readme):
        assert "Workload Identity" in gcp_readme

    def test_documents_artifact_registry(self, gcp_readme):
        assert "Artifact Registry" in gcp_readme or "GAR" in gcp_readme

    def test_documents_gcs_storage(self, gcp_readme):
        assert "GCS" in gcp_readme or "Cloud Storage" in gcp_readme

    def test_has_setup_commands(self, gcp_readme):
        assert "gcloud" in gcp_readme


# ═══════════════════════════════════════════════════════════════════════════════
# Security audit — no secrets in any workflow or deploy file
# ═══════════════════════════════════════════════════════════════════════════════

class TestSecurityAudit:
    def test_no_openai_key_in_workflows(self):
        for f in WORKFLOWS_DIR.glob("*.yml"):
            content = f.read_text()
            assert not re.search(r'sk-[A-Za-z0-9]{40,}', content), \
                f"Possible OpenAI key in {f.name}"

    def test_no_aws_access_key_in_workflows(self):
        for f in WORKFLOWS_DIR.glob("*.yml"):
            content = f.read_text()
            assert not re.search(r'AKIA[A-Z0-9]{16}', content), \
                f"Possible AWS access key in {f.name}"

    def test_no_github_pat_in_workflows(self):
        for f in WORKFLOWS_DIR.glob("*.yml"):
            content = f.read_text()
            assert not re.search(r'ghp_[A-Za-z0-9]{36}', content), \
                f"Possible GitHub PAT in {f.name}"

    def test_all_cd_workflows_use_secrets_references(self):
        for fname in ["cd-azure.yml", "cd-aws.yml"]:
            content = read_file(WORKFLOWS_DIR / fname)
            assert "secrets." in content, \
                f"{fname} should reference GitHub secrets"

    def test_no_password_in_deploy_files(self):
        for f in list(WORKFLOWS_DIR.glob("*.yml")) + \
                 list(DEPLOY_DIR.rglob("*.bicep")) + \
                 list(DEPLOY_DIR.rglob("*.tf")):
            content = f.read_text()
            # No plain password assignments (password = "value")
            assert not re.search(r'password\s*=\s*"[^"]{8,}"', content, re.IGNORECASE), \
                f"Possible hardcoded password in {f.name}"
