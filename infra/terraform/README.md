# RAGLab — Terraform Infrastructure Guide

Three cloud targets in `infra/terraform/`:

| Directory | Status | Runtime | Database | Registry |
|-----------|--------|---------|----------|----------|
| `azure/`  | ✅ Active | AKS | Azure PostgreSQL Flexible | ACR |
| `aws/`    | ✅ Active | EKS (Managed NG) | RDS PostgreSQL Multi-AZ | ECR |
| `gcp/`    | 🔜 R7 stub | GKE Autopilot | Cloud SQL | Artifact Registry |

---

## Prerequisites

- Terraform >= 1.7
- Cloud CLI authenticated (az login / aws configure / gcloud auth)
- kubectl installed
- A remote state backend configured (see below)

---

## Remote State Setup

### Azure
```bash
# Create storage account for state
az group create -n raglab-tfstate -l eastus
az storage account create -n raglabtfstate -g raglab-tfstate --sku Standard_LRS
az storage container create -n tfstate --account-name raglabtfstate

# Uncomment backend block in azure/main.tf then:
cd infra/terraform/azure
terraform init
```

### AWS
```bash
# Create S3 bucket + DynamoDB table for state
aws s3api create-bucket --bucket raglab-terraform-state --region us-east-1
aws dynamodb create-table \
  --table-name raglab-terraform-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# Uncomment backend block in aws/main.tf then:
cd infra/terraform/aws
terraform init
```

---

## Azure Deployment

```bash
cd infra/terraform/azure

# Plan
terraform plan \
  -var="image_tag=$(git rev-parse HEAD)" \
  -var="environment=prod" \
  -var="location=eastus" \
  -out=tfplan

# Apply
terraform apply tfplan

# Get AKS credentials
az aks get-credentials \
  --resource-group $(terraform output -raw aks_resource_group) \
  --name $(terraform output -raw aks_cluster_name)

# Apply workloads
terraform apply -target=module.workloads \
  -var="image_tag=$(git rev-parse HEAD)" \
  -var="acr_login_server=$(terraform output -raw acr_login_server)"
```

### Azure — Required Secrets (Key Vault)
After `terraform apply`, populate Key Vault secrets. Never pass via CLI or tfvars:

```bash
KV=$(terraform output -raw key_vault_uri)
az keyvault secret set --vault-name raglab-prod-kv --name azure-openai-key     --value "YOUR_KEY"
az keyvault secret set --vault-name raglab-prod-kv --name azure-openai-endpoint --value "YOUR_ENDPOINT"
az keyvault secret set --vault-name raglab-prod-kv --name azure-chat-deployment  --value "gpt-4o"
az keyvault secret set --vault-name raglab-prod-kv --name azure-embed-deployment --value "text-embedding-3-small"
```

---

## AWS Deployment

```bash
cd infra/terraform/aws

# Plan
terraform plan \
  -var="image_tag=$(git rev-parse HEAD)" \
  -var="environment=prod" \
  -var="aws_region=us-east-1" \
  -out=tfplan

# Apply (creates VPC, EKS, RDS, ECR, Secrets Manager)
terraform apply tfplan

# Get EKS credentials
aws eks update-kubeconfig \
  --name $(terraform output -raw eks_cluster_name) \
  --region us-east-1
```

### AWS — Required Secrets (Secrets Manager)
Terraform creates placeholder secrets. Populate after apply:

```bash
aws secretsmanager put-secret-value \
  --secret-id raglab/azure-openai-key \
  --secret-string "YOUR_AZURE_OPENAI_KEY"

aws secretsmanager put-secret-value \
  --secret-id raglab/azure-openai-endpoint \
  --secret-string "YOUR_ENDPOINT"

aws secretsmanager put-secret-value \
  --secret-id raglab/anthropic-api-key \
  --secret-string "YOUR_ANTHROPIC_KEY"
```

---

## Shared Variables

All variables are documented in `shared/variables.tf`. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `environment` | `prod` | dev / staging / prod |
| `image_tag` | — | **Required.** Git SHA of the build |
| `node_pool_min_count` | 2 | Cluster auto-scaling floor |
| `node_pool_max_count` | 10 | Cluster auto-scaling ceiling |
| `postgres_storage_gb` | 128 | Database disk size |
| `qdrant_replicas` | 2 | Qdrant StatefulSet replicas |
| `rabbitmq_replicas` | 3 | RabbitMQ HA quorum replicas |

---

## Security Principles

1. **No static cloud credentials anywhere.** Azure uses OIDC (`use_oidc = true`). AWS uses OIDC identity provider + IAM role assumption.
2. **Secrets in managed secret stores.** Azure: Key Vault. AWS: Secrets Manager. GCP (R7): Secret Manager. Never in tfvars, never in environment variables on CI runners, never in Docker images.
3. **State is encrypted.** Azure: Azure Storage with server-side encryption. AWS: S3 with AES-256 + DynamoDB lock.
4. **Private networking.** Postgres, RDS, and Qdrant are on private subnets. Only the api-gateway and ui services expose LoadBalancer services.
5. **Workload Identity.** AKS pods use Azure Workload Identity (OIDC). EKS pods use IAM Roles for Service Accounts (IRSA). No static credentials injected into pods.

---

## GCP — R7

GCP infrastructure is defined but disabled. See `gcp/main.tf` for the complete stubbed structure. Activates in R7 alongside:
- `auth-service` (JWT + multi-tenant RBAC)
- `GCSStorageBackend`
- `VertexEmbedder` / `VertexProvider`
- `.github/workflows/cd-gcp.yml` (add `on:` triggers in R7)
