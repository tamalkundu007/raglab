# RAGLab CI/CD Setup Guide

Three pipelines in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push + PR | Lint (ruff) + test (pytest) + Docker build check |
| `cd-azure.yml` | Push to `main` | Build → ACR → Azure Container Apps |
| `cd-aws.yml` | Push to `main` | Build → ECR → ECS Fargate |
| `cd-gcp.yml` | **Disabled** | Stub — activates in R7 |

---

## CI — Required Setup

CI has **zero cloud credentials** — all tests are infra-free. No secrets required.

Optionally add:
```
CODECOV_TOKEN   — for coverage reporting to codecov.io
```

---

## Azure CD — Required GitHub Secrets

Configure under `Settings → Secrets → Actions`:

| Secret | Value |
|--------|-------|
| `AZURE_CLIENT_ID` | Service principal / OIDC client ID |
| `AZURE_TENANT_ID` | Azure tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | Target resource group (e.g. `raglab-prod`) |
| `ACR_LOGIN_SERVER` | ACR login server (e.g. `raglab.azurecr.io`) |
| `ACA_ENVIRONMENT` | Container Apps environment name |
| `RAGLAB_AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `RAGLAB_AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT` | Chat model deployment name |
| `RAGLAB_AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name |

### Azure OIDC Setup (no long-lived secrets)

```bash
# 1. Create app registration
az ad app create --display-name "raglab-github-actions"

# 2. Create service principal
az ad sp create --id <app-id>

# 3. Add federated credential
az ad app federated-credential create \
  --id <app-id> \
  --parameters '{
    "name": "raglab-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:tamalkundu007/raglab:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'

# 4. Assign Contributor role
az role assignment create \
  --role Contributor \
  --assignee <app-id> \
  --scope /subscriptions/<subscription-id>/resourceGroups/raglab-prod

# 5. Assign AcrPush role
az role assignment create \
  --role AcrPush \
  --assignee <app-id> \
  --scope /subscriptions/<subscription-id>/resourceGroups/raglab-prod/providers/Microsoft.ContainerRegistry/registries/raglab
```

### Azure Infrastructure (first deploy)

```bash
# Deploy Bicep template (once, before CD pipeline)
az deployment group create \
  --resource-group raglab-prod \
  --template-file deploy/azure/main.bicep \
  --parameters \
    acrLoginServer=raglab.azurecr.io \
    imageTag=latest \
    acaEnvironment=raglab-env
```

---

## AWS CD — Required GitHub Secrets

| Secret | Value |
|--------|-------|
| `AWS_ACCOUNT_ID` | 12-digit AWS account ID |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ROLE_ARN` | IAM role ARN for OIDC (e.g. `arn:aws:iam::123456789012:role/raglab-github`) |
| `ECS_CLUSTER` | ECS cluster name (e.g. `raglab-cluster`) |
| `ECR_REGISTRY` | ECR registry URL (e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com`) |

### AWS OIDC Setup (no AWS_ACCESS_KEY_ID stored)

```bash
# 1. Create OIDC identity provider in IAM
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  --client-id-list sts.amazonaws.com

# 2. Create IAM role with trust policy
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:tamalkundu007/raglab:*"
      }
    }
  }]
}
EOF

aws iam create-role \
  --role-name raglab-github \
  --assume-role-policy-document file://trust-policy.json

# 3. Attach required policies
aws iam attach-role-policy \
  --role-name raglab-github \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

aws iam attach-role-policy \
  --role-name raglab-github \
  --policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess
```

### AWS Infrastructure (first deploy)

```bash
# Deploy with Terraform
cd deploy/aws
terraform init
terraform apply \
  -var="ecr_registry=123456789012.dkr.ecr.us-east-1.amazonaws.com" \
  -var="image_tag=latest"
```

---

## Security Principles

1. **No static cloud credentials in GitHub secrets** — both Azure and AWS use OIDC federated identity (short-lived tokens, no rotation needed).
2. **No LLM API keys in environment variables** — Azure: Key Vault references in ACA secrets. AWS: Secrets Manager references in ECS task definitions.
3. **No secrets in Docker images** — credentials injected at runtime via cloud-native secret stores.
4. **Rotate any token ever pasted in chat or logs** — immediately.

---

## Branch Strategy

| Branch | CI | CD |
|--------|----|----|
| `main` | ✅ | ✅ Azure + AWS |
| `develop` | ✅ | ❌ |
| `release/**` | ✅ | ❌ |
| PR | ✅ | ❌ |

---

## Monitoring After Deploy

```bash
# Azure — check Container App logs
az containerapp logs show \
  --name raglab-api-gateway \
  --resource-group raglab-prod \
  --follow

# AWS — check ECS logs
aws logs tail /ecs/raglab --follow --filter-pattern "api-gateway"
```
