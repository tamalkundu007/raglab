# RAGLab — GCP Deployment (R7 Stub)

This directory is intentionally empty in R1–R6.

GCP deployment activates in **Release 7** alongside:
- `GCSStorageBackend` (currently stubbed in storage-service)
- `VertexEmbedder` (currently stubbed in embedding-service)
- `VertexProvider` (currently stubbed in llm-service)
- `auth-service` (JWT + multi-tenant isolation)

## R7 Target Architecture

| Component | GCP Service |
|-----------|-------------|
| Container registry | Google Artifact Registry (GAR) |
| Runtime | Cloud Run (serverless per-service) |
| Vector store | Vertex AI Vector Search (or Qdrant on GKE) |
| Storage | Google Cloud Storage (GCS) |
| Secrets | Secret Manager |
| Auth | Identity Platform + Cloud Endpoints |
| Infra-as-code | Terraform with GCS backend |
| CI/CD auth | Workload Identity Federation (no static keys) |

## CI/CD Workflow

See `.github/workflows/cd-gcp.yml` — structure is defined, triggers are disabled.
Set `on:` triggers in R7 after GCP credentials and Workload Identity are configured.

## Credentials Setup (R7)

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create raglab-github \
  --location global --display-name "RAGLab GitHub Actions"

# Create provider
gcloud iam workload-identity-pools providers create-oidc raglab-github-provider \
  --location global \
  --workload-identity-pool raglab-github \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository"

# Grant access
gcloud iam service-accounts add-iam-policy-binding \
  raglab-deploy@PROJECT.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/raglab-github/attribute.repository/tamalkundu007/raglab"
```
