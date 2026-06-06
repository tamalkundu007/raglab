# RAGLab — GCP Infrastructure (Terraform Stub)
#
# THIS MODULE IS INTENTIONALLY DISABLED — activates in R7.
#
# Structure is defined so R7 implementers have a complete starting point.
# No resources will be created until this file is wired to a root module
# and the provider block is configured with real credentials.
#
# R7 target architecture:
#   - GKE Autopilot cluster (serverless node management)
#   - Cloud SQL for PostgreSQL (private IP, HA)
#   - Qdrant on GKE (StatefulSet via Helm)
#   - Cloud Pub/Sub (replaces RabbitMQ on GCP)
#   - Google Artifact Registry (GAR)
#   - Secret Manager for all credentials
#   - Workload Identity Federation for GKE pods
#   - VPC with Private Service Connect
#
# R7 dependencies (not yet active):
#   - GCSStorageBackend in storage-service
#   - VertexEmbedder in embedding-service
#   - VertexProvider in llm-service
#   - auth-service (JWT + multi-tenant RBAC)
#   - GCP CI/CD pipeline (.github/workflows/cd-gcp.yml, currently no triggers)

terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  # backend "gcs" {
  #   bucket = "raglab-terraform-state"
  #   prefix = "raglab/gcp"
  # }
}

# Provider configured via Workload Identity Federation in R7.
# No service account key files — ever.
# provider "google" {
#   project = var.gcp_project_id
#   region  = var.gcp_region
# }

# ── Variables (defined, not yet used) ────────────────────────────────────────

variable "gcp_project_id" {
  type        = string
  description = "GCP project ID"
  default     = ""
}

variable "gcp_region" {
  type        = string
  description = "GCP region for resource deployment"
  default     = "us-central1"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "image_tag" {
  type        = string
  description = "Docker image tag (git SHA)"
  default     = "latest"
}

variable "gke_node_count" {
  type        = number
  description = "Initial node count for GKE node pool (Autopilot manages this automatically)"
  default     = 3
}

variable "cloudsql_tier" {
  type        = string
  description = "Cloud SQL machine tier"
  default     = "db-custom-4-15360"  # 4 vCPU, 15 GB
}

# ── R7 Resources (commented — not created until R7) ──────────────────────────

# VPC
# resource "google_compute_network" "main" { ... }

# GKE Autopilot
# resource "google_container_cluster" "main" {
#   name     = "raglab-${var.environment}"
#   location = var.gcp_region
#   enable_autopilot = true
#   ...
# }

# Cloud SQL
# resource "google_sql_database_instance" "main" {
#   name             = "raglab-${var.environment}-postgres"
#   database_version = "POSTGRES_16"
#   settings { ... }
# }

# Google Artifact Registry
# resource "google_artifact_registry_repository" "main" {
#   repository_id = "raglab"
#   format        = "DOCKER"
#   location      = var.gcp_region
# }

# Secret Manager
# resource "google_secret_manager_secret" "anthropic_key" { ... }

# Workload Identity
# resource "google_service_account" "raglab_workload" { ... }
# resource "google_service_account_iam_binding" "workload_identity" { ... }

# ── Outputs (stubbed) ─────────────────────────────────────────────────────────

output "r7_stub_message" {
  value       = "GCP infrastructure activates in R7. See deploy/gcp/README.md for setup guide."
  description = "Reminder that this module is not yet active"
}
