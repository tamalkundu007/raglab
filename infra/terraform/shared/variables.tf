# RAGLab — Shared Terraform Variables
# Common variable definitions reused across Azure, AWS, and GCP modules.
# Each cloud root module imports these via variable blocks.

variable "environment" {
  type        = string
  description = "Deployment environment (dev, staging, prod)"
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  type        = string
  description = "Project identifier used in resource naming"
  default     = "raglab"
}

variable "image_tag" {
  type        = string
  description = "Docker image tag (git SHA) for all service deployments"
}

# ── Service ports ─────────────────────────────────────────────────────────────

variable "service_ports" {
  type        = map(number)
  description = "HTTP port each service listens on"
  default = {
    api-gateway  = 8000
    ingestion    = 8001
    embedding    = 8002
    indexing     = 8003
    retrieval    = 8004
    llm          = 8005
    pipeline     = 8006
    storage      = 8008
    ui           = 8009
    graph        = 8010
    observability = 8011
    auth         = 8012
  }
}

# ── Sizing ────────────────────────────────────────────────────────────────────

variable "node_pool_vm_size" {
  type        = map(string)
  description = "VM / instance size per cloud for the default node pool"
  default = {
    azure = "Standard_D4s_v5"  # 4 vCPU, 16 GB
    aws   = "m7i.xlarge"       # 4 vCPU, 16 GB
    gcp   = "n2-standard-4"    # 4 vCPU, 16 GB
  }
}

variable "node_pool_min_count" {
  type        = number
  description = "Minimum nodes in the cluster node pool"
  default     = 2
}

variable "node_pool_max_count" {
  type        = number
  description = "Maximum nodes in the cluster node pool"
  default     = 10
}

# ── Database ──────────────────────────────────────────────────────────────────

variable "postgres_sku" {
  type        = map(string)
  description = "Postgres compute tier per cloud"
  default = {
    azure = "GP_Standard_D4s_v3"  # General Purpose, 4 vCores
    aws   = "db.m7g.large"        # 2 vCPU, 8 GB
    gcp   = "db-custom-4-15360"   # 4 vCPU, 15 GB
  }
}

variable "postgres_storage_gb" {
  type        = number
  description = "Postgres storage in GB"
  default     = 128
}

# ── Qdrant ────────────────────────────────────────────────────────────────────

variable "qdrant_replicas" {
  type        = number
  description = "Number of Qdrant replicas in the cluster"
  default     = 2
}

variable "qdrant_storage_gb" {
  type        = number
  description = "Persistent storage per Qdrant replica in GB"
  default     = 50
}

# ── RabbitMQ ──────────────────────────────────────────────────────────────────

variable "rabbitmq_replicas" {
  type        = number
  description = "Number of RabbitMQ replicas"
  default     = 3  # minimum for HA quorum queues
}

# ── Tags / labels ─────────────────────────────────────────────────────────────

variable "common_tags" {
  type        = map(string)
  description = "Tags applied to all cloud resources"
  default = {
    Project     = "raglab"
    ManagedBy   = "terraform"
    Repository  = "tamalkundu007/raglab"
  }
}
