# RAGLab — Azure Kubernetes Scaling Refinements (R5)
#
# Adds to azure/workloads.tf:
#   - HorizontalPodAutoscaler for each service (CPU + memory triggers)
#   - PodDisruptionBudgets (maintain quorum during node drain)
#   - Redis StatefulSet (embedding cache + future semantic cache)
#   - Node taint + toleration for embedding workload isolation
#   - Resource quotas for raglab namespace

terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
  }
}

variable "aks_host"                   { type = string; sensitive = true }
variable "aks_client_certificate"     { type = string; sensitive = true }
variable "aks_client_key"             { type = string; sensitive = true }
variable "aks_cluster_ca_certificate" { type = string; sensitive = true }
variable "image_tag"                  { type = string }
variable "environment"                { type = string; default = "prod" }

locals {
  # Services that warrant HPA (stateless, CPU-bound)
  hpa_services = {
    "api-gateway"  = { min = 2, max = 8,  cpu_threshold = 60, mem_threshold = 80 }
    "embedding"    = { min = 2, max = 10, cpu_threshold = 70, mem_threshold = 85 }
    "retrieval"    = { min = 2, max = 8,  cpu_threshold = 65, mem_threshold = 80 }
    "llm"          = { min = 2, max = 6,  cpu_threshold = 70, mem_threshold = 85 }
    "pipeline"     = { min = 2, max = 8,  cpu_threshold = 65, mem_threshold = 80 }
    "graph"        = { min = 2, max = 6,  cpu_threshold = 65, mem_threshold = 80 }
    "ui"           = { min = 2, max = 4,  cpu_threshold = 70, mem_threshold = 80 }
    "auth"         = { min = 2, max = 6,  cpu_threshold = 60, mem_threshold = 75 }
  }

  # Services that need PDB (prevent all pods being drained simultaneously)
  pdb_services = [
    "api-gateway", "embedding", "retrieval", "llm",
    "pipeline", "graph", "auth", "ui",
  ]
}

# ── Namespace resource quota ───────────────────────────────────────────────────

resource "kubernetes_resource_quota" "raglab" {
  metadata {
    name      = "raglab-quota"
    namespace = "raglab"
  }
  spec {
    hard = {
      "requests.cpu"    = "20"
      "requests.memory" = "40Gi"
      "limits.cpu"      = "40"
      "limits.memory"   = "80Gi"
      "pods"            = "100"
    }
  }
}

# ── HorizontalPodAutoscaler ────────────────────────────────────────────────────

resource "kubernetes_horizontal_pod_autoscaler_v2" "services" {
  for_each = local.hpa_services

  metadata {
    name      = "raglab-${each.key}-hpa"
    namespace = "raglab"
  }

  spec {
    min_replicas = each.value.min
    max_replicas = each.value.max

    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "raglab-${each.key}"
    }

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = each.value.cpu_threshold
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = each.value.mem_threshold
        }
      }
    }

    behavior {
      scale_up {
        stabilization_window_seconds = 60
        select_policy                = "Max"
        policy {
          type           = "Percent"
          value          = 100
          period_seconds = 60
        }
      }
      scale_down {
        stabilization_window_seconds = 300  # wait 5 min before scaling down
        select_policy                = "Min"
        policy {
          type           = "Pods"
          value          = 1
          period_seconds = 120
        }
      }
    }
  }
}

# ── PodDisruptionBudgets ───────────────────────────────────────────────────────

resource "kubernetes_pod_disruption_budget_v1" "services" {
  for_each = toset(local.pdb_services)

  metadata {
    name      = "raglab-${each.key}-pdb"
    namespace = "raglab"
  }

  spec {
    min_available = "50%"  # at least half pods always available during drain

    selector {
      match_labels = {
        app = "raglab-${each.key}"
      }
    }
  }
}

# ── Redis (Embedding cache + R6 semantic cache) ────────────────────────────────

resource "helm_release" "redis" {
  name       = "redis"
  namespace  = "raglab"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "redis"
  version    = "19.6.2"

  # Sentinel mode for HA (1 master + 2 replicas + 3 sentinels)
  set { name = "architecture";                  value = "replication" }
  set { name = "sentinel.enabled";              value = "true" }
  set { name = "replica.replicaCount";          value = "2" }
  set { name = "persistence.storageClass";      value = "managed-premium" }
  set { name = "persistence.size";              value = "8Gi" }
  set { name = "master.resources.requests.memory"; value = "256Mi" }
  set { name = "master.resources.limits.memory";   value = "512Mi" }
  set { name = "replica.resources.requests.memory"; value = "256Mi" }
  set { name = "replica.resources.limits.memory";   value = "512Mi" }

  # TLS + auth via Kubernetes Secret (Key Vault CSI in production)
  set { name = "auth.enabled";  value = "true" }
  set { name = "tls.enabled";   value = "false" }  # TLS terminated at service mesh
}

# ── Embedding workload node pool (dedicated, GPU-optional) ────────────────────
# Separate node pool for embedding workloads — allows GPU nodes without
# wasting GPU capacity on non-embedding services.

resource "azurerm_kubernetes_cluster_node_pool" "embedding" {
  name                  = "embedding"
  kubernetes_cluster_id = var.aks_cluster_id
  vm_size               = var.embedding_vm_size
  node_count            = 2
  enable_auto_scaling   = true
  min_count             = 2
  max_count             = 6
  os_disk_size_gb       = 128
  vnet_subnet_id        = var.aks_subnet_id

  node_taints = ["workload=embedding:NoSchedule"]
  node_labels = { workload = "embedding", service = "raglab" }

  tags = { Environment = var.environment, ManagedBy = "terraform" }

  lifecycle { ignore_changes = [node_count] }
}

variable "aks_cluster_id"    { type = string }
variable "embedding_vm_size" { type = string; default = "Standard_D4s_v5" }
variable "aks_subnet_id"     { type = string }

# ── Network Policy (deny all by default, allow raglab internal) ───────────────

resource "kubernetes_network_policy" "raglab_default_deny" {
  metadata {
    name      = "default-deny"
    namespace = "raglab"
  }
  spec {
    pod_selector {}
    policy_types = ["Ingress", "Egress"]
  }
}

resource "kubernetes_network_policy" "raglab_internal" {
  metadata {
    name      = "allow-raglab-internal"
    namespace = "raglab"
  }
  spec {
    pod_selector {}
    ingress {
      from {
        namespace_selector {
          match_labels = { name = "raglab" }
        }
      }
    }
    egress {
      to {
        namespace_selector {
          match_labels = { name = "raglab" }
        }
      }
    }
    # Allow egress to external APIs (OpenAI, Anthropic endpoints)
    egress {
      ports {
        port     = "443"
        protocol = "TCP"
      }
    }
    policy_types = ["Ingress", "Egress"]
  }
}
