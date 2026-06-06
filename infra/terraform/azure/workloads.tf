# RAGLab — Azure Kubernetes Workloads
# Deploys Qdrant, RabbitMQ, and all RAGLab microservices to AKS.
# Applied after main.tf provisions the cluster.

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

provider "kubernetes" {
  host                   = var.aks_host
  client_certificate     = base64decode(var.aks_client_certificate)
  client_key             = base64decode(var.aks_client_key)
  cluster_ca_certificate = base64decode(var.aks_cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = var.aks_host
    client_certificate     = base64decode(var.aks_client_certificate)
    client_key             = base64decode(var.aks_client_key)
    cluster_ca_certificate = base64decode(var.aks_cluster_ca_certificate)
  }
}

variable "aks_host"                  { type = string; sensitive = true }
variable "aks_client_certificate"    { type = string; sensitive = true }
variable "aks_client_key"            { type = string; sensitive = true }
variable "aks_cluster_ca_certificate"{ type = string; sensitive = true }
variable "acr_login_server"          { type = string }
variable "image_tag"                 { type = string }
variable "qdrant_replicas"           { type = number; default = 2 }
variable "rabbitmq_replicas"         { type = number; default = 3 }
variable "key_vault_uri"             { type = string }

# ── Namespace ─────────────────────────────────────────────────────────────────

resource "kubernetes_namespace" "raglab" {
  metadata { name = "raglab" }
}

# ── Qdrant StatefulSet ────────────────────────────────────────────────────────

resource "helm_release" "qdrant" {
  name       = "qdrant"
  namespace  = kubernetes_namespace.raglab.metadata[0].name
  repository = "https://qdrant.github.io/qdrant-helm"
  chart      = "qdrant"
  version    = "0.9.0"

  set { name = "replicaCount";               value = var.qdrant_replicas }
  set { name = "persistence.storageClass";   value = "managed-premium" }
  set { name = "persistence.size";           value = "50Gi" }
  set { name = "config.collection.replication_factor"; value = "2" }
  set { name = "service.type";               value = "ClusterIP" }
  set { name = "resources.requests.memory";  value = "2Gi" }
  set { name = "resources.requests.cpu";     value = "500m" }
  set { name = "resources.limits.memory";    value = "4Gi" }
  set { name = "resources.limits.cpu";       value = "2000m" }
}

# ── RabbitMQ StatefulSet ──────────────────────────────────────────────────────

resource "helm_release" "rabbitmq" {
  name       = "rabbitmq"
  namespace  = kubernetes_namespace.raglab.metadata[0].name
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "rabbitmq"
  version    = "14.4.2"

  set { name = "replicaCount";              value = var.rabbitmq_replicas }
  set { name = "persistence.storageClass"; value = "managed-premium" }
  set { name = "persistence.size";         value = "8Gi" }
  set { name = "clustering.enabled";       value = "true" }
  set { name = "auth.username";            value = "raglab" }
  # Password sourced from Key Vault via Workload Identity at pod startup
  set { name = "resources.requests.memory"; value = "512Mi" }
  set { name = "resources.limits.memory";   value = "1Gi" }
}

# ── RAGLab services ───────────────────────────────────────────────────────────

locals {
  services = {
    "api-gateway"  = { port = 8000, replicas = 2, cpu = "250m", memory = "512Mi", external = true }
    "ingestion"    = { port = 8001, replicas = 2, cpu = "250m", memory = "512Mi", external = false }
    "embedding"    = { port = 8002, replicas = 2, cpu = "1000m", memory = "2Gi",  external = false }
    "indexing"     = { port = 8003, replicas = 2, cpu = "250m", memory = "512Mi", external = false }
    "retrieval"    = { port = 8004, replicas = 2, cpu = "500m", memory = "1Gi",   external = false }
    "llm"          = { port = 8005, replicas = 2, cpu = "500m", memory = "1Gi",   external = false }
    "pipeline"     = { port = 8006, replicas = 2, cpu = "500m", memory = "1Gi",   external = false }
    "storage"      = { port = 8008, replicas = 1, cpu = "125m", memory = "256Mi", external = false }
    "ui"           = { port = 8009, replicas = 2, cpu = "125m", memory = "256Mi", external = true  }
    "graph"        = { port = 8010, replicas = 2, cpu = "500m", memory = "1Gi",   external = false }
    "observability"= { port = 8011, replicas = 1, cpu = "125m", memory = "256Mi", external = false }
    "auth"         = { port = 8012, replicas = 2, cpu = "250m", memory = "512Mi", external = false }
  }
}

resource "kubernetes_deployment" "services" {
  for_each = local.services

  metadata {
    name      = "raglab-${each.key}"
    namespace = kubernetes_namespace.raglab.metadata[0].name
    labels    = { app = "raglab-${each.key}", version = var.image_tag }
  }

  spec {
    replicas = each.value.replicas

    selector {
      match_labels = { app = "raglab-${each.key}" }
    }

    template {
      metadata {
        labels = { app = "raglab-${each.key}", version = var.image_tag }
      }

      spec {
        container {
          name  = each.key
          image = "${var.acr_login_server}/raglab/${each.key}:${var.image_tag}"

          port { container_port = each.value.port }

          resources {
            requests = { cpu = each.value.cpu, memory = each.value.memory }
            limits   = { memory = each.value.memory }
          }

          env {
            name  = "RAGLAB_SERVICE_NAME"
            value = each.key
          }
          env {
            name  = "RAGLAB_JSON_LOGS"
            value = "true"
          }

          # Secrets sourced from Key Vault via Azure Workload Identity (CSI driver)
          env_from {
            secret_ref { name = "raglab-secrets" }
          }

          liveness_probe {
            http_get { path = "/health"; port = each.value.port }
            initial_delay_seconds = 20
            period_seconds        = 15
            failure_threshold     = 3
          }

          readiness_probe {
            http_get { path = "/health"; port = each.value.port }
            initial_delay_seconds = 10
            period_seconds        = 10
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "services" {
  for_each = local.services

  metadata {
    name      = "raglab-${each.key}"
    namespace = kubernetes_namespace.raglab.metadata[0].name
  }

  spec {
    selector = { app = "raglab-${each.key}" }
    port {
      port        = each.value.port
      target_port = each.value.port
    }
    type = each.value.external ? "LoadBalancer" : "ClusterIP"
  }
}
