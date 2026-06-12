# RAGLab — GCP Infrastructure (R7 — Activated)
# GKE Autopilot + Cloud SQL + Memorystore Redis + Artifact Registry + Secret Manager
# Remote state: GCS bucket. Auth: Workload Identity Federation.

terraform {
  required_version = ">= 1.7"
  backend "gcs" {}
  required_providers {
    google     = { source = "hashicorp/google",     version = "~> 5.0" }
    kubernetes = { source = "hashicorp/kubernetes",  version = "~> 2.30" }
    helm       = { source = "hashicorp/helm",        version = "~> 2.14" }
  }
}

variable "project_id"  { type = string }
variable "region"      { type = string; default = "us-central1" }
variable "environment" { type = string; default = "staging" }
variable "image_tag"   { type = string; default = "latest" }

locals {
  prefix = "raglab-${var.environment}"
  labels = { project = "raglab", environment = var.environment, managed_by = "terraform" }
}

provider "google" { project = var.project_id; region = var.region }

resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com", "sqladmin.googleapis.com", "redis.googleapis.com",
    "artifactregistry.googleapis.com", "secretmanager.googleapis.com",
    "iam.googleapis.com", "cloudresourcemanager.googleapis.com",
    "servicenetworking.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "raglab" {
  name = "${local.prefix}-vpc"; auto_create_subnetworks = false
  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "gke" {
  name          = "${local.prefix}-gke-subnet"
  network       = google_compute_network.raglab.id
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  secondary_ip_range { range_name = "pods";     ip_cidr_range = "10.1.0.0/16" }
  secondary_ip_range { range_name = "services"; ip_cidr_range = "10.2.0.0/20" }
}

resource "google_artifact_registry_repository" "raglab" {
  location = var.region; repository_id = "raglab"; format = "DOCKER"
  description = "RAGLab container images"; labels = local.labels
  depends_on  = [google_project_service.apis]
}

resource "google_container_cluster" "raglab" {
  name     = "${local.prefix}-gke"
  location = var.region
  enable_autopilot = true
  network    = google_compute_network.raglab.id
  subnetwork = google_compute_subnetwork.gke.id
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  workload_identity_config { workload_pool = "${var.project_id}.svc.id.goog" }
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }
  master_authorized_networks_config { cidr_blocks { cidr_block = "0.0.0.0/0"; display_name = "all" } }
  deletion_protection = var.environment == "prod"
  depends_on = [google_project_service.apis]
}

resource "google_sql_database_instance" "raglab" {
  name             = "${local.prefix}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region
  deletion_protection = var.environment == "prod"
  settings {
    tier              = var.environment == "prod" ? "db-n1-standard-2" : "db-f1-micro"
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_autoresize   = true; disk_size = 20
    backup_configuration {
      enabled = true; point_in_time_recovery_enabled = true; start_time = "02:00"
      backup_retention_settings { retained_backups = 7 }
    }
    ip_configuration { ipv4_enabled = false; private_network = google_compute_network.raglab.id; require_ssl = true }
    database_flags { name = "cloudsql.iam_authentication"; value = "on" }
  }
  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "raglab" { name = "raglab"; instance = google_sql_database_instance.raglab.name }

resource "google_redis_instance" "raglab" {
  name           = "${local.prefix}-redis"
  tier           = var.environment == "prod" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.environment == "prod" ? 4 : 1
  region         = var.region
  authorized_network = google_compute_network.raglab.id
  redis_version  = "REDIS_7_0"
  redis_configs  = { maxmemory-policy = "allkeys-lru" }
  labels         = local.labels
  depends_on     = [google_project_service.apis]
}

resource "google_secret_manager_secret" "redis_url" {
  secret_id = "raglab-redis-url"; labels = local.labels
  replication { auto {} }; depends_on = [google_project_service.apis]
}
resource "google_secret_manager_secret_version" "redis_url" {
  secret      = google_secret_manager_secret.redis_url.id
  secret_data = "redis://${google_redis_instance.raglab.host}:${google_redis_instance.raglab.port}/0"
}

resource "google_service_account" "raglab_workload" {
  account_id   = "${local.prefix}-workload"
  display_name = "RAGLab workload identity SA"
}
resource "google_project_iam_member" "workload_secret_accessor" {
  project = var.project_id; role = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.raglab_workload.email}"
}
resource "google_project_iam_member" "workload_sql_client" {
  project = var.project_id; role = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.raglab_workload.email}"
}
resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.raglab_workload.id
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[raglab/raglab-workload]"
}

output "gke_cluster_name"   { value = google_container_cluster.raglab.name }
output "gar_repository"     { value = google_artifact_registry_repository.raglab.id }
output "redis_host"         { value = google_redis_instance.raglab.host; sensitive = true }
output "db_connection_name" { value = google_sql_database_instance.raglab.connection_name }
output "workload_sa_email"  { value = google_service_account.raglab_workload.email }
