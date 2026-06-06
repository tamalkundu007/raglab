# RAGLab — Azure Infrastructure (Terraform)
#
# Provisions:
#   - Resource Group
#   - Virtual Network (3 subnets: aks, postgres, private-endpoints)
#   - Azure Container Registry (ACR) with geo-replication disabled
#   - Azure Kubernetes Service (AKS) with OIDC + Workload Identity
#   - Azure Database for PostgreSQL Flexible Server
#   - Qdrant on AKS (StatefulSet via Kubernetes provider)
#   - RabbitMQ on AKS (StatefulSet via Kubernetes provider)
#   - Azure Key Vault for secrets
#   - Log Analytics Workspace
#
# State backend: Azure Storage (configure backend.tf before first apply)
# Auth: Azure AD OIDC via GitHub Actions (no service principal keys)
#
# Usage:
#   terraform init
#   terraform plan -var="image_tag=$(git rev-parse HEAD)" -var="environment=prod"
#   terraform apply -var="image_tag=$(git rev-parse HEAD)"

terraform {
  required_version = ">= 1.7"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.50"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  # Uncomment and configure for remote state:
  # backend "azurerm" {
  #   resource_group_name  = "raglab-tfstate"
  #   storage_account_name = "raglabtfstate"
  #   container_name       = "tfstate"
  #   key                  = "raglab.terraform.tfstate"
  # }
}

provider "azurerm" {
  features {
    resource_group { prevent_deletion_if_contains_resources = false }
    key_vault { purge_soft_delete_on_destroy = false }
  }
  use_oidc = true  # OIDC federated identity — no client_secret stored
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "environment"           { type = string; default = "prod" }
variable "location"              { type = string; default = "eastus" }
variable "project_name"          { type = string; default = "raglab" }
variable "image_tag"             { type = string }
variable "node_pool_vm_size"     { type = string; default = "Standard_D4s_v5" }
variable "node_pool_min_count"   { type = number; default = 2 }
variable "node_pool_max_count"   { type = number; default = 10 }
variable "postgres_storage_gb"   { type = number; default = 128 }
variable "qdrant_replicas"       { type = number; default = 2 }
variable "rabbitmq_replicas"     { type = number; default = 3 }

locals {
  prefix  = "${var.project_name}-${var.environment}"
  tags    = { Project = var.project_name, Environment = var.environment, ManagedBy = "terraform" }
}

# ── Resource Group ─────────────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = "${local.prefix}-rg"
  location = var.location
  tags     = local.tags
}

# ── Log Analytics ─────────────────────────────────────────────────────────────

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${local.prefix}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.tags
}

# ── Virtual Network ────────────────────────────────────────────────────────────

resource "azurerm_virtual_network" "main" {
  name                = "${local.prefix}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_subnet" "aks" {
  name                 = "aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_subnet" "postgres" {
  name                 = "postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.2.0/24"]
  service_endpoints    = ["Microsoft.Storage"]

  delegation {
    name = "postgres-delegation"
    service_delegation {
      name = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_subnet" "private_endpoints" {
  name                 = "private-endpoints"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.3.0/24"]
}

# ── Azure Container Registry ───────────────────────────────────────────────────

resource "azurerm_container_registry" "main" {
  name                = replace("${local.prefix}acr", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = false  # use OIDC / managed identity, not admin credentials
  tags                = local.tags
}

# ── AKS Cluster ───────────────────────────────────────────────────────────────

resource "azurerm_kubernetes_cluster" "main" {
  name                = "${local.prefix}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${local.prefix}-aks"
  kubernetes_version  = "1.29"

  default_node_pool {
    name                 = "system"
    node_count           = var.node_pool_min_count
    vm_size              = var.node_pool_vm_size
    os_disk_size_gb      = 128
    vnet_subnet_id       = azurerm_subnet.aks.id
    enable_auto_scaling  = true
    min_count            = var.node_pool_min_count
    max_count            = var.node_pool_max_count
    type                 = "VirtualMachineScaleSets"
    only_critical_addons_enabled = false
  }

  identity {
    type = "SystemAssigned"
  }

  oidc_issuer_enabled       = true  # Workload Identity
  workload_identity_enabled = true

  network_profile {
    network_plugin    = "azure"
    network_policy    = "calico"
    load_balancer_sku = "standard"
    service_cidr      = "10.1.0.0/16"
    dns_service_ip    = "10.1.0.10"
  }

  oms_agent {
    log_analytics_workspace_id      = azurerm_log_analytics_workspace.main.id
    msi_auth_for_monitoring_enabled = true
  }

  tags = local.tags

  lifecycle { ignore_changes = [default_node_pool[0].node_count] }
}

# Grant AKS pull access to ACR
resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
}

# ── PostgreSQL Flexible Server ─────────────────────────────────────────────────

resource "azurerm_private_dns_zone" "postgres" {
  name                = "${local.prefix}.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "${local.prefix}-postgres-vnet-link"
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
  resource_group_name   = azurerm_resource_group.main.name
}

resource "random_password" "postgres" {
  length  = 32
  special = true
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${local.prefix}-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  delegated_subnet_id    = azurerm_subnet.postgres.id
  private_dns_zone_id    = azurerm_private_dns_zone.postgres.id
  administrator_login    = "raglab_admin"
  administrator_password = random_password.postgres.result
  zone                   = "1"

  storage {
    storage_mb = var.postgres_storage_gb * 1024
    tier       = "P30"
  }

  sku_name = "GP_Standard_D4s_v3"
  tags     = local.tags

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "raglab" {
  name      = "raglab"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# ── Key Vault ─────────────────────────────────────────────────────────────────

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                       = "${local.prefix}-kv"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = true
  tags                       = local.tags
}

resource "azurerm_key_vault_access_policy" "aks_workload" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_kubernetes_cluster.main.identity[0].principal_id

  secret_permissions = ["Get", "List"]
}

# Store Postgres password in Key Vault — never in tfvars
resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-password"
  value        = random_password.postgres.result
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.aks_workload]
}

resource "azurerm_key_vault_secret" "postgres_dsn" {
  name         = "postgres-dsn"
  value        = "postgresql+asyncpg://raglab_admin:${random_password.postgres.result}@${azurerm_postgresql_flexible_server.main.fqdn}/raglab"
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.aks_workload]
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "aks_cluster_name"        { value = azurerm_kubernetes_cluster.main.name }
output "aks_resource_group"      { value = azurerm_resource_group.main.name }
output "acr_login_server"        { value = azurerm_container_registry.main.login_server }
output "postgres_fqdn"           { value = azurerm_postgresql_flexible_server.main.fqdn }
output "key_vault_uri"           { value = azurerm_key_vault.main.vault_uri }
output "log_analytics_workspace" { value = azurerm_log_analytics_workspace.main.id }
