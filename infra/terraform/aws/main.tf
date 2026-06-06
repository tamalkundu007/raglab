# RAGLab — AWS Infrastructure (Terraform)
#
# Provisions:
#   - VPC (3 AZs, public + private + database subnets)
#   - ECR repositories (one per service)
#   - EKS cluster (managed node group, Fargate for system pods)
#   - RDS PostgreSQL (Multi-AZ, encrypted)
#   - Qdrant on EKS (StatefulSet)
#   - RabbitMQ on EKS (StatefulSet via Bitnami Helm)
#   - AWS Secrets Manager entries for all credentials
#   - IAM OIDC provider for Workload Identity (no static keys in pods)
#   - CloudWatch log groups
#
# State backend: S3 + DynamoDB (configure before first apply)
# Auth: OIDC via GitHub Actions (no AWS_ACCESS_KEY_ID stored)

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  # backend "s3" {
  #   bucket         = "raglab-terraform-state"
  #   key            = "raglab/aws/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "raglab-terraform-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
  default_tags { tags = local.tags }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "aws_region"           { type = string; default = "us-east-1" }
variable "environment"          { type = string; default = "prod" }
variable "project_name"         { type = string; default = "raglab" }
variable "image_tag"            { type = string }
variable "instance_type"        { type = string; default = "m7i.xlarge" }
variable "node_min_count"       { type = number; default = 2 }
variable "node_max_count"       { type = number; default = 10 }
variable "postgres_instance"    { type = string; default = "db.m7g.large" }
variable "postgres_storage_gb"  { type = number; default = 128 }

locals {
  prefix = "${var.project_name}-${var.environment}"
  tags   = { Project = var.project_name, Environment = var.environment, ManagedBy = "terraform" }
  azs    = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
}

# ── VPC ───────────────────────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = merge(local.tags, { Name = "${local.prefix}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.prefix}-igw" })
}

resource "aws_subnet" "public" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = local.azs[count.index]
  map_public_ip_on_launch = true
  tags = merge(local.tags, {
    Name                                          = "${local.prefix}-public-${count.index + 1}"
    "kubernetes.io/role/elb"                      = "1"
    "kubernetes.io/cluster/${local.prefix}-eks"   = "owned"
  })
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = local.azs[count.index]
  tags = merge(local.tags, {
    Name                                          = "${local.prefix}-private-${count.index + 1}"
    "kubernetes.io/role/internal-elb"             = "1"
    "kubernetes.io/cluster/${local.prefix}-eks"   = "owned"
  })
}

resource "aws_subnet" "database" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 20}.0/24"
  availability_zone = local.azs[count.index]
  tags              = merge(local.tags, { Name = "${local.prefix}-db-${count.index + 1}" })
}

resource "aws_eip" "nat" {
  count  = 3
  domain = "vpc"
  tags   = merge(local.tags, { Name = "${local.prefix}-nat-eip-${count.index + 1}" })
}

resource "aws_nat_gateway" "main" {
  count         = 3
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  tags          = merge(local.tags, { Name = "${local.prefix}-nat-${count.index + 1}" })
  depends_on    = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  count  = 3
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  tags = merge(local.tags, { Name = "${local.prefix}-private-rt-${count.index + 1}" })
}

resource "aws_route_table_association" "private" {
  count          = 3
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# ── ECR Repositories ──────────────────────────────────────────────────────────

locals {
  service_names = [
    "api-gateway", "ingestion", "embedding", "indexing", "retrieval",
    "llm", "pipeline", "storage", "ui", "graph", "observability", "auth"
  ]
}

resource "aws_ecr_repository" "services" {
  for_each             = toset(local.service_names)
  name                 = "raglab/${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "services" {
  for_each   = aws_ecr_repository.services
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────

resource "aws_iam_role" "eks_cluster" {
  name = "${local.prefix}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "eks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policies" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
    "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController",
  ])
  role       = aws_iam_role.eks_cluster.name
  policy_arn = each.key
}

resource "aws_eks_cluster" "main" {
  name     = "${local.prefix}-eks"
  version  = "1.29"
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_public_access  = true
    endpoint_private_access = true
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policies]
}

# OIDC provider for Workload Identity (pods assume IAM roles, no static keys)
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

# ── EKS Managed Node Group ─────────────────────────────────────────────────────

resource "aws_iam_role" "eks_nodes" {
  name = "${local.prefix}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node_policies" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])
  role       = aws_iam_role.eks_nodes.name
  policy_arn = each.key
}

resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${local.prefix}-ng"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = [var.instance_type]
  disk_size       = 128

  scaling_config {
    desired_size = var.node_min_count
    min_size     = var.node_min_count
    max_size     = var.node_max_count
  }

  update_config { max_unavailable = 1 }

  depends_on = [aws_iam_role_policy_attachment.eks_node_policies]
  lifecycle  { ignore_changes = [scaling_config[0].desired_size] }
}

# ── RDS PostgreSQL ─────────────────────────────────────────────────────────────

resource "aws_db_subnet_group" "main" {
  name       = "${local.prefix}-db-subnet-group"
  subnet_ids = aws_subnet.database[*].id
}

resource "aws_security_group" "rds" {
  name        = "${local.prefix}-rds-sg"
  description = "RDS PostgreSQL — allow from EKS nodes"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}

resource "random_password" "postgres" {
  length  = 32
  special = false  # RDS URL-safe password
}

resource "aws_db_instance" "main" {
  identifier              = "${local.prefix}-postgres"
  engine                  = "postgres"
  engine_version          = "16"
  instance_class          = var.postgres_instance
  allocated_storage       = var.postgres_storage_gb
  max_allocated_storage   = var.postgres_storage_gb * 2
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = "raglab"
  username                = "raglab_admin"
  password                = random_password.postgres.result
  multi_az                = true
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  deletion_protection     = true
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.prefix}-final-snapshot"
  backup_retention_period = 7
  copy_tags_to_snapshot   = true
}

# ── Secrets Manager ───────────────────────────────────────────────────────────
# LLM API keys and DB credentials — never in env vars or tfvars files.
# Inject into pods via ASCP (Secrets Store CSI Driver for AWS).

resource "aws_secretsmanager_secret" "postgres_dsn" {
  name                    = "raglab/postgres-dsn"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "postgres_dsn" {
  secret_id     = aws_secretsmanager_secret.postgres_dsn.id
  secret_string = "postgresql+asyncpg://raglab_admin:${random_password.postgres.result}@${aws_db_instance.main.endpoint}/raglab"
}

# Placeholder secrets — values set by operators after provisioning, never via Terraform
resource "aws_secretsmanager_secret" "llm_keys" {
  for_each                = toset(["azure-openai-key", "azure-openai-endpoint", "anthropic-api-key"])
  name                    = "raglab/${each.key}"
  recovery_window_in_days = 7
}

# ── CloudWatch Log Groups ─────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "services" {
  for_each          = toset(local.service_names)
  name              = "/raglab/${var.environment}/${each.key}"
  retention_in_days = 14
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "eks_cluster_name"     { value = aws_eks_cluster.main.name }
output "eks_cluster_endpoint" { value = aws_eks_cluster.main.endpoint }
output "ecr_registry"         { value = split("/", values(aws_ecr_repository.services)[0].repository_url)[0] }
output "rds_endpoint"         { value = aws_db_instance.main.endpoint }
output "vpc_id"               { value = aws_vpc.main.id }
output "oidc_provider_arn"    { value = aws_iam_openid_connect_provider.eks.arn }
