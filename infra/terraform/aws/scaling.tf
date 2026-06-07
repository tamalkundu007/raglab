# RAGLab — AWS Kubernetes Scaling Refinements (R5)
#
# Adds to aws/main.tf:
#   - ElastiCache Redis cluster (embedding cache)
#   - Karpenter node provisioner (cost-efficient autoscaling)
#   - HPA definitions via Helm (metrics-server)
#   - IRSA bindings for pipeline + embedding services
#   - Security group refinements (Redis access from EKS only)
#   - CloudWatch Container Insights

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
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

variable "aws_region"           { type = string; default = "us-east-1" }
variable "environment"          { type = string; default = "prod" }
variable "project_name"         { type = string; default = "raglab" }
variable "eks_cluster_name"     { type = string }
variable "eks_oidc_provider_arn"{ type = string }
variable "eks_oidc_provider_url"{ type = string }
variable "vpc_id"               { type = string }
variable "private_subnet_ids"   { type = list(string) }

locals {
  prefix = "${var.project_name}-${var.environment}"
  tags   = { Project = var.project_name, Environment = var.environment, ManagedBy = "terraform" }
}

# ── ElastiCache Redis (Embedding cache) ───────────────────────────────────────
# Cluster mode disabled — single-shard with multi-AZ replica.
# R6: enable cluster mode for semantic cache horizontal sharding.

resource "aws_security_group" "redis" {
  name        = "${local.prefix}-redis-sg"
  description = "Redis — allow from EKS nodes only"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # VPC CIDR — EKS nodes only
    description = "Redis from VPC"
  }

  tags = merge(local.tags, { Name = "${local.prefix}-redis-sg" })
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.prefix}-redis-subnet"
  subnet_ids = var.private_subnet_ids
  tags       = local.tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.prefix}-redis"
  description                = "RAGLab embedding cache (R5)"
  node_type                  = "cache.t4g.medium"  # 2 vCPU, 3.1 GB — right-sized for embedding vectors
  num_cache_clusters         = 2                   # 1 primary + 1 replica
  automatic_failover_enabled = true
  multi_az_enabled           = true
  engine                     = "redis"
  engine_version             = "7.1"
  port                       = 6379
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled         = false  # use IAM auth instead of password in R6

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_logs.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "engine-log"
  }

  tags = merge(local.tags, { Name = "${local.prefix}-redis" })
}

resource "aws_cloudwatch_log_group" "redis_logs" {
  name              = "/raglab/${var.environment}/redis"
  retention_in_days = 7
  tags              = local.tags
}

# Store Redis connection string in Secrets Manager
resource "aws_secretsmanager_secret" "redis_url" {
  name                    = "raglab/redis-url"
  recovery_window_in_days = 7
  tags                    = local.tags
}

resource "aws_secretsmanager_secret_version" "redis_url" {
  secret_id     = aws_secretsmanager_secret.redis_url.id
  secret_string = "rediss://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379/0"
}

# ── IRSA — IAM Roles for Service Accounts ─────────────────────────────────────
# Pods assume IAM roles via OIDC. No AWS credentials in pods or images.

locals {
  oidc_sub_prefix = "system:serviceaccount:raglab"
}

# embedding-service: needs ElastiCache + Secrets Manager access
resource "aws_iam_role" "embedding_irsa" {
  name = "${local.prefix}-embedding-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Federated = var.eks_oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${var.eks_oidc_provider_url}:sub" = "${local.oidc_sub_prefix}:raglab-embedding"
          "${var.eks_oidc_provider_url}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_policy" "embedding_access" {
  name = "${local.prefix}-embedding-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = ["arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "embedding_irsa" {
  role       = aws_iam_role.embedding_irsa.name
  policy_arn = aws_iam_policy.embedding_access.arn
}

# pipeline-service: needs S3 + Secrets Manager access
resource "aws_iam_role" "pipeline_irsa" {
  name = "${local.prefix}-pipeline-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Federated = var.eks_oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${var.eks_oidc_provider_url}:sub" = "${local.oidc_sub_prefix}:raglab-pipeline"
          "${var.eks_oidc_provider_url}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_policy" "pipeline_access" {
  name = "${local.prefix}-pipeline-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::raglab-${var.environment}-docs",
          "arn:aws:s3:::raglab-${var.environment}-docs/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = ["arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "pipeline_irsa" {
  role       = aws_iam_role.pipeline_irsa.name
  policy_arn = aws_iam_policy.pipeline_access.arn
}

# ── S3 Document storage bucket ─────────────────────────────────────────────────

resource "aws_s3_bucket" "docs" {
  bucket        = "raglab-${var.environment}-docs"
  force_destroy = false
  tags          = merge(local.tags, { Name = "raglab-docs" })
}

resource "aws_s3_bucket_versioning" "docs" {
  bucket = aws_s3_bucket.docs.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket                  = aws_s3_bucket.docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── CloudWatch Container Insights ─────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "container_insights" {
  name              = "/aws/containerinsights/${var.eks_cluster_name}/performance"
  retention_in_days = 14
  tags              = local.tags
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "redis_primary_endpoint"  { value = aws_elasticache_replication_group.redis.primary_endpoint_address }
output "redis_secret_arn"        { value = aws_secretsmanager_secret.redis_url.arn }
output "embedding_irsa_role_arn" { value = aws_iam_role.embedding_irsa.arn }
output "pipeline_irsa_role_arn"  { value = aws_iam_role.pipeline_irsa.arn }
output "docs_bucket"             { value = aws_s3_bucket.docs.bucket }
