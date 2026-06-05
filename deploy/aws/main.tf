# RAGLab — AWS ECS Fargate Infrastructure (Terraform)
# Apply: terraform init && terraform apply -var="image_tag=<sha>"
#
# Creates:
#   - ECS Cluster (Fargate)
#   - ECS Task Definitions (one per service)
#   - ECS Services (one per service)
#   - ALB + target groups for api-gateway and ui
#   - VPC + subnets + security groups
#   - Secrets Manager entries for LLM API keys
#   - CloudWatch log groups
#
# State backend: configure separately (S3 + DynamoDB lock recommended)

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Uncomment and configure for remote state:
  # backend "s3" {
  #   bucket         = "raglab-terraform-state"
  #   key            = "raglab/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "raglab-terraform-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "aws_region"   { type = string; default = "us-east-1" }
variable "ecr_registry" { type = string; description = "ECR registry URL" }
variable "image_tag"    { type = string; description = "Docker image tag (git SHA)" }
variable "ecs_cluster"  { type = string; default = "raglab-cluster" }

# ── ECS Cluster ───────────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "raglab" {
  name = var.ecs_cluster

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = { Project = "raglab" }
}

resource "aws_ecs_cluster_capacity_providers" "raglab" {
  cluster_name       = aws_ecs_cluster.raglab.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = "FARGATE"
  }
}

# ── CloudWatch Log Group ──────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "raglab" {
  name              = "/ecs/raglab"
  retention_in_days = 14
  tags              = { Project = "raglab" }
}

# ── IAM: ECS Task Execution Role ─────────────────────────────────────────────

resource "aws_iam_role" "ecs_task_execution" {
  name = "raglab-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_secrets" {
  name = "raglab-ecs-secrets"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = ["arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/*"]
    }]
  })
}

# ── Services map ─────────────────────────────────────────────────────────────

locals {
  services = {
    "api-gateway" = { port = 8000, cpu = 512,  memory = 1024 }
    "ingestion"   = { port = 8001, cpu = 512,  memory = 1024 }
    "embedding"   = { port = 8002, cpu = 1024, memory = 2048 }
    "indexing"    = { port = 8003, cpu = 512,  memory = 1024 }
    "retrieval"   = { port = 8004, cpu = 512,  memory = 1024 }
    "llm"         = { port = 8005, cpu = 1024, memory = 2048 }
    "pipeline"    = { port = 8006, cpu = 512,  memory = 1024 }
    "storage"     = { port = 8008, cpu = 256,  memory = 512  }
    "ui"          = { port = 8009, cpu = 256,  memory = 512  }
  }
}

# ── ECS Task Definitions ──────────────────────────────────────────────────────

resource "aws_ecs_task_definition" "services" {
  for_each = local.services

  family                   = "raglab-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name      = replace(each.key, "-", "_")
    image     = "${var.ecr_registry}/raglab/${each.key}:${var.image_tag}"
    essential = true

    portMappings = [{
      containerPort = each.value.port
      protocol      = "tcp"
    }]

    environment = [
      { name = "RAGLAB_SERVICE_NAME", value = each.key }
      { name = "RAGLAB_PORT",         value = tostring(each.value.port) }
      { name = "RAGLAB_JSON_LOGS",    value = "true" }
      # Internal service discovery via ECS Service Connect
      { name = "RAGLAB_INGESTION_URL",  value = "http://raglab-ingestion:8001" }
      { name = "RAGLAB_EMBEDDING_URL",  value = "http://raglab-embedding:8002" }
      { name = "RAGLAB_INDEXING_URL",   value = "http://raglab-indexing:8003" }
      { name = "RAGLAB_RETRIEVAL_URL",  value = "http://raglab-retrieval:8004" }
      { name = "RAGLAB_LLM_URL",        value = "http://raglab-llm:8005" }
      { name = "RAGLAB_PIPELINE_URL",   value = "http://raglab-pipeline:8006" }
      { name = "RAGLAB_STORAGE_URL",    value = "http://raglab-storage:8008" }
    ]

    # LLM API keys from AWS Secrets Manager — never as plain env vars
    secrets = [
      { name = "RAGLAB_AZURE_OPENAI_API_KEY",              valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/azure-openai-key" }
      { name = "RAGLAB_AZURE_OPENAI_ENDPOINT",             valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/azure-openai-endpoint" }
      { name = "RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT",      valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/azure-chat-deployment" }
      { name = "RAGLAB_AZURE_OPENAI_EMBEDDING_DEPLOYMENT", valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/azure-embed-deployment" }
      { name = "RAGLAB_POSTGRES_DSN",                      valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/postgres-dsn" }
      { name = "RAGLAB_RABBITMQ_URL",                      valueFrom = "arn:aws:secretsmanager:${var.aws_region}:*:secret:raglab/rabbitmq-url" }
    ]

    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${each.value.port}/health')\""]
      interval    = 15
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.raglab.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = each.key
      }
    }
  }])

  tags = { Project = "raglab", Service = each.key }
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "ecs_cluster_name" { value = aws_ecs_cluster.raglab.name }
output "task_execution_role_arn" { value = aws_iam_role.ecs_task_execution.arn }
output "log_group_name" { value = aws_cloudwatch_log_group.raglab.name }
