// RAGLab — Azure Container Apps Infrastructure
// Deploy: az deployment group create --resource-group <rg> --template-file main.bicep
//
// Creates:
//   - Azure Container Apps (one per active service)
//   - Shared Container Apps Environment (with Qdrant + Postgres + RabbitMQ)
//   - Key Vault references for LLM API keys
//   - Log Analytics workspace for structured logging

@description('Azure Container Registry login server')
param acrLoginServer string

@description('Docker image tag (git SHA)')
param imageTag string

@description('Azure Container Apps environment name')
param acaEnvironment string

@description('Azure region')
param location string = resourceGroup().location

// ── Services configuration ───────────────────────────────────────────────────

var services = [
  { name: 'api-gateway',   port: 8000, external: true,  cpu: '0.5', memory: '1Gi' }
  { name: 'ingestion',     port: 8001, external: false, cpu: '0.5', memory: '1Gi' }
  { name: 'embedding',     port: 8002, external: false, cpu: '1.0', memory: '2Gi' }
  { name: 'indexing',      port: 8003, external: false, cpu: '0.5', memory: '1Gi' }
  { name: 'retrieval',     port: 8004, external: false, cpu: '0.5', memory: '1Gi' }
  { name: 'llm',           port: 8005, external: false, cpu: '1.0', memory: '2Gi' }
  { name: 'pipeline',      port: 8006, external: false, cpu: '0.5', memory: '1Gi' }
  { name: 'storage',       port: 8008, external: false, cpu: '0.25', memory: '0.5Gi' }
  { name: 'ui',            port: 8009, external: true,  cpu: '0.25', memory: '0.5Gi' }
]

// ── Container Apps (one per service) ────────────────────────────────────────

resource containerApps 'Microsoft.App/containerApps@2023-05-01' = [for svc in services: {
  name: 'raglab-${svc.name}'
  location: location
  properties: {
    environmentId: resourceId('Microsoft.App/managedEnvironments', acaEnvironment)
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [
        {
          server: acrLoginServer
          identity: 'system'
        }
      ]
      ingress: {
        external: svc.external
        targetPort: svc.port
        transport: 'http'
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      secrets: [
        { name: 'azure-openai-key',      keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/azure-openai-key' }
        { name: 'azure-openai-endpoint', keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/azure-openai-endpoint' }
        { name: 'azure-chat-deployment', keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/azure-chat-deployment' }
        { name: 'azure-embed-deployment',keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/azure-embed-deployment' }
        { name: 'postgres-dsn',          keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/postgres-dsn' }
        { name: 'rabbitmq-url',          keyVaultUrl: 'https://raglab-kv.vault.azure.net/secrets/rabbitmq-url' }
      ]
    }
    template: {
      containers: [
        {
          name: svc.name
          image: '${acrLoginServer}/raglab/${svc.name}:${imageTag}'
          resources: {
            cpu: json(svc.cpu)
            memory: svc.memory
          }
          env: [
            { name: 'RAGLAB_SERVICE_NAME',                        value: svc.name }
            { name: 'RAGLAB_PORT',                                value: string(svc.port) }
            { name: 'RAGLAB_JSON_LOGS',                           value: 'true' }
            { name: 'RAGLAB_AZURE_OPENAI_API_KEY',                secretRef: 'azure-openai-key' }
            { name: 'RAGLAB_AZURE_OPENAI_ENDPOINT',               secretRef: 'azure-openai-endpoint' }
            { name: 'RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT',        secretRef: 'azure-chat-deployment' }
            { name: 'RAGLAB_AZURE_OPENAI_EMBEDDING_DEPLOYMENT',   secretRef: 'azure-embed-deployment' }
            { name: 'RAGLAB_POSTGRES_DSN',                        secretRef: 'postgres-dsn' }
            { name: 'RAGLAB_RABBITMQ_URL',                        secretRef: 'rabbitmq-url' }
            // Service discovery — internal FQDNs within ACA environment
            { name: 'RAGLAB_INGESTION_URL',   value: 'http://raglab-ingestion' }
            { name: 'RAGLAB_EMBEDDING_URL',   value: 'http://raglab-embedding' }
            { name: 'RAGLAB_INDEXING_URL',    value: 'http://raglab-indexing' }
            { name: 'RAGLAB_RETRIEVAL_URL',   value: 'http://raglab-retrieval' }
            { name: 'RAGLAB_LLM_URL',         value: 'http://raglab-llm' }
            { name: 'RAGLAB_PIPELINE_URL',    value: 'http://raglab-pipeline' }
            { name: 'RAGLAB_STORAGE_URL',     value: 'http://raglab-storage' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: svc.port
              }
              initialDelaySeconds: 20
              periodSeconds: 15
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: svc.port
              }
              initialDelaySeconds: 10
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}]

// ── Outputs ──────────────────────────────────────────────────────────────────

output gatewayFQDN string = containerApps[0].properties.configuration.ingress.fqdn
output uiFQDN string = containerApps[8].properties.configuration.ingress.fqdn
