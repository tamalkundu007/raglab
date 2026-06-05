"""Settings for the llm-service."""

from raglab_common.settings import BaseServiceSettings


class LLMSettings(BaseServiceSettings):
    service_name: str = "llm"
    port: int = 8005

    # Default generation params
    default_max_tokens: int = 1024
    default_temperature: float = 0.0      # deterministic by default for RAG
    default_top_p: float = 1.0

    # Azure OpenAI chat deployment (may differ from embedding deployment)
    azure_openai_chat_deployment: str = ""   # e.g. "gpt-4o"

    # Anthropic
    anthropic_model: str = "claude-sonnet-4-20250514"

    # Ollama
    ollama_chat_model: str = "llama3.2"

    # OpenAI
    openai_chat_model: str = "gpt-4o-mini"

    # RAG prompt template (can be overridden per request)
    rag_system_prompt: str = (
        "You are a precise assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information "
        "to answer, say so clearly. Do not fabricate facts."
    )
