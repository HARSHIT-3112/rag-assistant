from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    secret_key: str = "dev-secret-change-in-prod"

    gemini_api_key: str

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_assistant"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    redis_url: str = "redis://localhost:6379/0"
    faiss_index_dir: str = "./data/faiss"

    chunk_size: int = 800
    chunk_overlap: int = 150

    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
