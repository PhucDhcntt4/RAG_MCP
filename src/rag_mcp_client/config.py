
from functools import lru_cache

from pydantic import Field, field_validator # type: ignore
from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    search_api_key: str = Field(min_length=1)

    rag_search_url: str = (
        "http://127.0.0.1:8000/api/v1/knowledge/search"
    )

    rag_products_url: str = (
        "http://127.0.0.1:8000/api/v1/products"
    )

    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8001, ge=1, le=65535)
    mcp_path: str = "/mcp"

    @field_validator("mcp_path")
    @classmethod
    def normalize_mcp_path(cls, value: str) -> str:
        value = value.strip()

        if not value:
            return "/mcp"

        if not value.startswith("/"):
            return f"/{value}"

        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()