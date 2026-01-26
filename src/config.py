"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DataForSEO API
    dataforseo_login: str = Field(default="", description="DataForSEO API login")
    dataforseo_password: SecretStr = Field(default="", description="DataForSEO API password")

    # Apify API
    apify_api_token: SecretStr = Field(default="", description="Apify API token")

    # Jungle Scout API
    junglescout_api_key: SecretStr = Field(default="", description="Jungle Scout API key")
    junglescout_api_key_name: str = Field(default="", description="Jungle Scout API key name")

    # Meta Marketing API (optional — for Audience Reach)
    meta_app_id: str = Field(default="", description="Meta App ID")
    meta_app_secret: SecretStr = Field(default="", description="Meta App Secret")
    meta_access_token: SecretStr = Field(default="", description="Meta Marketing API access token")
    meta_ad_account_id: str = Field(default="", description="Meta Ad Account ID (without act_ prefix)")

    # Database
    database_url: str = Field(
        default="sqlite:///data/keywords.db",
        description="Database connection URL",
    )

    # Redis (optional)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_enabled: bool = Field(default=False, description="Enable Redis caching")

    # Pipeline Settings
    default_location_code: int = Field(default=2840, description="Default location code (2840=US)")
    default_language_code: str = Field(default="en", description="Default language code")
    default_marketplace: str = Field(default="us", description="Default Amazon marketplace")
    batch_size: int = Field(default=100, description="Batch size for API requests")
    max_retries: int = Field(default=3, description="Maximum retry attempts for API calls")
    rate_limit_delay: float = Field(default=1.0, description="Delay between API calls in seconds")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="text", description="Log output format")

    @property
    def dataforseo_credentials(self) -> tuple[str, str]:
        """Get DataForSEO credentials as tuple."""
        return (self.dataforseo_login, self.dataforseo_password.get_secret_value())

    @property
    def meta_configured(self) -> bool:
        """Check if Meta Marketing API credentials are configured."""
        return bool(
            self.meta_app_id
            and self.meta_app_secret.get_secret_value()
            and self.meta_access_token.get_secret_value()
            and self.meta_ad_account_id
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
