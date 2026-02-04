"""Configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration loaded from environment."""

    # Internet Archive credentials (optional - for authenticated requests)
    ia_access_key: str | None = None
    ia_secret_key: str | None = None

    # S3 configuration (optional for CLI commands that don't use S3)
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_endpoint: str | None = None
    s3_presign_expiry: int = 604800  # 7 days in seconds
    s3_raw_prefix: str = "raw/ia"  # Path prefix for files
    s3_meta_prefix: str = "meta/ia"  # Path prefix for metadata files

    # Download configuration
    download_timeout: float = 120.0  # IA files can be large
    connect_timeout: float = 10.0
    max_retries: int = 3

    # Concurrency settings
    max_concurrent_downloads: int = 4  # Be respectful to IA

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Public base URL for generating external links (e.g., https://dl.example.com)
    base_url: str | None = None

    model_config = {"env_prefix": "IA_DL_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Get settings instance."""
    return Settings()
