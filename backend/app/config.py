"""
Application configuration — reads from environment variables / a .env file.

Why pydantic-settings instead of scattered os.getenv() calls: this crashes
loudly and immediately at startup if DATABASE_URL is missing, telling you
exactly what's wrong — instead of failing confusingly deep inside some
unrelated function the first time it's used.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
