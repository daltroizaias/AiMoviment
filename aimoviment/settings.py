import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # === Arquivo .env ===
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
    )

    # === Variáveis do .env ===
    FLASK_PORT: int
    FLASK_DEBUG: int
    SECRET_KEY: str
    MODELS_TRAINED: str
    VIDEO_URL: str


configuracao = Config()
