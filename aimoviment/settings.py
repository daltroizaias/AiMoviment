from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # === Arquivo .env ===
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
    )

    # === Variáveis do .env ===
    FLASK_PORT: int = 8000
    FLASK_DEBUG: int
    SECRET_KEY: str


app_config = Config()
