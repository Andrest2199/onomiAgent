from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PROJECT: str
    ORG_ID: str
    LOG_FILE: str
    OPENAI_MODEL: str = "gpt-5-nano"
    OPENAI_MAX_OUTPUT_TOKENS: int = 1200
    OPENAI_VECTOR_STORE_ID: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
