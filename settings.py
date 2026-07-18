from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PROJECT: str
    ORG_ID: str
    LOG_FILE: str # -> Nombre del archivo que creara de logs
    OPENAI_MODEL: str = "gpt-5-nano" # -> Nombre del modelo por default
    OPENAI_MAX_OUTPUT_TOKENS: int = 3000 # -> Limita el numero de tokens en la respuesta del modelo
    OPENAI_VECTOR_STORE_ID: str | None = None # -> El id del vector de documentos del cliente guardado en Open AI
    OPENAI_COMPACT_THRESHOLD: int | None = 12000 # -> Limita que las respuestas no sean tan grandes
    OPENAI_MAX_MESSAGES_PER_CONVERSATION: int = 5 # -> Limita el numero de mensajes por conversacion para evitar consumos grandes
    OPENAI_MAX_INPUT_TOKENS_PER_CONVERSATION: int | None = 20000 # -> Limita el numero de tokens de una pregunta considerando la suma de mensaje de una conversación

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
