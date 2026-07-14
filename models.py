from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List


# Definimos la estructura interna de agent_config
class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str = "ONOMI"
    language: str = "español"
    voice:  Optional[str] = "nova"
    tone: Optional[str] = ""
    docs: List[str] = Field(default_factory=list)
    apis: List[str] = Field(default_factory=list)

    @field_validator("agent_name", "language")
    @classmethod
    def check_config_text(cls, value, info):
        if not isinstance(value, str):
            raise TypeError(
                f"{info.field_name.capitalize().replace('_', ' ')} debe ser enviado como cadena de texto"
            )
        if not value.strip():
            raise ValueError(f"El campo '{info.field_name}' no puede estar vacío")
        return value

# Definimos la estructura del request al endpoint onomi
class ONOMIRequest(BaseModel):
    compania_id: str
    compania_name: str
    id_employee: str
    permission_type: str
    question: str
    agent_config: AgentConfig
    previous_conversation_id: Optional[str] = ""

    @field_validator(
        "compania_id",
        "compania_name",
        "id_employee",
        "permission_type",
        "question",
        "agent_config",
        "previous_conversation_id",
    )
    @classmethod
    def check_not_empty(cls, value, info):
        field_name = info.field_name

        if field_name == "previous_conversation_id":
            if value is None:
                return value
            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name.capitalize().replace('_', ' ')} debe ser enviado como cadena de texto"
                )
            return value

        if field_name == "agent_config":
            if not isinstance(value, AgentConfig):
                raise TypeError(
                    f"{field_name.capitalize().replace('_', ' ')} debe ser enviado como AgentConfig"
                )
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name.capitalize().replace('_', ' ')} debe ser enviado como cadena de texto"
            )
        if not value.strip():
            raise ValueError(f"El campo '{field_name}' no puede estar vacío")
        return value
