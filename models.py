from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


# 1. Definimos la estructura interna de agent_config
class AgentConfig(BaseModel):
    docs: List[str]
    apis: List[str]
    agent: List[str]


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
            if not value.docs and not value.apis and not value.agent:
                raise ValueError(f"El campo '{field_name}' no puede estar vacío")
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name.capitalize().replace('_', ' ')} debe ser enviado como cadena de texto"
            )
        if not value.strip():
            raise ValueError(f"El campo '{field_name}' no puede estar vacío")
        return value
