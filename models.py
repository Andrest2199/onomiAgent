from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ONOMIRequest(BaseModel):
    question: str
    id_employee: str
    compania: str
    database: str
    thread_id: Optional[str] = ""

    @field_validator("question", "id_employee", "compania", "database", "thread_id")
    @classmethod
    def check_not_empty(cls, value, info):
        if info.field_name != "thread_id" and not value.strip():
            raise ValueError(f"El campo '{info.field_name}' no puede estar vacío")
        if not isinstance(value, str):
            raise TypeError(f"{info.field_name.capitalize().replace('_', ' ')} debe ser enviado como cadena de texto")
        return value