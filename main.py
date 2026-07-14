from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from models import ONOMIRequest
from settings import settings

app = FastAPI(
    title="ONOMI Agent API",
    description="API para interactuar con el agente de RRHH.",
    version="2.0.0",
)


@app.get("/")
async def read_root():
    return {
        "app_name": "ONOMI Agent API",
        "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
        "openai_model": settings.OPENAI_MODEL,
    }


@app.post("/onomi")
async def onomi(request: ONOMIRequest):
    """
    Procesa una pregunta textual al asistente ONOMI.
    """
    try:
        from assistant import onomi_assistant

        # Llama la función principal con los datos del request
        data = onomi_assistant(
            request.compania_id,
            request.compania_name,
            request.id_employee,
            request.permission_type,
            request.question,
            request.agent_config,
            request.previous_conversation_id
        )
        return data

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...), id_employee: str = "", compania: str = ""
):
    """
    Transcribe un audio enviado por el usuario.
    """
    try:
        from assistant import transcribe

        if not audio:
            raise HTTPException(status_code=400, detail="Archivo de audio no recibido.")
        if not id_employee or not compania:
            raise HTTPException(
                status_code=400, detail="ID de empleado o compañía no proporcionados."
            )

        result = transcribe(id_employee, compania, audio)
        if result:
            return {"transcription": result.text}
        else:
            raise HTTPException(
                status_code=500, detail="Falló la transcripción de audio."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/retrieve_messages")
async def get_messages(conversation_id: str = ""):
    try:
        from utils.messages import retrieve_messages_thread

        conversation_id = conversation_id

        if not conversation_id.strip():
            raise HTTPException(
                status_code=400,
                detail="No se proporcionó ningún ID de conversación.",
            )

        if not isinstance(conversation_id, str):
            raise HTTPException(
                status_code=400,
                detail="El ID de conversación debe ser una cadena de texto.",
            )

        data = retrieve_messages_thread(conversation_id)

        if "error" in data:
            raise HTTPException(status_code=404, detail=data.get("error"))

        return JSONResponse(content=data, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
