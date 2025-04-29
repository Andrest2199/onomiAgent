from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from assistant import onomi_assistant, transcribe
from utils.messages import retrieve_messages_thread
from models import ONOMIRequest

app = FastAPI(
    title="ONOMI Assistant API",
    description="API para interactuar con el asistente de RRHH.",
    version="1.0.0"
)

@app.post("/onomi")
async def onomi(request: ONOMIRequest):
    """
    Procesa una pregunta textual al asistente ONOMI.
    """
    try:
        # Llama la función principal con los datos del request
        data = onomi_assistant(
            request.id_employee,
            request.compania,
            request.question,
            request.database,
            request.thread_id,
            request.is_admin
        )
        return data

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), id_employee: str = "", compania: str = ""):
    """
    Transcribe un audio enviado por el usuario.
    """
    try:
        if not audio:
            raise HTTPException(status_code=400, detail="Archivo de audio no recibido.")
        if not id_employee or not compania:
            raise HTTPException(status_code=400, detail="ID de empleado o compañía no proporcionados.")

        result = transcribe(id_employee, compania, audio)
        if result:
            return {"transcription": result.text}
        else:
            raise HTTPException(status_code=500, detail="Falló la transcripción de audio.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve_messages")
async def get_messages(thread_id: str):
    try:
        if not thread_id.strip():
            raise HTTPException(status_code=400, detail="No Se Proporcionó Ningun ID de Hilo de Conversación")
        
        if not isinstance(thread_id, str):
            raise HTTPException(status_code=400, detail="El ID de Thread debe ser una cadena de texto")
        
        data = retrieve_messages_thread(thread_id)
        
        if "error" in data:
            raise HTTPException(status_code=404, detail=data.get("error"))

        return JSONResponse(content=data, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))