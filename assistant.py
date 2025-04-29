#%% assistant.py
import os
import io
import time
import logging
from openai import OpenAI
from utils.functions import handle_required_action
from utils.messages import retrieve_annotation
from dotenv import load_dotenv

# Carga .env
load_dotenv()
# Configuramos las variables de entorno
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")
assistant_id = os.environ.get("OPENAI_ASSISTANT")
log_file = os.environ.get("LOG_FILE")

# Set log
logging.basicConfig(filename=log_file,format="%(levelname)s|%(asctime)s|%(message)s",level=logging.INFO)
# Create instance of openAI client
client = OpenAI(organization=org_id,api_key=api_key)

#Main function
def onomi_assistant(id_employee: str, company: str, question: str, database: str, thread_id: str, is_admin: bool):
    # Declare variables
    response = {}
    tokens_use = 0
    # Define thread
    if not thread_id:
        # Create a thread
        thread = client.beta.threads.create(metadata={"user": id_employee,"company": company})
    else:
        # Retrieve a thread
        thread = client.beta.threads.update(thread_id, metadata={"modified": "true", "user": id_employee, "company": company})
    logging.info(f"%s|%s| THREAD INFO: {thread}",id_employee,company)
    # Create a message and add it to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        content=question,
        role="user",
        metadata={"user": id_employee,"company": company}
    )
    logging.info(f"%s|%s| MESSAGE INFO: {message}",id_employee,company)
    # Run the thread
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        metadata={"user": id_employee,"company": company}
    )
    logging.info(f"%s|%s| RUN ID: {run.id}",id_employee,company)
    # Define max iterations for run control
    MAX_ITERATIONS = 75
    iteration_count = 0
    # Retrieve and Check status until completed
    while run.status not in ["completed", "failed", "cancelled", "incomplete", "expired"]:
        iteration_count += 1
        if iteration_count > MAX_ITERATIONS:
            run = client.beta.threads.runs.cancel(thread_id=thread.id,run_id=run.id)
            logging.error(f"%s|%s| Maximum iterations reached in RUN cycle. Exiting.", id_employee, company)
            break
        # Verifiy running status 
        if run.status in ["queued", "in_progress", "cancelling"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
            logging.info(f"%s|%s| RUN STATUS: {run.status}",id_employee,company)
        # Handle the requires action
        elif run.status == "requires_action":
            logging.info(f"%s|%s| ENTER REQUIRED ACTION: {run.status}", id_employee, company)
            action_result = handle_required_action(client, run, thread.id, company, id_employee, is_admin)
            
            if isinstance(action_result, dict) and action_result.get("status") == "error":
                logging.error(f"%s|%s| ACTION FAILED: {action_result.get('message')}", id_employee, company)
                response.update({"assistant": action_result.get("message")})
                logging.error(f"%s|%s| TOKENS: {run}", id_employee, company)
                return format_response(question, id_employee, company, database, response, thread.id, tokens_use)
            else:
                run = action_result

    # Retrieve message when status is completed or failed
    if run.status == "completed":
        logging.info(f"%s|%s| RUN STATUS: {run.status}",id_employee,company)
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        logging.info(f"%s|%s| MESSAGES: {messages}",id_employee,company)
        last_message = messages.data[0]
        response[last_message.role] = retrieve_annotation(client, thread.id, last_message.id)
    else: # Status posibles ["failed", "cancelled", "incomplete", "expired", "unhandle"]
        logging.info(f"%s|%s| RUN ENDED WITH STATUS: {run.status}", id_employee, company)
        logging.info(f"%s|%s| RUN DETAIL: {run}", id_employee, company)
        response.update({"assistant": "Lo sentimos, en este momento la respuesta no está disponible. Si necesita ayuda inmediata, por favor contacte directo a su departamento de RH."})
        
    tokens_use = run.usage.total_tokens or 0
    logging.info(f"%s|%s| RUN USAGE: {tokens_use}",id_employee,company)
    # Return JSON response
    return format_response(question, id_employee, company, database, response, thread.id, tokens_use)

def transcribe(id_employee: str, company: str, audio):
    """
    Transcribe el audio a text usando whisper 1 de Open AI (auto deteccion de idioma).
    
    Parameters:
        audio (file-like): A Django `request.FILES['audio']` o un archivo abierto con `open(path, 'rb')`
    
    Returns:
        str: Texto transcrito o False
    """
    try:
        logging.info(f"%s|%s| BEGIN TRANSCRIPTION: {audio}",id_employee,company)
        # Convertimos el InMemoryUploadedFile a BytesIO
        audio_bytes = io.BytesIO(audio.read())
        audio_bytes.name = audio.name  # Agrega nombre al archivo
        audio_bytes.seek(0)  # Asegura que empieza desde el inicio

        # Llamada a la API Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="text"
        )
        logging.info(f"%s|%s| TRANSCRIPTION: %s", id_employee, company, transcript)
        return transcript
    except Exception as e:
        logging.error(f"%s|%s| ERROR TRANSCRIPTION: %s", id_employee, company, str(e))
        return False
    
def format_response(question, id_employee, company, database, response, thread_id, tokens):
    return {
        "question": question,
        "id_employee": id_employee,
        "compania": company,
        "database": database,
        "response": response,
        "thread_id": thread_id,
        "tokens": tokens
    }