import logging

from openai import OpenAI

from settings import settings
from utils.assistant_helper import AssistantHelper

# Configuramos las variables de entorno
api_key = settings.OPENAI_API_KEY
org_id = settings.ORG_ID
project = settings.PROJECT
log_file = settings.LOG_FILE

# Set log
logging.basicConfig(
    filename=log_file,
    format="%(levelname)s|%(asctime)s|%(message)s",
    level=logging.INFO,
)

# Create instance of openAI client
client = OpenAI(api_key=api_key, organization=org_id, project=project)
assistant_helper = AssistantHelper(client)


# Main function
def onomi_assistant(
    company_id: str,
    company_name: str,
    id_employee: str,
    permission_type: str,
    question: str,
    agent_config: any,
    previous_conversation_id: str = None,
):
    system_instructions = assistant_helper.build_system_instructions(
        company_name, agent_config, permission_type
    )

    metadata = {
        "company": company_id,
        "user": id_employee,
        "tipo_permiso": permission_type,
    }
    conversation_id = previous_conversation_id or None
    input_payload = [{"role": "user", "content": question}]
    tools = assistant_helper.build_response_tools(
        question, agent_config, permission_type
    )

    try:
        conversation_id, metadata = assistant_helper.prepare_conversation(
            conversation_id=conversation_id,
            metadata=metadata,
            system_instructions=system_instructions,
            input_payload=input_payload,
            tools=tools,
            id_employee=id_employee,
            company_id=company_id,
            company_name=company_name,
        )

        response = assistant_helper.create_response(
            system_instructions=system_instructions,
            input_payload=input_payload,
            tools=tools,
            conversation_id=conversation_id,
            metadata=metadata,
            prompt_cache_key=assistant_helper.build_prompt_cache_key(company_id),
            output_token_limit=assistant_helper.max_output_tokens,
        )
        assistant_helper.log_response_status(
            "RESPONSE CREATED",
            id_employee,
            company_id,
            company_name,
            response,
            conversation_id,
        )

        responses = [response]
        max_tool_iterations = 8
        for _ in range(max_tool_iterations):
            tool_outputs = assistant_helper.build_tool_outputs(
                response, company_id, company_name, id_employee, permission_type
            )
            if not tool_outputs:
                break

            response = assistant_helper.create_response(
                system_instructions=system_instructions,
                input_payload=tool_outputs,
                tools=tools,
                conversation_id=conversation_id,
                metadata=metadata,
                prompt_cache_key=assistant_helper.build_prompt_cache_key(company_id),
                output_token_limit=assistant_helper.max_output_tokens,
            )
            responses.append(response)
            assistant_helper.log_response_status(
                "RESPONSE AFTER TOOLS",
                id_employee,
                company_id,
                company_name,
                response,
                conversation_id,
            )
        else:
            raise RuntimeError(
                "Máximo de iteraciones de tools alcanzadas en el flujo de Responses."
            )

        if assistant_helper.is_incomplete_for_max_output_tokens(response):
            logging.warning(
                "%s|%s|%s| RESPONSE INCOMPLETE RETRY: %s CONVERSATION: %s",
                id_employee,
                company_id,
                company_name,
                assistant_helper.get_response_value(response, "id"),
                conversation_id,
            )
            response = assistant_helper.create_response(
                system_instructions=system_instructions,
                input_payload=[
                    {
                        "role": "developer",
                        "content": (
                            "La respuesta anterior quedó incompleta por límite de tokens. "
                            "Entrega una respuesta final completa, breve y directa, sin repetir "
                            "razonamiento interno ni detalles técnicos."
                        ),
                    }
                ],
                tools=None,
                conversation_id=conversation_id,
                metadata=metadata,
                prompt_cache_key=assistant_helper.build_prompt_cache_key(company_id),
                output_token_limit=assistant_helper.retry_max_output_tokens,
            )
            responses.append(response)
            assistant_helper.log_response_status(
                "RESPONSE RETRY AFTER INCOMPLETE",
                id_employee,
                company_id,
                company_name,
                response,
                conversation_id,
            )

        tokens_use = assistant_helper.get_token_usage(responses)
        logging.info(
            "%s|%s|%s| RESPONSE USAGE: %s",
            id_employee,
            company_id,
            company_name,
            tokens_use,
        )
        return assistant_helper.format_response(
            company_id,
            company_name,
            id_employee,
            question,
            {"assistant": assistant_helper.get_response_output_text(response)},
            conversation_id,
            tokens_use,
        )
    except Exception as e:
        logging.error(
            "%s|%s|%s| ERROR RESPONSE FLOW: %s",
            id_employee,
            company_id,
            company_name,
            str(e),
        )
        return assistant_helper.format_response(
            company_id,
            company_name,
            id_employee,
            question,
            {
                "assistant": "Disculpe las molestias. En este momento, estamos presentando fallas, favor de contactar a soporte si lo requiere o espere unos minutos e intente de nuevo."
            },
            conversation_id,
            assistant_helper.empty_token_usage(),
        )


def transcribe(id_employee: str, company: str, audio):
    """
    Transcribe el audio a texto usando whisper-1 de OpenAI.

    Parameters:
        audio (file-like): FastAPI `UploadFile`, Django `request.FILES['audio']`,
            o un archivo abierto con `open(path, 'rb')`.

    Returns:
        Transcription | bool: Objeto con `.text` o False.
    """
    try:
        filename = getattr(audio, "filename", None) or getattr(audio, "name", "audio")
        content_type = getattr(audio, "content_type", None)
        audio_file = getattr(audio, "file", audio)

        if hasattr(audio_file, "seek"):
            audio_file.seek(0)

        file_payload = (
            (filename, audio_file, content_type)
            if content_type
            else (filename, audio_file)
        )

        logging.info("%s|%s|%s| BEGIN TRANSCRIPTION", id_employee, company, filename)

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=file_payload,
        )
        logging.info(
            "%s|%s|%s| TRANSCRIPTION: %s",
            id_employee,
            company,
            filename,
            transcript.text,
        )
        return transcript
    except Exception as e:
        logging.error(
            "%s|%s|%s| ERROR TRANSCRIPTION: %s",
            id_employee,
            company,
            getattr(audio, "filename", None) or getattr(audio, "name", "audio"),
            str(e),
        )
        return False
