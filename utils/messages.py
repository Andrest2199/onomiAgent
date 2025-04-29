import os
from openai import OpenAI
from dotenv import load_dotenv

# Carga .env
load_dotenv()

# Configuramos las variables de entorno
api_key = os.environ.get("OPENAI_API_KEY")
org_id = os.environ.get("OPENAI_ORG_ID")

def retrieve_annotation(client,thread_id,message_id):
    """
    Recupera las referencias a archivos.
    """
    unique_citations = {}
    citations = []
    citation_index = 1
    # Retrieve the message object
    message = client.beta.threads.messages.retrieve(thread_id=thread_id,message_id=message_id)
    # Extract the message and the annotation
    message_content = message.content[0].text.value
    annotations = message.content[0].text.annotations
    # Iterate over the annotations and add footnotes
    for annotation in annotations:
        # Check if the annotation text already has a citation to avoid duplicates
        citation_text = annotation.text
        # Determine the source of the citation
        if (file_citation := getattr(annotation, 'file_citation', None)):
            file_id = file_citation.file_id
            try:
                cited_file = client.files.retrieve(file_id)
                filename = cited_file.filename
            except Exception as e:
                print(f"Error retrieving file {file_id}: {e}")
                filename = file_id
        elif (file_path := getattr(annotation, 'file_path', None)):
            file_id = file_path.file_id
            try:
                cited_file = client.files.retrieve(file_id)
                filename = cited_file.filename
            except Exception as e:
                print(f"Error retrieving file {file_id}: {e}")
                filename = file_id
        else:
            continue

        # Only add citation if it is unique
        if filename not in unique_citations:
            # Add to unique citations and append citation text with index
            unique_citations[filename] = citation_index
            citations.append(f'[{citation_index}] Fuente: {filename}')
            citation_index += 1  # Increment the citation index for the next unique file
        # Replace annotation text in the message content with the correct index
        message_content = message_content.replace(citation_text, f' [{unique_citations[filename]}]')
        # Note: File download functionality not implemented above for brevity
    if citations:
        message_content += "\n\nReferencias:\n" + "\n".join(citations)
    
    return message_content

def retrieve_messages_thread(thread_id):
    """
    Recupera todos los mensajes en un thread, agregando anotaciones y referencias.
    """
    client = OpenAI(organization=org_id,api_key=api_key)
    response = {}
    try:
        messages = client.beta.threads.messages.list(thread_id)
        for index, message in enumerate(messages):
            response[index] = {message.role: retrieve_annotation(client, thread_id, message.id)}
    except Exception as e:
        return {"error": str(e)}
    return response