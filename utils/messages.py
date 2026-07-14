from openai import OpenAI
from settings import settings


client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    organization=settings.ORG_ID,
    project=settings.PROJECT,
)


def get_value(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_text_and_references(content):
    if isinstance(content, str):
        return content, []

    text_parts = []
    references = []
    seen_references = set()

    for part in content or []:
        text = get_value(part, "text", "")
        if text:
            text_parts.append(text)

        for annotation in get_value(part, "annotations", []) or []:
            filename = get_value(annotation, "filename")
            url = get_value(annotation, "url")
            reference = filename or url

            if reference and reference not in seen_references:
                seen_references.add(reference)
                references.append(reference)

    return "".join(text_parts), references


def format_message(item):
    if get_value(item, "type") != "message":
        return None

    role = get_value(item, "role")
    if role not in {"user", "assistant"}:
        return None

    text, references = extract_text_and_references(get_value(item, "content", []))
    if not text:
        return None

    message = {
        "id": get_value(item, "id"),
        "role": role,
        "content": text,
    }
    if references:
        message["references"] = references

    return message


def retrieve_messages_thread(conversation_id):
    """
    Recupera los mensajes de una conversación de la Responses API.
    """
    response = {}
    try:
        items = client.conversations.items.list(
            conversation_id=conversation_id,
            limit=100,
            order="asc",
        )

        messages = [
            message
            for item in getattr(items, "data", [])
            if (message := format_message(item)) is not None
        ]

        response["conversation_id"] = conversation_id
        response["messages"] = messages

        for index, message in enumerate(messages):
            response[index] = {message["role"]: message["content"]}
    except Exception as e:
        return {"error": str(e)}
    return response
