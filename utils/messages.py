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


def get_conversation(conversation_id):
    try:
        return client.conversations.retrieve(conversation_id=conversation_id)
    except TypeError:
        return client.conversations.retrieve(conversation_id)


def get_conversation_metadata(conversation_id):
    conversation = get_conversation(conversation_id)
    return get_value(conversation, "metadata", {}) or {}


def get_conversation_chain(conversation_id):
    chain = []
    seen = set()
    current_id = conversation_id

    while current_id and current_id not in seen:
        seen.add(current_id)
        chain.append(current_id)

        metadata = get_conversation_metadata(current_id)
        current_id = metadata.get("previous_conversation_id")

    return list(reversed(chain))


def get_conversation_messages(conversation_id):
    items = client.conversations.items.list(
        conversation_id=conversation_id,
        limit=25,
        order="asc",
    )
    print(items)

    return [
        message
        for item in getattr(items, "data", [])
        if (message := format_message(item)) is not None
    ]


def retrieve_messages_thread(conversation_id):
    """
    Recupera los mensajes de una conversación de Responses API.
    """
    response = {}
    try:
        conversation_ids = get_conversation_chain(conversation_id)
        messages = []
        for current_id in conversation_ids:
            messages.extend(get_conversation_messages(current_id))

        response["conversation_id"] = conversation_id
        response["conversation_ids"] = conversation_ids
        response["messages"] = messages

        for index, message in enumerate(messages):
            response[index] = {message["role"]: message["content"]}
    except Exception as e:
        return {"error": str(e)}
    return response
