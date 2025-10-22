import os
from typing import List
from openai import OpenAI

_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

_SYSTEM = (
    "You are a helpful assistant that must answer ONLY using the provided context. "
    'If the answer is not in the context, reply exactly: "I don\'t know."'
)

_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer only using the context. If not in the context, say \"I don't know.\""
)

def answer_with_context(question: str, context_docs: List[str], model: str = "gpt-3.5-turbo") -> str:
    client = _get_client()
    context = "\n\n---\n\n".join(context_docs[:6])  # cap context
    prompt = _TEMPLATE.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()