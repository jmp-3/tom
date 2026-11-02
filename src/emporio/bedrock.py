"""Amazon Bedrock access.

A single entry point to the models. Conversation goes through the Converse API
(`converse`), which normalizes messages and `toolConfig` across model families;
embeddings are not served by the Converse API, so they use `invoke_model`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import boto3

from emporio.config import get_settings


@lru_cache(maxsize=2)
def get_client(service: str = "bedrock-runtime"):
    """Return a boto3 Bedrock client.

    With `AWS_BEARER_TOKEN_BEDROCK` in the environment, boto3 authenticates with
    an API key and needs no IAM credentials; otherwise it falls back to the
    standard credential chain.
    """
    settings = get_settings()
    return boto3.client(service, region_name=settings.region)


def converse(
    prompt: str,
    *,
    model_id: str | None = None,
    system: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Run a single conversation turn. Returns the raw Converse API response."""
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "modelId": model_id or settings.model_id,
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    return get_client().converse(**kwargs)


def text_of(response: dict[str, Any]) -> str:
    """Join the text blocks of a Converse API response."""
    blocks = response["output"]["message"]["content"]
    return "".join(b["text"] for b in blocks if "text" in b).strip()


def embed(text: str, *, model_id: str | None = None) -> list[float]:
    """Return the embedding vector for a piece of text."""
    settings = get_settings()
    response = get_client().invoke_model(
        modelId=model_id or settings.embed_model_id,
        body=json.dumps({"inputText": text}),
    )
    return json.loads(response["body"].read())["embedding"]
