"""Amazon Bedrock access.

A single entry point to the models. Conversation goes through the Converse API
(`converse`), which normalizes messages and `toolConfig` across model families;
embeddings are not served by the Converse API, so they use `invoke_model`.

`chat_model` is the same Converse API seen through LangChain, which is what the
agent loop consumes, and `embedding_model` is its counterpart for the policy
index. `converse` and `embed` stay as the raw path: no framework in the way,
already proven against this account, and therefore the control to fall back on
when a LangChain error is ambiguous about where it came from.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

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


def chat_model(
    *,
    model_id: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    max_retries: int = 2,
    streaming: bool = False,
) -> ChatBedrockConverse:
    """Return the chat model the agent runs on, configured from the environment.

    One function, so that changing models is changing `BEDROCK_MODEL_ID` in the
    `.env` and nothing else -- which is what makes the fallback ladder cheap.

    `get_settings()` runs first on purpose: it loads the `.env`, and
    `ChatBedrockConverse` reads `AWS_BEARER_TOKEN_BEDROCK` from the environment
    while it is being constructed. Built in the other order, the API key would
    be invisible and authentication would fail. If a future version of
    `langchain-aws` stops honoring the key, the way out is to hand it the client
    this module already builds: `ChatBedrockConverse(client=get_client())`.

    `max_retries` is explicit rather than inherited: a silent retry inside an
    agent loop resends the whole history, so the cost of an accident grows with
    the conversation. `streaming` is explicit for a different reason -- Bedrock
    advertises streaming for Gemma, but `langchain-aws` does not have the model
    in its allowlist, so leaving it unset only produces a warning on every call.
    """
    settings = get_settings()
    return ChatBedrockConverse(
        # `model` is the public name of the field; `model_id` is its alias.
        model=model_id or settings.model_id,
        region_name=settings.region,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        disable_streaming=not streaming,
    )


def embedding_model(*, model_id: str | None = None) -> BedrockEmbeddings:
    """Return the embedding model the policy index is built and queried with.

    Here next to `chat_model` for the same reason: one place where a model id
    from the environment becomes a model object. The framework class rather than
    the `embed` below because it batches a whole document into one call, and
    because `Embeddings` is the interface the retriever expects.

    `get_settings()` runs first for the same reason as in `chat_model`: it loads
    the `.env`, and the constructor reads `AWS_BEARER_TOKEN_BEDROCK` from the
    environment while it is being built.
    """
    settings = get_settings()
    return BedrockEmbeddings(
        model_id=model_id or settings.embed_model_id,
        region_name=settings.region,
    )


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
