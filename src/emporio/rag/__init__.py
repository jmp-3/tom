"""Retrieval over the store's policy manual.

The other axis of the agent's answers. The catalog and the orders are
structured data -- 65 products, 20 orders -- where a deterministic filter covers
the whole space of questions and keeps every number auditable; the policies are
eight pages of free text, where the question "posso devolver?" has no column to
filter on. Hence a typed tool over CSVs there and semantic retrieval here.
"""

from emporio.rag.chunker import (
    POLICIES_PDF_NAME,
    Chunk,
    PolicyError,
    chunk_policies,
    pdf_digest,
    policies_path,
)
from emporio.rag.index import (
    CHUNKS_FILE,
    DEFAULT_INDEX_DIR,
    VECTORS_FILE,
    IndexCheck,
    PolicyIndex,
    build_index,
    check_index,
    load_index,
    write_index,
)
from emporio.rag.retriever import DEFAULT_K, PolicyRetriever, policy_retriever

__all__ = [
    "CHUNKS_FILE",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_K",
    "POLICIES_PDF_NAME",
    "VECTORS_FILE",
    "Chunk",
    "IndexCheck",
    "PolicyError",
    "PolicyIndex",
    "PolicyRetriever",
    "build_index",
    "check_index",
    "chunk_policies",
    "load_index",
    "pdf_digest",
    "policies_path",
    "policy_retriever",
    "write_index",
]
