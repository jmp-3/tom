"""Cosine retrieval over the policy index, as a LangChain retriever.

A `BaseRetriever` subclass and not an `InMemoryVectorStore`, on purpose. The
search itself is three lines of numpy -- one dot product against a 25 x 1024
matrix -- so a vector store would contribute no speed; what it would cost is the
staleness metadata of `index.py`, which a generic store has no field for.

What the framework interface *is* worth is the seam: `BaseRetriever` is what the
`search_policies` tool consumes, so a different retrieval strategy (BM25,
reranking, a real vector store if the manual ever becomes a thousand pages)
replaces this class alone and nothing else changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from numpy.typing import NDArray
from pydantic import ConfigDict

from emporio import bedrock
from emporio.rag.index import PolicyIndex, load_index

# Three sections is what fits an answer: the manual cross-references itself
# (the PIX discount is in 3 and its exception in 6.2), and a single hit would
# make the agent state a rule and miss its exception.
DEFAULT_K: Final = 3


class PolicyRetriever(BaseRetriever):
    """Returns the sections of the manual closest to a question.

    `metadata` carries `section` and `title` because the citation is the point:
    the agent is told to answer policy questions with the section it applied, so
    a customer -- and whoever reads the transcript -- can check it against the
    document.
    """

    # `PolicyIndex` holds a numpy array, which pydantic cannot validate.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: PolicyIndex
    embeddings: Embeddings
    k: int = DEFAULT_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        scores = self.similarities(query)
        # `argsort` ascending, then reversed: at this size the cost of sorting
        # everything is below the cost of being clever about it.
        ranked = np.argsort(scores)[::-1][: max(1, self.k)]
        return [self._document(int(position), float(scores[position])) for position in ranked]

    def similarities(self, query: str) -> NDArray[np.float32]:
        """Cosine similarity between the question and every chunk.

        The stored rows are already unit length (see `index._normalize`), so
        normalizing the query makes the dot product the cosine itself.
        """
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return self.index.vectors @ vector

    def _document(self, position: int, score: float) -> Document:
        chunk = self.index.chunks[position]
        return Document(
            page_content=chunk.text,
            metadata={
                "section": chunk.section,
                "title": chunk.title,
                "parent": chunk.parent,
                "page": chunk.page,
                "score": round(score, 4),
            },
        )


@lru_cache(maxsize=1)
def policy_retriever(index_dir: Path | None = None, k: int = DEFAULT_K) -> PolicyRetriever:
    """The retriever the tool uses, built once per process.

    Cached because both halves are expensive for the same reason: the
    index is read from disk and the embedding client opens a boto3 session, and
    neither changes between two questions in one conversation. Raises
    `PolicyError` when the index has not been built.
    """
    return PolicyRetriever(index=load_index(index_dir), embeddings=bedrock.embedding_model(), k=k)
