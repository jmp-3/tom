"""The policy index: chunks, their embeddings, and the metadata that dates it.

Two files instead of one, both generated and neither committed:

*   `index/policies.json` -- the metadata and the chunks as readable text. This
    is the file you open to answer "why did it retrieve 4.2 and not 4.1?", and
    half a megabyte of floats in the same document would make it unreadable.
*   `index/policies.npy` -- the embedding matrix, 25 x 1024 float32, already
    L2-normalized so that retrieval is a dot product and nothing else.

No vector database. Twenty-five chunks are a single `np.dot`, and FAISS or
Chroma would add a dependency, a storage format and a second place for the
index to go stale, in exchange for a search that is already instantaneous. What
a generic vector store would *not* carry is the part that matters here: the
staleness metadata below.

`embedding_model`, `dimension` and `pdf_digest` are written next to the vectors
because the failure they prevent is silent. A vector produced by one embedding
model and compared against a query embedded by another yields a number that is
a perfectly valid cosine and a meaningless ranking: the agent keeps answering,
with the wrong section. The CLI compares the three on startup and says to
rebuild.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from langchain_core.embeddings import Embeddings
from numpy.typing import NDArray

from emporio import bedrock
from emporio.config import PROJECT_ROOT, get_settings
from emporio.rag.chunker import Chunk, PolicyError, chunk_policies, pdf_digest, policies_path

DEFAULT_INDEX_DIR: Final = PROJECT_ROOT / "index"
CHUNKS_FILE: Final = "policies.json"
VECTORS_FILE: Final = "policies.npy"


@dataclass(frozen=True)
class PolicyIndex:
    """The built index, in memory. `vectors[i]` is the embedding of `chunks[i]`."""

    chunks: list[Chunk]
    vectors: NDArray[np.float32]
    embedding_model: str
    pdf_digest: str
    built_at: str

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True)
class IndexCheck:
    """Why a stored index can no longer be trusted. All false means it can."""

    missing: bool = False
    model_changed: bool = False
    pdf_changed: bool = False

    @property
    def usable(self) -> bool:
        return not (self.missing or self.model_changed or self.pdf_changed)


def _normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale every row to unit length, so cosine similarity is a dot product.

    Done once, at build time, rather than on every query: the division is the
    only part of the similarity that does not depend on the question.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A chunk can only embed to the zero vector if it is empty, which the
    # chunker does not produce -- guarded anyway, because the alternative is a
    # row of NaN that makes every later comparison quietly false.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def build_index(
    *,
    embeddings: Embeddings | None = None,
    pdf: Path | None = None,
    index_dir: Path | None = None,
) -> PolicyIndex:
    """Chunk the manual, embed every chunk, and write both index files.

    One `embed_documents` call for the whole document: batching is what the
    framework's `Embeddings` class is for, and 25 separate `invoke_model` calls
    would be 25 round trips for the same bytes.
    """
    target = pdf if pdf is not None else policies_path()
    chunks = chunk_policies(target)
    model = embeddings if embeddings is not None else bedrock.embedding_model()

    embedded = model.embed_documents([chunk.text for chunk in chunks])
    # Checked before the conversion: a short batch would otherwise become a
    # matrix whose rows no longer line up with the chunks, and the mismatch
    # would surface as a wrong citation rather than as an error.
    if len(embedded) != len(chunks):
        raise PolicyError(
            f"the embedding model returned {len(embedded)} vectors for {len(chunks)} chunks"
        )
    vectors = _normalize(np.asarray(embedded, dtype=np.float32))

    index = PolicyIndex(
        chunks=chunks,
        vectors=vectors,
        # Off the model object when it names itself, which is what makes the
        # stored id the one actually used rather than the one configured.
        embedding_model=str(getattr(model, "model_id", "") or get_settings().embed_model_id),
        pdf_digest=pdf_digest(target),
        built_at=datetime.now().isoformat(timespec="seconds"),
    )
    write_index(index, index_dir)
    return index


def write_index(index: PolicyIndex, index_dir: Path | None = None) -> tuple[Path, Path]:
    """Write both files, creating `index/` if it is not there.

    `index/` is in the `.gitignore`, and git does not version an empty
    directory, so a fresh clone has no such folder: creating it on demand is
    what makes `build-index` the first command that works after cloning.
    """
    directory = index_dir if index_dir is not None else DEFAULT_INDEX_DIR
    directory.mkdir(parents=True, exist_ok=True)
    chunks_path = directory / CHUNKS_FILE
    vectors_path = directory / VECTORS_FILE

    payload: dict[str, Any] = {
        "embedding_model": index.embedding_model,
        "dimension": index.dimension,
        "built_at": index.built_at,
        "pdf_digest": index.pdf_digest,
        "chunks": [
            {
                "id": c.id,
                "section": c.section,
                "title": c.title,
                "parent": c.parent,
                "page": c.page,
                "text": c.text,
            }
            for c in index.chunks
        ],
    }
    chunks_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.save(vectors_path, index.vectors)
    return chunks_path, vectors_path


def load_index(index_dir: Path | None = None) -> PolicyIndex:
    """Read the index back. Raises `PolicyError` when it is absent or broken."""
    directory = index_dir if index_dir is not None else DEFAULT_INDEX_DIR
    chunks_path = directory / CHUNKS_FILE
    vectors_path = directory / VECTORS_FILE
    if not chunks_path.is_file() or not vectors_path.is_file():
        raise PolicyError(f"no policy index in {directory}. Build it with `emporio build-index`.")

    try:
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path).astype(np.float32)
        chunks = [Chunk(**chunk) for chunk in payload["chunks"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PolicyError(f"the policy index in {directory} is unreadable: {exc}") from exc

    # The two files are written together and read separately, so a half-copied
    # pair is possible; it has to fail here rather than rank the wrong text.
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise PolicyError(
            f"the policy index in {directory} is inconsistent: {len(chunks)} chunks against "
            f"a {vectors.shape} matrix. Rebuild it with `emporio build-index`."
        )
    if int(payload.get("dimension", vectors.shape[1])) != vectors.shape[1]:
        raise PolicyError(
            f"the policy index in {directory} declares dimension "
            f"{payload['dimension']} and stores {vectors.shape[1]}."
        )

    return PolicyIndex(
        chunks=chunks,
        vectors=vectors,
        embedding_model=str(payload.get("embedding_model", "")),
        pdf_digest=str(payload.get("pdf_digest", "")),
        built_at=str(payload.get("built_at", "")),
    )


def check_index(index_dir: Path | None = None, *, pdf: Path | None = None) -> IndexCheck:
    """Whether the stored index still matches the configured model and the PDF.

    Never raises: this runs on startup, where the answer is a warning and not a
    crash. An unreadable index counts as missing, which is also the advice --
    rebuild it.
    """
    try:
        index = load_index(index_dir)
    except PolicyError:
        return IndexCheck(missing=True)

    model_changed = index.embedding_model != get_settings().embed_model_id
    try:
        pdf_changed = bool(index.pdf_digest) and index.pdf_digest != pdf_digest(pdf)
    except (PolicyError, OSError):
        # The manual is not where it was. The index is still internally
        # consistent, so it is not stale -- it is just no longer verifiable.
        pdf_changed = False
    return IndexCheck(model_changed=model_changed, pdf_changed=pdf_changed)
