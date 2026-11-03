"""The policy pipeline without Bedrock: chunking, the index files, the ranking.

What is testable offline is everything except the embeddings themselves: that
the manual is cut at its own section boundaries and nowhere else, that the
tables survive as rows, that the two index files are written together and
refuse to be read apart, that a stale index is detected instead of silently
trusted, and that the retriever ranks by the dot product and hands back the
section to cite. Whether `amazon.titan-embed-text-v2:0` puts 4.1 above 4.2 for
"me arrependi da compra" is a question about a model, and that one is answered
by asking it -- see the README.

The embeddings here are word counts over the vocabulary of the texts being
embedded. They are not semantic and are not meant to be: they make the ranking
deterministic, which is what the arithmetic of the retriever can be checked
against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from emporio.agent import PUBLIC_TOOLS, STORE_TOOLS
from emporio.agent.tools import search_policies
from emporio.rag import (
    CHUNKS_FILE,
    VECTORS_FILE,
    Chunk,
    PolicyError,
    PolicyIndex,
    PolicyRetriever,
    build_index,
    check_index,
    chunk_policies,
    load_index,
    policies_path,
    write_index,
)

# The ten sections and eighteen subsections of the manual, in reading order.
# Sections 5, 6 and 8 are headings with no text of their own -- their titles go
# on as the parent of the subsections below them -- so they are not here.
EXPECTED_SECTIONS = [
    "1",
    "1.1",
    "1.2",
    "2",
    "3",
    "3.1",
    "4",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "5.1",
    "5.2",
    "5.3",
    "6.1",
    "6.2",
    "7",
    "7.1",
    "7.2",
    "7.3",
    "8.1",
    "8.2",
    "8.3",
    "9",
    "10",
]


class WordEmbeddings(Embeddings):
    """Word counts over a vocabulary learned from the documents.

    Deterministic, offline, and ranks by literal word overlap, which is enough
    to assert that the retriever sorts by similarity and returns `k` of them.
    """

    def __init__(self) -> None:
        self.vocabulary: list[str] = []

    @staticmethod
    def _words(text: str) -> list[str]:
        return re.findall(r"\w+", text.casefold())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.vocabulary = sorted({word for text in texts for word in self._words(text)})
        return [self._count(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._count(text)

    def _count(self, text: str) -> list[float]:
        words = self._words(text)
        return [float(words.count(term)) for term in self.vocabulary]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return chunk_policies()


@pytest.fixture
def embeddings() -> WordEmbeddings:
    """One embedder per test: its vocabulary is learned while the index is built.

    The same instance has to answer the queries afterwards, or the query vector
    would be counted over a different vocabulary -- a different space.
    """
    return WordEmbeddings()


@pytest.fixture
def index(tmp_path: Path, embeddings: WordEmbeddings) -> PolicyIndex:
    """An index over the real manual, embedded by word counts and written to disk."""
    return build_index(embeddings=embeddings, index_dir=tmp_path / "index")


def section(chunks: list[Chunk], number: str) -> Chunk:
    return next(chunk for chunk in chunks if chunk.section == number)


class TestChunking:
    """The manual is cut at its own headings, and at nothing else."""

    def test_every_numbered_section_with_a_body_becomes_one_chunk(
        self, chunks: list[Chunk]
    ) -> None:
        assert [chunk.section for chunk in chunks] == EXPECTED_SECTIONS

    def test_a_numbered_list_item_is_not_mistaken_for_a_heading(self, chunks: list[Chunk]) -> None:
        """Section 7.2 is a list numbered 1 to 5, shaped exactly like a heading.

        The five items belong to 7.2's text. Were they read as headings, the
        document would gain a second section 1 and lose the rest of 7.
        """
        assert "1. Saudação" in section(chunks, "7.2").text
        assert "5. Fechamento" in section(chunks, "7.2").text
        assert len({chunk.section for chunk in chunks}) == len(chunks)

    def test_a_heading_with_no_text_of_its_own_becomes_the_parent_instead(
        self, chunks: list[Chunk]
    ) -> None:
        assert "8" not in {chunk.section for chunk in chunks}
        assert section(chunks, "8.1").parent == "Garantia"

    def test_a_subsection_opens_with_its_own_heading_and_its_parent(
        self, chunks: list[Chunk]
    ) -> None:
        first_line = section(chunks, "4.1").text.splitlines()[0]
        assert first_line == (
            "4.1 Direito de Arrependimento (Compras Online) "
            "(dentro de 4. Política de Trocas e Devoluções)"
        )

    def test_the_tables_survive_as_rows(self, chunks: list[Chunk]) -> None:
        """The four tables hold the numbers customers ask about most."""
        assert "| Sábado | 09:00 às 13:00 |" in section(chunks, "2").text
        assert "| PIX | Pagamento à vista | 5% de desconto" in section(chunks, "3").text
        assert "| SEDEX (Correios) | 2 a 5 dias úteis |" in section(chunks, "5.2").text
        assert "| CNPJ | 12.345.678/0001-90 |" in section(chunks, "1.2").text

    def test_the_running_header_and_the_page_number_are_gone(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            assert "Manual de Políticas e Procedimentos" not in chunk.text
            assert not re.search(r"^Página \d+$", chunk.text, flags=re.MULTILINE)

    def test_the_e_mail_is_the_address_that_resolves(self, chunks: list[Chunk]) -> None:
        """The manual spells the domain with an accent in 1.2 and without it on the cover."""
        assert "contato@emporiodamusica.com.br" in section(chunks, "1.2").text
        for chunk in chunks:
            assert "empóriodamusica" not in chunk.text

    def test_the_return_window_is_in_4_1_and_not_in_its_neighbours(
        self, chunks: list[Chunk]
    ) -> None:
        """The section the withdrawal question has to land on, by content."""
        assert "7 (sete) dias corridos" in section(chunks, "4.1").text
        assert "arrependimento" in section(chunks, "4.1").text.casefold()
        assert "30 (trinta) dias" in section(chunks, "4.2").text


class TestManualLookup:
    """Finding the PDF, whose delivered name carries an accent."""

    def test_the_delivered_name_is_found(self) -> None:
        assert policies_path().is_file()

    def test_a_renamed_manual_is_still_found(self, tmp_path: Path) -> None:
        renamed = tmp_path / "policies.pdf"
        renamed.write_bytes(policies_path().read_bytes())
        assert policies_path(tmp_path) == renamed

    def test_no_manual_at_all_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyError, match="no PDF found"):
            policies_path(tmp_path)

    def test_two_candidates_refuse_to_guess(self, tmp_path: Path) -> None:
        for name in ("one.pdf", "two.pdf"):
            (tmp_path / name).write_bytes(b"%PDF-1.4")
        with pytest.raises(PolicyError, match="more than one PDF"):
            policies_path(tmp_path)


class TestIndexFiles:
    """Two generated files, written together and refusing to be read apart."""

    def test_the_directory_is_created_on_demand(self, tmp_path: Path) -> None:
        """A fresh clone has no `index/`: it is in the .gitignore in full."""
        target = tmp_path / "does" / "not" / "exist"
        build_index(embeddings=WordEmbeddings(), index_dir=target)
        assert (target / CHUNKS_FILE).is_file()
        assert (target / VECTORS_FILE).is_file()

    def test_the_chunks_file_stays_readable_by_a_person(
        self, index: PolicyIndex, tmp_path: Path
    ) -> None:
        payload: dict[str, Any] = json.loads(
            (tmp_path / "index" / CHUNKS_FILE).read_text(encoding="utf-8")
        )
        assert payload["chunks"][0].keys() == {
            "id",
            "section",
            "title",
            "parent",
            "page",
            "text",
        }
        assert "4.1" in {chunk["section"] for chunk in payload["chunks"]}

    def test_what_was_written_is_what_comes_back(self, index: PolicyIndex, tmp_path: Path) -> None:
        stored = load_index(tmp_path / "index")
        assert stored.chunks == index.chunks
        assert stored.embedding_model == index.embedding_model
        assert stored.pdf_digest == index.pdf_digest
        assert np.allclose(stored.vectors, index.vectors)

    def test_every_row_is_unit_length(self, index: PolicyIndex) -> None:
        """Normalized at build time, which is what makes the search a dot product."""
        assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)

    def test_a_half_copied_pair_refuses_to_load(self, index: PolicyIndex, tmp_path: Path) -> None:
        np.save(tmp_path / "index" / VECTORS_FILE, index.vectors[:-3])
        with pytest.raises(PolicyError, match="inconsistent"):
            load_index(tmp_path / "index")

    def test_a_missing_index_is_a_warning_and_not_an_exception(self, tmp_path: Path) -> None:
        check = check_index(tmp_path / "nothing")
        assert check.missing and not check.usable

    def test_an_index_built_with_another_embedding_model_is_stale(
        self, index: PolicyIndex, tmp_path: Path
    ) -> None:
        """The silent failure this metadata exists for.

        Vectors from one model against a query from another still produce a
        cosine; it just ranks nonsense, and every answer stays fluent.
        """
        write_index(
            PolicyIndex(
                chunks=index.chunks,
                vectors=index.vectors,
                embedding_model="amazon.titan-embed-text-v1",
                pdf_digest=index.pdf_digest,
                built_at=index.built_at,
            ),
            tmp_path / "index",
        )
        check = check_index(tmp_path / "index")
        assert check.model_changed and not check.usable

    def test_a_manual_edited_after_the_build_is_stale(
        self, index: PolicyIndex, tmp_path: Path
    ) -> None:
        write_index(
            PolicyIndex(
                chunks=index.chunks,
                vectors=index.vectors,
                embedding_model=index.embedding_model,
                pdf_digest="0000000000000000",
                built_at=index.built_at,
            ),
            tmp_path / "index",
        )
        check = check_index(tmp_path / "index")
        assert check.pdf_changed and not check.usable

    def test_a_fresh_index_is_usable(self, index: PolicyIndex, tmp_path: Path) -> None:
        assert check_index(tmp_path / "index").usable


class TestRetriever:
    """The ranking, and the citation it carries back."""

    @pytest.fixture
    def retriever(self, index: PolicyIndex, embeddings: WordEmbeddings) -> PolicyRetriever:
        return PolicyRetriever(index=index, embeddings=embeddings, k=3)

    def test_it_returns_k_documents_best_first(self, retriever: PolicyRetriever) -> None:
        found = retriever.invoke("parcelamento sem juros parcela mínima")
        assert len(found) == 3
        scores = [document.metadata["score"] for document in found]
        assert scores == sorted(scores, reverse=True)

    def test_the_best_match_is_the_section_whose_words_were_asked_for(
        self, retriever: PolicyRetriever
    ) -> None:
        """Word counts, so this asserts the arithmetic and not the semantics."""
        found = retriever.invoke("rastreamento código Correios Jadlog despachado")
        assert found[0].metadata["section"] in {"5.2", "5.3"}

    def test_every_document_carries_the_section_to_cite(self, retriever: PolicyRetriever) -> None:
        document = retriever.invoke("garantia legal noventa dias")[0]
        assert document.metadata.keys() == {"section", "title", "parent", "page", "score"}
        assert document.page_content.startswith(document.metadata["section"])

    def test_k_is_honoured(self, index: PolicyIndex, embeddings: WordEmbeddings) -> None:
        retriever = PolicyRetriever(index=index, embeddings=embeddings, k=1)
        assert len(retriever.invoke("frete grátis")) == 1


class TestPolicyTool:
    """How the policy search reaches the model."""

    def test_it_is_offered_in_both_session_modes(self) -> None:
        """Store policy is published information: an anonymous visitor may read it."""
        assert search_policies in PUBLIC_TOOLS
        assert search_policies in STORE_TOOLS

    def test_the_model_only_gets_to_write_the_question(self) -> None:
        assert set(search_policies.args) == {"question"}

    def test_it_returns_the_excerpts_with_their_sections(
        self,
        index: PolicyIndex,
        embeddings: WordEmbeddings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        retriever = PolicyRetriever(index=index, embeddings=embeddings, k=2)
        monkeypatch.setattr("emporio.agent.tools.policy_retriever", lambda: retriever)

        payload = search_policies.invoke({"question": "frete grátis acima de quanto?"})

        assert payload["shown"] == 2
        assert len(payload["excerpts"]) == 2
        for excerpt in payload["excerpts"]:
            assert excerpt.keys() == {"section", "title", "text"}
            assert excerpt["section"] in EXPECTED_SECTIONS
