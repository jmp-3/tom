"""Cuts the policy manual into one chunk per numbered section.

The manual is eight pages of prose: ten numbered sections, eighteen numbered
subsections and four tables (company data, opening hours, payment methods,
shipping modes). Chunking follows that structure rather than a fixed window
because the document is already hierarchical and the section is the unit the
agent cites back to the customer ("conforme a seção 4.1"). A chunk that
straddled two sections would have nothing to cite.

Two extraction details earn the code they cost.

`pdfplumber` and not `pypdf`: the four tables hold the numbers customers ask
about most -- opening hours, the PIX discount, shipping deadlines -- and a
text-only extractor flattens their cells into one unreadable line. Here the
tables are located, read as cells and re-rendered as pipe rows, while the text
bands around them are read separately, so "Sábado | 09:00 às 13:00" survives as
a row.

Headings are recognized by font and not by their shape alone: the numbered
list of section 7.2 ("1. Saudação: cumprimentar o cliente...") is
indistinguishable from a top-level heading ("1. Sobre a Empório da Música") to
a regular expression. In this PDF every real heading is Arial-Bold at 12pt or
more and every list item is regular Arial at 10pt, so the weight is what tells
them apart -- measured on the file, not assumed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pdfplumber

from emporio.config import get_settings
from emporio.data.loader import normalize_text

# The file as it is delivered. `policies_path` also accepts it renamed, which is
# what makes the accent in the name harmless.
POLICIES_PDF_NAME: Final = "políticas_da_loja.pdf"

# `(section number, subsection number, title)`. The trailing dot is optional
# because top-level headings carry one ("4. Política de...") and subsections do
# not ("4.1 Direito de...").
_HEADING: Final = re.compile(r"^(\d{1,2})(?:\.(\d{1,2}))?\.?\s+(\S.*)$")

# Headings are 12pt (subsections) and 14pt, body text is 10pt. Compared on the
# rounded size because the sizes come out of the PDF with float dust: a 12pt
# heading reports 11.999999999999943, which a bare `>= 12.0` rejects -- and the
# section it silently swallowed was 1.2, the company data table.
_MIN_HEADING_SIZE: Final = 12.0

# Running header and footer, repeated on every page of the manual.
_BOILERPLATE: Final = (
    re.compile(r"^Empório da Música\s+Manual de Políticas e Procedimentos$"),
    re.compile(r"^Página \d+$"),
)

# The manual writes the store's e-mail twice and only one of the two resolves:
# the cover page has `contato@emporiodamusica.com.br`, the company table in
# section 1.2 spells the domain with the accent of the brand name. Corrected
# here, at extraction, rather than in the prompt: the agent may repeat this
# address to a customer, and a rule in the persona would be a second place to
# keep a fact that belongs to the document.
_CORRECTIONS: Final[tuple[tuple[str, str], ...]] = (
    ("empóriodamusica.com.br", "emporiodamusica.com.br"),
)


class PolicyError(RuntimeError):
    """The policy manual is missing, unreadable, or not shaped as expected."""


@dataclass(frozen=True)
class Chunk:
    """One section of the manual, with the citation it is meant to produce.

    `text` is what gets embedded and what the model reads; it opens with the
    heading, and with the parent heading for a subsection, so that a chunk
    carries its own context. Without it, "em até 7 dias corridos" retrieves as
    a sentence about seven days and not as a sentence about returns.
    """

    id: int
    section: str
    title: str
    parent: str
    page: int
    text: str


def policies_path(data_dir: Path | None = None) -> Path:
    """Locate the policy manual, by exact name first and then by stem.

    The delivered name carries an accent, and a name written on one machine and
    read on another can be NFC on one side and NFD on the other: different
    bytes, identical on screen. Comparing accent-stripped stems is what makes
    the lookup survive that, and it also accepts the file renamed to something
    plain -- which is the first thing anyone tries when an accented name fails.
    """
    directory = data_dir if data_dir is not None else get_settings().data_dir
    exact = directory / POLICIES_PDF_NAME
    if exact.is_file():
        return exact

    wanted = normalize_text(Path(POLICIES_PDF_NAME).stem)
    pdfs = sorted(p for p in directory.glob("*.pdf") if p.is_file())
    by_stem = [p for p in pdfs if normalize_text(p.stem) == wanted]
    if len(by_stem) == 1:
        return by_stem[0]
    if len(pdfs) == 1:
        return pdfs[0]
    if not pdfs:
        raise PolicyError(
            f"no PDF found in {directory}. Expected {POLICIES_PDF_NAME}. Check DATA_DIR."
        )
    raise PolicyError(
        f"more than one PDF in {directory}: {[p.name for p in pdfs]}. "
        f"Leave only {POLICIES_PDF_NAME} in place."
    )


def pdf_digest(path: Path | None = None) -> str:
    """Short SHA-256 of the manual, stored with the index to detect a new edition."""
    target = path if path is not None else policies_path()
    return hashlib.sha256(target.read_bytes()).hexdigest()[:16]


def _clean(text: str) -> str:
    for wrong, right in _CORRECTIONS:
        text = text.replace(wrong, right)
    return " ".join(text.split())


def _is_boilerplate(text: str) -> bool:
    return any(pattern.match(text) for pattern in _BOILERPLATE)


def _render_table(rows: list[list[str | None]]) -> str:
    """A table as one pipe-separated row per line.

    Pipes and not prose because the rows are lookups: the model is asked "what
    time do you close on Saturday?" and has to find one cell. A cell wrapping
    over two lines in the PDF comes back with a newline inside it, which would
    break the row apart, so whitespace is collapsed per cell.
    """
    return "\n".join("| " + " | ".join(_clean(cell or "") for cell in row) + " |" for row in rows)


@dataclass(frozen=True)
class Line:
    """One line of the document: body text, a table block, or a heading."""

    page: int
    text: str
    heading: tuple[str, str] | None = None


def _heading_of(line: dict[str, Any], text: str) -> tuple[str, str] | None:
    """`("4.1", "Direito de Arrependimento")` when this line is a heading."""
    match = _HEADING.match(text)
    if match is None:
        return None
    first = line["chars"][0]
    if round(float(first["size"]), 1) < _MIN_HEADING_SIZE or "Bold" not in str(first["fontname"]):
        return None
    major, minor, title = match.groups()
    return (f"{major}.{minor}" if minor else major, title.strip())


def _page_lines(page: Any) -> Iterator[Line]:
    """The lines of one page, with each table emitted once, in its own place.

    Tables are found by their bounding box and the text lines falling inside
    one are dropped in favour of the rendered rows. Replacing the *first* such
    line keeps the table where the document put it, which matters because the
    chunk it belongs to is decided by the heading above it.
    """
    tables = sorted(page.find_tables(), key=lambda table: table.bbox[1])
    spans = [(table.bbox[1], table.bbox[3]) for table in tables]
    rendered = [_render_table(table.extract()) for table in tables]
    emitted: set[int] = set()

    for line in page.extract_text_lines():
        middle = (line["top"] + line["bottom"]) / 2
        inside = next((i for i, (top, bottom) in enumerate(spans) if top <= middle <= bottom), None)
        if inside is not None:
            if inside not in emitted:
                emitted.add(inside)
                yield Line(page.page_number, rendered[inside])
            continue
        text = _clean(line["text"])
        if not text or _is_boilerplate(text):
            continue
        yield Line(page.page_number, text, _heading_of(line, text))


def document_lines(path: Path | None = None) -> list[Line]:
    """Every line of the manual, in reading order. Exposed for debugging."""
    target = path if path is not None else policies_path()
    with pdfplumber.open(target) as pdf:
        return [line for page in pdf.pages for line in _page_lines(page)]


def _compose(section: str, title: str, parent: str, body: list[str]) -> str:
    """The text of a chunk: its heading, its parent's, then the body.

    Numbered as the document numbers itself -- "2." for a section, "4.1" for a
    subsection -- so that a citation read off the chunk matches the printed
    manual.
    """
    header = f"{section} {title}" if "." in section else f"{section}. {title}"
    if parent:
        header += f" (dentro de {section.split('.')[0]}. {parent})"
    return "\n".join([header, "", *body])


def chunk_policies(path: Path | None = None) -> list[Chunk]:
    """One chunk per numbered section or subsection that has a body.

    A heading with no text of its own ("8. Garantia", immediately followed by
    8.1) produces no chunk: a title alone embeds as a topic label and would win
    queries it cannot answer. The title is not lost -- it goes on as the parent
    of the subsections below it.
    """
    chunks: list[Chunk] = []
    section = title = parent = top_title = ""
    page = 1
    body: list[str] = []

    def close() -> None:
        nonlocal body
        if section and body:
            chunks.append(
                Chunk(
                    id=len(chunks),
                    section=section,
                    title=title,
                    parent=parent,
                    page=page,
                    text=_compose(section, title, parent, body),
                )
            )
        body = []

    for line in document_lines(path):
        if line.heading is None:
            body.append(line.text)
            continue
        close()
        section, title = line.heading
        page = line.page
        # A top-level heading becomes the context of everything numbered under
        # it; the heading itself has no parent.
        if "." in section:
            parent = top_title
        else:
            parent, top_title = "", title
    close()

    if not chunks:
        raise PolicyError(f"no numbered section found in {path or policies_path()}")
    return chunks
