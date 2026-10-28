# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical-page PDF Sources with exact page-level evidence retrieval."""

from __future__ import annotations

import hashlib
import io
import math
import re
import unicodedata
from collections import Counter
from urllib.parse import quote, unquote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]

_IDENTITY_PREFIX = "pdf:"
_TERM = re.compile(r"[a-z]+|\d+")
_MAX_POOL = 160
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "at", "by", "from", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "what", "which", "who", "how",
    "why", "when", "where", "do", "does", "did", "can", "could", "should", "would", "will",
    "i", "we", "you", "it", "its", "our", "your", "this", "that", "these", "those", "please",
    "any", "all", "if", "then", "than", "so", "such", "also", "not", "no", "has", "have",
    "had", "reported", "report", "stated", "state", "given", "give", "use", "used", "using",
    "only", "details", "shown", "within", "about", "into", "over", "under", "between",
    "according", "based", "following", "above", "below", "there", "their", "them", "they",
})


class PDFDocumentInput(BaseModel):
    """One complete original PDF keyed by document identity."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page with the digest of the whole source document."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """A captured physical page whose identity follows the exact revision."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the fixed page identity from document, revision digest and page."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")
    if not isinstance(document_sha256, str) or len(document_sha256) != 64:
        raise ValueError("document_sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    identity = f"{_IDENTITY_PREFIX}{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        digest = hashlib.sha256(f"{document_id}\x1f{document_sha256}\x1f{page_number}".encode()).hexdigest()
        identity = f"{_IDENTITY_PREFIX}sha256:{digest}"
    return identity


def _read_pages(content: bytes) -> list[str]:
    """Extract every physical page of one complete PDF document."""

    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF document requires a user password")
        texts: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text()
            texts.append(extracted if isinstance(extracted, str) else "")
        return texts
    except ValueError:
        raise
    except Exception as error:  # pypdf raises a wide range of parser errors
        raise ValueError(f"PDF document could not be parsed completely: {error}") from error


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one complete PDF atomically."""

    if not isinstance(value.document_id, str) or not value.document_id.strip():
        raise ValueError("document_id must be a non-blank string")
    if not isinstance(value.content, (bytes, bytearray)):
        raise ValueError("PDF content must be bytes")
    content = bytes(value.content)
    if not content:
        raise ValueError("PDF document must not be empty")
    digest = hashlib.sha256(content).hexdigest()
    texts = _read_pages(content)
    if not texts:
        raise ValueError("PDF document does not contain any physical page")
    return tuple(
        PDFPageInput(
            document_id=value.document_id,
            document_sha256=digest,
            page_number=number,
            text=text.strip(),
        )
        for number, text in enumerate(texts, start=1)
    )


class _PDFPageAdapter:
    """Resolve one page input into a captured page Source and read it back."""

    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class _PDFPageTextEvidenceProjection:
    """Standard powercontext.text-evidence projection for one physical page."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> object:
        return {
            "source_type": "pdf-page",
            "source_id": source.name,
            "content": f"{source.document_id}\n{source.text}",
            "metadata": {
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        }


PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    adapter=_PDFPageAdapter(),
    version="1",
    projections=(_PDFPageTextEvidenceProjection(),),
)


def _analyze(text: str) -> list[str]:
    return _TERM.findall(unicodedata.normalize("NFKC", text).casefold())


def _content_terms(question: str) -> list[str]:
    """Return the ordered distinct evidence terms of one question."""

    ordered: dict[str, None] = {}
    for term in _analyze(question):
        if len(term) > 1 and term not in _QUERY_STOPWORDS:
            ordered.setdefault(term, None)
    return list(ordered)


def _query_plan(question: str) -> tuple[tuple[str, int], ...]:
    """Plan high-precision and high-recall lexical queries with per-query caps."""

    terms = _content_terms(question)
    # Two coarse queries carry the bulk of the evidence and the sliding windows only
    # rescue questions whose terms sit far apart. Every extra query costs a full
    # candidate hydration pass, and the budget has to hold for whole filing sets, so
    # the plan stays deliberately short.
    plan: list[tuple[str, int]] = [(question, 80)]
    if terms:
        plan.append((" ".join(terms), 80))
    for index in range(0, max(0, len(terms) - 3), 4):
        window = " ".join(terms[index : index + 4])
        if window:
            plan.append((window, 40))
        if len(plan) >= 4:
            break
    return tuple(plan)


async def _candidates(memory, question: str) -> tuple[dict, dict]:
    """Collect fused retrieval candidates with their rank evidence."""

    from powercontext.artifacts import MemoryCitation
    from powercontext.builtin.runtime import SearchMemoryRequest

    fused: dict[tuple[int, str], float] = {}
    citations: dict[tuple[int, str], object] = {}
    for variant, cap in _query_plan(question):
        page = await memory.search(SearchMemoryRequest(query=variant, mode="fts", limit=cap))
        for rank, hit in enumerate(page.hits):
            key = (hit.memory_ref.revision, hit.entry_id)
            fused[key] = fused.get(key, 0.0) + 1.0 / (60.0 + rank)
            citations.setdefault(
                key,
                MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                ),
            )
    return fused, citations


def _score(question: str, documents: dict[object, str], *, k1: float = 1.2, b: float = 0.75) -> dict:
    """Score pooled pages with IDF coverage, adjacency, window focus and number density."""

    terms = _content_terms(question)
    if not terms or not documents:
        return {key: (0.0, 0.0, 0.0, 0.0, 0.0) for key in documents}
    sequences = {key: _analyze(text) for key, text in documents.items()}
    tokenized = {key: Counter(tokens) for key, tokens in sequences.items()}
    total = len(tokenized)
    document_frequency: Counter[str] = Counter()
    for counts in tokenized.values():
        document_frequency.update(set(counts))
    idf = {
        term: math.log(1.0 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in terms
    }
    evidence_mass = sum(idf.values()) or 1.0
    pairs = list(zip(terms, terms[1:], strict=False))
    average = sum(sum(counts.values()) for counts in tokenized.values()) / max(1, total)
    scores: dict[object, tuple[float, float, float, float, float]] = {}
    for key, counts in tokenized.items():
        length = sum(counts.values()) or 1
        weighted = 0.0
        coverage = 0.0
        for term in terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            coverage += idf[term]
            weighted += idf[term] * frequency * (k1 + 1.0) / (
                frequency + k1 * (1.0 - b + b * length / max(1.0, average))
            )
        tokens = sequences[key]
        adjacency = 0.0
        if pairs:
            adjacent = set(zip(tokens, tokens[1:], strict=False))
            adjacency = sum(1 for pair in pairs if pair in adjacent) / len(pairs)
        focus = 0.0
        size, stride = 110, 55
        for start in range(0, max(1, len(tokens) - size + stride), stride):
            window = set(tokens[start : start + size])
            found = sum(1 for term in terms if term in window)
            if found / len(terms) > focus:
                focus = found / len(terms)
        density = sum(1 for token in tokens if token[:1].isdigit()) / length
        scores[key] = (weighted, coverage / evidence_mass, adjacency, focus, density)
    return scores



def _page_location(reference: object) -> tuple[str, int] | None:
    """Recover the exact document page addressed by one Source reference."""

    if getattr(reference, "source_type", None) != "pdf-page":
        return None
    source_id = getattr(reference, "source_id", "")
    if not isinstance(source_id, str) or not source_id.startswith(_IDENTITY_PREFIX):
        return None
    body = source_id[len(_IDENTITY_PREFIX) :]
    if body.startswith("sha256:"):
        return None
    document, _, page = body.rpartition(":")
    document, _, digest = document.rpartition(":")
    if not document or len(digest) != 64:
        return None
    try:
        page_number = int(page)
    except ValueError:
        return None
    if page_number < 1:
        return None
    return unquote(document), page_number


def _compact(text: str) -> str:
    """Return the lowercase alphanumeric skeleton used for name matching."""

    return "".join(character for character in text.lower() if character.isalnum())


def _document_cores(document_id: str) -> frozenset[str]:
    """Return the name-like cores that identify one document."""

    cores: set[str] = set()
    prefix = ""
    for character in _compact(document_id):
        if not character.isalpha():
            break
        prefix += character
    if len(prefix) >= 3:
        cores.add(prefix)
    for token in re.split(r"[^a-z0-9]+", document_id.lower()):
        if token.isalpha() and len(token) >= 4:
            cores.add(token)
    return frozenset(cores)


def _addressed_documents(question: str, document_ids: object) -> frozenset[str]:
    """Return the documents the question names outright."""

    skeleton = _compact(question)
    addressed = set()
    for document_id in document_ids:
        for core in _document_cores(document_id):
            if core and core in skeleton:
                addressed.add(document_id)
                break
    return frozenset(addressed)


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[object, ...]:
    """Return up to ``limit`` distinct exact page citations for one question."""

    from powercontext.builtin.runtime import GetMemoryEntryRequest

    if limit < 1:
        return ()
    if not isinstance(question, str) or not question.strip():
        return ()
    fused, citations = await _candidates(memory, question)
    if not fused:
        return ()

    # Read the strongest fused candidates first so the pool stays bounded under load.
    ranked_keys = sorted(fused, key=lambda key: (-fused[key], key))[:_MAX_POOL]
    texts: dict[tuple[int, str], str] = {}
    locations: dict[tuple[int, str], set[tuple[str, int]]] = {}
    for key in ranked_keys:
        entry = (await memory.get(GetMemoryEntryRequest(citation=citations[key]))).entry
        texts[key] = entry.text
        found: set[tuple[str, int]] = set()
        for reference in entry.sources:
            location = _page_location(reference)
            if location is not None:
                found.add(location)
        locations[key] = found

    lexical = _score(question, texts)
    coverage_peak = max((value[1] for value in lexical.values()), default=0.0) or 1.0
    fused_peak = max(fused.values(), default=0.0) or 1.0
    density_peak = max((value[4] for value in lexical.values()), default=0.0) or 1.0
    combined = {
        key: 0.50 * lexical[key][1] / coverage_peak
        + 0.15 * lexical[key][2]
        + 0.10 * lexical[key][3] / coverage_peak
        + 0.05 * fused[key] / fused_peak
        + 0.20 * lexical[key][4] / density_peak
        for key in lexical
    }
    ranked = sorted(combined, key=lambda key: (-combined[key], -lexical[key][1], key))
    # A named document is the strongest evidence boundary; other documents only
    # fill the remaining quota when the question names none or too little.
    addressed = _addressed_documents(question, {name for found in locations.values() for name, _ in found})
    scopes = (addressed, frozenset()) if addressed else (frozenset(),)
    results: list[object] = []
    seen: set[tuple[str, int]] = set()
    for scope in scopes:
        for key in ranked:
            found = locations[key]
            if not found:
                continue
            if scope and not any(name in scope for name, _ in found):
                continue
            # Keep citations and distinct pages in one-to-one correspondence.
            if len(found - seen) != 1:
                continue
            results.append(citations[key])
            seen.update(found)
            if len(results) >= limit:
                return tuple(results)
    return tuple(results)
