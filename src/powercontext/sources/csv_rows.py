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

"""Row-level CSV Sources with stable identity and standard text evidence."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]

_UTF8_BOM = b"\xef\xbb\xbf"
_IDENTITY_PREFIX = "csv:"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document keyed by document and primary key."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One resolved data row with its original columns and values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured row Source whose identity is stable across reorderings."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def validate_row_width(self) -> CSVRowSource:
        """Keep ``columns`` and ``values`` aligned so projection never drops a field."""

        if len(self.columns) != len(self.values):
            raise ValueError("columns and values must describe the same row width")
        return self


def _validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    return document_id


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the fixed row identity with percent-encoded components."""

    _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value:
        raise ValueError("primary_key_value must be a non-empty string")
    identity = (
        f"{_IDENTITY_PREFIX}{quote(document_id, safe='')}:"
        f"{quote(primary_key_value, safe='')}"
    )
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def _parse_csv_records(text: str) -> list[list[str]]:
    try:
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter=",",
            quotechar='"',
            doublequote=True,
            strict=True,
        )
        return [record for record in reader]
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error


def _validate_header(header: list[str]) -> tuple[str, ...]:
    if not header:
        raise ValueError("CSV document must not have an empty header record")
    seen: set[str] = set()
    for column in header:
        if not column or not column.strip():
            raise ValueError("header columns must be non-empty and not pure whitespace")
        if column in seen:
            raise ValueError(f"header columns must be unique: {column!r}")
        seen.add(column)
    return tuple(header)


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse one complete CSV document into validated row inputs atomically."""

    if not isinstance(value.content, (bytes, bytearray)):
        raise ValueError("CSV content must be bytes")
    content = bytes(value.content)
    if content.startswith(_UTF8_BOM):
        content = content[len(_UTF8_BOM) :]
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV document must be valid UTF-8: {error}") from error

    _validate_document_id(value.document_id)
    if not isinstance(value.primary_key, str) or not value.primary_key:
        raise ValueError("primary_key must be a non-empty string")

    records = _parse_csv_records(text)
    if not records:
        raise ValueError("CSV document must not be empty")
    header = _validate_header(records[0])
    if value.primary_key not in header:
        raise ValueError(f"primary_key must exactly match one header column: {value.primary_key!r}")
    primary_key_index = header.index(value.primary_key)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            # Completely blank physical records after the header are ignored.
            continue
        if len(record) != len(header):
            raise ValueError(
                f"data record width {len(record)} does not match header width {len(header)}"
            )
        primary_key_value = record[primary_key_index]
        if not primary_key_value or not primary_key_value.strip():
            raise ValueError("primary key values must not be empty or pure whitespace")
        if primary_key_value in seen_keys:
            raise ValueError(f"primary key values must be unique within one document: {primary_key_value!r}")
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=value.document_id,
                primary_key=value.primary_key,
                primary_key_value=primary_key_value,
                columns=header,
                values=tuple(record),
            )
        )
    return tuple(rows)


class _CSVRowAdapter:
    """Resolve one row input into a captured row Source and read it back."""

    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class _CSVRowTextEvidenceProjection:
    """Standard powercontext.text-evidence projection for one CSV row."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> object:
        row = dict(zip(source.columns, source.values, strict=True))
        content = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        return {
            "source_type": "csv-row",
            "source_id": source.source_id,
            "content": content,
            "metadata": {
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        }


CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    adapter=_CSVRowAdapter(),
    version="1",
    projections=(_CSVRowTextEvidenceProjection(),),
)
