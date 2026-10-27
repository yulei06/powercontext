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

"""Deterministic context packs and prefix-reusing deltas for transport."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ContextPackError(ValueError):
    """A context pack or delta violated the transport protocol."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object, field_name: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{field_name} must be an array of Base64 strings")
    decoded: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{field_name} must contain Base64 strings")
        try:
            decoded.append(base64.b64decode(entry, validate=True))
        except (binascii.Error, ValueError) as error:
            raise ContextPackError(f"{field_name} contains an invalid Base64 chunk: {error}") from error
    return tuple(decoded)


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": _encode_items(items)}
    return f"sha256:{hashlib.sha256(_canonical(unsigned)).hexdigest()}"


def _require_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContextPackError(f"{field_name} must be a sha256:<hex> digest")
    return value


def _loads(raw: object, fields: frozenset[str]) -> dict[str, Any]:
    if not isinstance(raw, (bytes, bytearray)):
        raise ContextPackError("wire payload must be bytes")
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ContextPackError(f"wire payload is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ContextPackError("wire payload must be a JSON object")
    keys = set(payload)
    if keys != fields:
        missing = sorted(fields - keys)
        extra = sorted(keys - fields)
        raise ContextPackError(f"wire payload fields do not match the schema (missing={missing}, extra={extra})")
    return payload


def split_utf8_chunks(text: str, limit: int) -> tuple[bytes, ...]:
    """Split one text into newline-aligned UTF-8 chunks no larger than ``limit`` bytes."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ContextPackError("chunk_bytes must be a positive integer")
    items: list[bytes] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if buffer:
            items.append("".join(buffer).encode("utf-8"))
            buffer = []
            size = 0

    for line in text.splitlines(keepends=True):
        line_size = len(line.encode("utf-8"))
        if line_size <= limit:
            if size + line_size > limit:
                flush()
            buffer.append(line)
            size += line_size
            continue
        # A single line exceeding the cap falls back to per-character packing.
        flush()
        for character in line:
            width = len(character.encode("utf-8"))
            if width > limit:
                raise ContextPackError("chunk_bytes is too small to hold a single UTF-8 character")
            if size + width > limit:
                flush()
            buffer.append(character)
            size += width
    flush()
    return tuple(items)


@dataclass(frozen=True)
class ContextPack:
    """One transport packet carrying ordered immutable content chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int = field(default=0)

    @classmethod
    def build(cls, items: object) -> ContextPack:
        """Build a complete pack from an ordered sequence of byte chunks."""

        try:
            normalized = tuple(item if isinstance(item, bytes) else bytes(item) for item in items)  # type: ignore[union-attr]
        except TypeError as error:
            raise ContextPackError("pack items must be a sequence of bytes") from error
        digest = _pack_digest(normalized)
        probe = cls(items=normalized, digest=digest, byte_count=0)
        size = len(probe._wire(0))  # noqa: SLF001 - same-class probe
        while True:
            measured = len(probe._wire(size))  # noqa: SLF001 - same-class probe
            if measured == size:
                break
            size = measured
        return cls(items=normalized, digest=digest, byte_count=size)

    @classmethod
    def from_content(cls, content: object, *, chunk_bytes: int = 512) -> ContextPack:
        """Chunk one UTF-8 text deterministically without changing a byte."""

        if isinstance(content, str):
            try:
                data = content.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ContextPackError(f"content is not encodable as UTF-8: {error}") from error
            text = content
        elif isinstance(content, (bytes, bytearray)):
            data = bytes(content)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError(f"content is not valid UTF-8: {error}") from error
        else:
            raise ContextPackError("content must be str or bytes")
        items = split_utf8_chunks(text, chunk_bytes)
        if b"".join(items) != data:
            raise ContextPackError("chunking did not preserve the original content bytes")
        return cls.build(items)

    def _wire(self, byte_count: int) -> bytes:
        return _canonical(
            {
                "schema": PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": byte_count,
            }
        )

    def to_bytes(self) -> bytes:
        """Serialize the pack exactly as sent on the wire."""

        return self._wire(self.byte_count)

    @classmethod
    def from_bytes(cls, raw: object) -> ContextPack:
        """Validate and decode one complete pack from the wire."""

        payload = _loads(raw, _PACK_FIELDS)
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError(f"pack schema must be {PACK_SCHEMA}")
        items = _decode_items(payload["items"], "items")
        digest = _require_digest(payload["digest"], "digest")
        if digest != _pack_digest(items):
            raise ContextPackError("pack digest does not match its items")
        declared = payload["byte_count"]
        if not isinstance(declared, int) or isinstance(declared, bool):
            raise ContextPackError("byte_count must be an integer")
        if declared != len(bytes(raw)):  # type: ignore[arg-type]
            raise ContextPackError("byte_count does not match the serialized pack length")
        return cls(items=items, digest=digest, byte_count=declared)


@dataclass(frozen=True)
class ContextPackDelta:
    """One prefix-reusing delta that rebuilds a target pack from a base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    def __post_init__(self) -> None:
        _require_digest(self.base_digest, "base_digest")
        _require_digest(self.target_digest, "target_digest")
        if not isinstance(self.prefix_items, int) or isinstance(self.prefix_items, bool):
            raise ContextPackError("prefix_items must be an integer")
        if self.prefix_items < 0:
            raise ContextPackError("prefix_items must not be negative")
        if not isinstance(self.appended, tuple) or not all(
            isinstance(item, (bytes, bytearray)) for item in self.appended
        ):
            raise ContextPackError("appended must be a tuple of byte chunks")
        if any(not isinstance(item, bytes) for item in self.appended):
            object.__setattr__(self, "appended", tuple(bytes(item) for item in self.appended))

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Keep the longest shared prefix of ``base`` and append the target suffix."""

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("delta endpoints must be context packs")
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Actual serialized size of this delta on the wire."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta exactly as sent on the wire."""

        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, raw: object) -> ContextPackDelta:
        """Validate and decode one delta from the wire."""

        payload = _loads(raw, _DELTA_FIELDS)
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError(f"delta schema must be {DELTA_SCHEMA}")
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool):
            raise ContextPackError("prefix_items must be an integer")
        if prefix < 0:
            raise ContextPackError("prefix_items must not be negative")
        return cls(
            base_digest=_require_digest(payload["base_digest"], "base_digest"),
            target_digest=_require_digest(payload["target_digest"], "target_digest"),
            prefix_items=prefix,
            appended=_decode_items(payload["appended"], "appended"),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from one base pack and one received delta."""

    if not isinstance(base, ContextPack):
        raise ContextPackError("delta base must be a context pack")
    if not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta must be a context pack delta")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the available base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the available base pack chunks")
    rebuilt = (*base.items[: delta.prefix_items], *delta.appended)
    target = ContextPack.build(rebuilt)
    if target.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target digest")
    return target
