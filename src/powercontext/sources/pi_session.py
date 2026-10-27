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

"""Pi session branch capture with immutable per-event Sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]

_UTF8_BOM = b"\xef\xbb\xbf"
_IDENTITY_PREFIX = "pi:"
_SESSION_TYPE = "session"
_SESSION_VERSION = 3
_VISIBLE_ROLES = frozenset({"user", "assistant", "toolResult", "bashExecution"})


class PiSessionInput(BaseModel):
    """One complete Pi v3 session archive and the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected branch, with its original record."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A captured Pi event whose identity is the session and event pair."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the fixed event identity with percent-encoded components."""

    for label, value in (("session_id", session_id), ("event_id", event_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
        if value != value.strip():
            raise ValueError(f"{label} must not contain leading or trailing whitespace")
    identity = f"{_IDENTITY_PREFIX}{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        # Keep the identity bounded while still deriving it from the same pair.
        digest = hashlib.sha256(f"{session_id}\x1f{event_id}".encode()).hexdigest()
        identity = f"{_IDENTITY_PREFIX}sha256:{digest}"
    return identity


def _decode(content: bytes) -> str:
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Pi session content must be bytes")
    raw = bytes(content)
    if raw.startswith(_UTF8_BOM):
        raw = raw[len(_UTF8_BOM) :]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Pi session archive must be valid UTF-8: {error}") from error


def _load_records(content: bytes) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
    """Decode one archive into its header and the remaining ordered records."""

    records: list[dict[str, JsonValue]] = []
    for number, line in enumerate(_decode(content).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session record {number} is not valid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Pi session record {number} must be a JSON object")
        records.append(value)
    if not records:
        raise ValueError("Pi session archive must contain at least the session header")
    header, body = records[0], records[1:]
    if header.get("type") != _SESSION_TYPE:
        raise ValueError("the first Pi session record must be the session header")
    if header.get("version") != _SESSION_VERSION:
        raise ValueError(f"Pi session version must be {_SESSION_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must declare a non-empty id")
    return header, body


def _graph(
    body: list[dict[str, JsonValue]],
) -> tuple[dict[str, dict[str, JsonValue]], list[str]]:
    """Validate the parent graph and return nodes plus their arrival order."""

    nodes: dict[str, dict[str, JsonValue]] = {}
    order: list[str] = []
    for record in body:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("every Pi session record must declare a non-empty string id")
        if event_id in nodes:
            raise ValueError(f"duplicate Pi session record id: {event_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError(f"record {event_id!r} must declare parentId as null or a non-empty string")
            if parent_id not in nodes:
                raise ValueError(f"record {event_id!r} points at a parent that has not appeared: {parent_id!r}")
        nodes[event_id] = record
        order.append(event_id)
    return nodes, order


def _branch(nodes: dict[str, dict[str, JsonValue]], order: list[str], leaf_id: str | None) -> list[str]:
    """Resolve the root-to-leaf ancestor chain for one selected leaf."""

    leaf = leaf_id if leaf_id is not None else (order[-1] if order else None)
    if leaf is None:
        raise ValueError("Pi session archive does not contain any selectable node")
    if leaf not in nodes:
        raise ValueError(f"selected leaf is not present in the archive: {leaf!r}")
    chain: list[str] = []
    seen: set[str] = set()
    cursor: str | None = leaf
    while cursor is not None:
        if cursor in seen:
            raise ValueError(f"Pi session parent graph contains a cycle at {cursor!r}")
        seen.add(cursor)
        chain.append(cursor)
        parent = nodes[cursor].get("parentId")
        cursor = parent if isinstance(parent, str) else None
    chain.reverse()
    return chain


def _compact_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _text_blocks(content: object) -> list[str]:
    """Collect the visible text carried by one message body."""

    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return parts


def _message_body(message: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Derive the visible role and body of one message record."""

    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    content = message.get("content")
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        name = tool_name if isinstance(tool_name, str) and tool_name else "unknown"
        call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else "unknown"
        text = "\n".join(part for part in _text_blocks(content) if part)
        return role, f"[tool_result {name} {call_id}] {text}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        command_text = command if isinstance(command, str) else ""
        output_text = output if isinstance(output, str) else ""
        return role, f"$ {command_text}\n{output_text}"
    if role not in _VISIBLE_ROLES:
        return None
    parts: list[str] = [
        block
        for block in (
            _block_body(item) for item in (content if isinstance(content, list) else [content])
        )
        if block
    ]
    return role, "\n".join(parts)


def _block_body(block: object) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if kind == "toolCall":
        name = block.get("name")
        call_id = block.get("id")
        arguments = block.get("arguments")
        tool_name = name if isinstance(name, str) and name else "unknown"
        tool_call_id = call_id if isinstance(call_id, str) and call_id else "unknown"
        if arguments is None:
            arguments = {}
        return f"[tool_call {tool_name} {tool_call_id}] {_compact_json(arguments)}"
    return ""


def _record_body(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Derive the visible role and body of one non-header record."""

    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        return _message_body(message)
    if event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary:
            return None
        return str(event_type), summary
    if event_type == "custom_message":
        parts = [part for part in _text_blocks(record.get("content")) if part]
        if not parts:
            return None
        return "custom", "\n".join(parts)
    # Every other node keeps its place in the graph but carries no evidence.
    return None


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse the selected branch into its ordered visible event inputs."""

    header, body = _load_records(value.content)
    session_id = header["id"]
    if not isinstance(session_id, str):
        raise ValueError("Pi session header must declare a non-empty id")
    nodes, order = _graph(body)
    events: list[PiEventInput] = []
    for event_id in _branch(nodes, order, value.leaf_id):
        record = nodes[event_id]
        derived = _record_body(record)
        if derived is None:
            continue
        role, body_text = derived
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=record.get("parentId") if isinstance(record.get("parentId"), str) else None,
                event_type=str(record.get("type")),
                role=role,
                content=f"{role}: {body_text}",
                record=record,
            )
        )
    return tuple(events)


def record_digest(record: dict[str, JsonValue]) -> str:
    """Return the canonical digest used to detect rewritten history."""

    return hashlib.sha256(_compact_json(record).encode()).hexdigest()


class _PiEventAdapter:
    """Resolve one parsed event into an immutable Source and read it back."""

    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class _PiEventTextEvidenceProjection:
    """Standard powercontext.text-evidence projection for one Pi event."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> object:
        return {
            "source_type": "pi-event",
            "source_id": source.name,
            "content": source.content,
            "metadata": {
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        }


PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    adapter=_PiEventAdapter(),
    version="1",
    projections=(_PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Replay one Pi session file and submit only unseen branch events."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        events = parse_pi_session(
            PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id)
        )
        progress = _imported(session.checkpoint)
        pending: list[PiEventInput] = []
        for event in events:
            digest = record_digest(event.record)
            seen = progress.get(event.event_id)
            if seen is not None:
                if seen != digest:
                    raise ValueError(
                        f"Pi event {event.event_id!r} was already imported with different content"
                    )
                continue
            pending.append(event)
        for event in pending:
            await session.submit(event.event_id, "pi-event", event)
            progress[event.event_id] = record_digest(event.record)
        checkpoint: dict[str, Any] | None = None if session.checkpoint is None and not progress else {
            "events": {event_id: progress[event_id] for event_id in sorted(progress)}
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _imported(checkpoint: JsonValue) -> dict[str, str]:
    """Read the durable per-event digests from an opaque checkpoint."""

    if not isinstance(checkpoint, dict):
        return {}
    raw = checkpoint.get("events")
    if not isinstance(raw, dict):
        return {}
    progress: dict[str, str] = {}
    for event_id, digest in raw.items():
        if isinstance(event_id, str) and isinstance(digest, str):
            progress[event_id] = digest
    return progress
