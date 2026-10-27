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

"""Dependency-driven availability for Sources, Memory entries and Topic Memory.

Every Scope owns one evidence DAG. A Source carries an explicit availability
event stream; derived nodes recompute availability from their parents under an
``all`` or ``any`` policy. Availability is propagated into the native Memory
entry state and into Topic Memory visibility, while every registered reference
stays readable for audit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, RetireMemoryEntryRequest
from powercontext.limits import MAX_SCOPE_ID_LENGTH

__all__ = [
    "LifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "SourceLifecycleService",
]

_POLICIES = ("all", "any")
_KINDS = ("source", "memory", "topic")
_PURPOSES = ("generation", "audit")


class SourceLifecycleError(ValueError):
    """Report an invalid lifecycle registration, event or resolution."""


class SourceLifecycleEvent(BaseModel):
    """One availability event for a registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("must be a non-blank trimmed string")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence", mode="before")
    @classmethod
    def validate_positive(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("must be a positive integer")  # noqa: TRY003
        return value

    @field_validator("available", mode="before")
    @classmethod
    def validate_available(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("available must be a boolean")  # noqa: TRY003
        return value

    @model_validator(mode="after")
    def validate_reason_length(self) -> SourceLifecycleEvent:
        if len(self.reason) > 512:
            raise ValueError("reason is too long")  # noqa: TRY003
        return self


class _Node(BaseModel):
    """One registered node of the evidence DAG."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["source", "memory", "topic"]
    order: int
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] | None = None
    revision: int = 0
    sequence: int = 0
    available: bool = True
    source: dict[str, Any] | None = None
    citation: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None


class SourceLifecycleRepository:
    """Persist one Scope's evidence DAG rows in a caller-owned transaction."""

    async def load(self, connection: AsyncConnection, scope_id: str) -> dict[str, _Node]:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        rows = (
            await connection.execute(select(table.c.node_id, table.c.payload).where(table.c.scope_id == scope_id))
        ).all()
        nodes: dict[str, _Node] = {}
        for node_id, payload in rows:
            data = json.loads(payload)
            data["parents"] = tuple(data.get("parents") or ())
            nodes[str(node_id)] = _Node.model_validate(data)
        return nodes

    async def save(self, connection: AsyncConnection, scope_id: str, node_id: str, node: _Node) -> None:
        table = SOURCE_LIFECYCLE_NODES_TABLE
        payload = json.dumps(node.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        await connection.execute(delete(table).where(table.c.scope_id == scope_id, table.c.node_id == node_id))
        await connection.execute(
            table.insert().values(scope_id=scope_id, node_id=node_id, payload=payload)
        )


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip() or node_id != node_id.strip():
        raise SourceLifecycleError("node_id must be a non-blank trimmed string")
    if len(node_id) > 128:
        raise SourceLifecycleError("node_id is too long")
    return node_id


def _require_policy(policy: object) -> str:
    if policy not in _POLICIES:
        raise SourceLifecycleError("policy must be all or any")
    return str(policy)


def active_nodes(nodes: Mapping[str, _Node]) -> frozenset[str]:
    """Propagate Source availability through one acyclic registration order."""

    active = {node_id for node_id, node in nodes.items() if node.kind == "source" and node.available}
    for node_id in sorted(nodes, key=lambda item: nodes[item].order):
        node = nodes[node_id]
        if node.kind == "source":
            continue
        support = [parent in active for parent in node.parents]
        if not support:
            continue
        if (all(support) if node.policy == "all" else any(support)):
            active.add(node_id)
    return frozenset(active)


class SourceLifecycleService:
    """Register nodes and apply availability events for one Scope."""

    def __init__(self, runtime: Any, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._repository = SourceLifecycleRepository()

    async def register_source(self, node_id: str, source_ref: Any, *, revision: int) -> None:
        node_id = _require_node_id(node_id)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")
        await self._runtime.records.for_scope(self.scope_id).get_source(
            source_ref.source_type, source_ref.source_id
        )
        candidate = _Node(
            kind="source",
            order=0,
            revision=revision,
            sequence=0,
            available=True,
            source=source_ref.model_dump(mode="json"),
        )
        await self._register(node_id, candidate, binding=source_ref.model_dump(mode="json"))

    async def register_memory(
        self, node_id: str, citation: Any, *, parents: Sequence[str], policy: str = "all"
    ) -> None:
        node_id = _require_node_id(node_id)
        policy = _require_policy(policy)
        await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=citation))
        payload = citation.model_dump(mode="json")
        candidate = _Node(
            kind="memory",
            order=0,
            parents=tuple(parents),
            policy=policy,
            citation=payload,
        )
        await self._register(node_id, candidate, binding={**payload, "parents": list(parents), "policy": policy})

    async def register_topic(
        self, node_id: str, artifact_ref: Any, *, parents: Sequence[str], policy: str = "all"
    ) -> None:
        node_id = _require_node_id(node_id)
        policy = _require_policy(policy)
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision
        )
        payload = artifact_ref.model_dump(mode="json")
        candidate = _Node(
            kind="topic",
            order=0,
            parents=tuple(parents),
            policy=policy,
            artifact=payload,
        )
        await self._register(node_id, candidate, binding={**payload, "parents": list(parents), "policy": policy})

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event must be a SourceLifecycleEvent")
        async with (
            self._runtime._scoped_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
            self._runtime._database.transaction() as connection,
        ):
            nodes = await self._repository.load(connection, self.scope_id)
            node = nodes.get(event.node_id)
            if node is None or node.kind != "source":
                raise SourceLifecycleError("event node is not a registered Source")
            if event.revision != node.revision:
                raise SourceLifecycleError("event revision does not match the registered Source revision")
            replayed = event.sequence == node.sequence and event.available == node.available
            if event.sequence < node.sequence or (
                event.sequence == node.sequence and event.available != node.available
            ):
                raise SourceLifecycleError("event sequence must increase monotonically per Source")
            if not replayed:
                updated = node.model_copy(update={"sequence": event.sequence, "available": event.available})
                nodes[event.node_id] = updated
                await self._repository.save(connection, self.scope_id, event.node_id, updated)
            active = active_nodes(nodes)
        self._runtime.lifecycle.note_suppressed(self.scope_id, nodes, active)
        await self._reconcile(nodes, active, reason=event.reason)
        return sorted(active)

    async def resolve(self, node_id: str, *, purpose: str = "generation") -> Any:
        node_id = _require_node_id(node_id)
        if purpose not in _PURPOSES:
            raise SourceLifecycleError("unknown resolve purpose")
        nodes = await self._load()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "generation" and node_id not in active_nodes(nodes):
            raise SourceLifecycleError("node is not available for generation")
        return self._reference(node)

    async def active(self) -> list[str]:
        return sorted(active_nodes(await self._load()))

    async def reconcile(self, reason: str = "lifecycle") -> None:
        nodes = await self._load()
        active = active_nodes(nodes)
        self._runtime.lifecycle.note_suppressed(self.scope_id, nodes, active)
        await self._reconcile(nodes, active, reason=reason)

    async def _register(self, node_id: str, candidate: _Node, *, binding: dict[str, Any]) -> None:
        async with (
            self._runtime._scoped_operation(self.scope_id),
            self._runtime._locked(self.scope_id),
            self._runtime._database.transaction() as connection,
        ):
            nodes = await self._repository.load(connection, self.scope_id)
            existing = nodes.get(node_id)
            if existing is not None:
                if self._binding(existing) == binding:
                    return
                raise SourceLifecycleError("node_id is already bound to a different registration")
            if candidate.kind != "source":
                self._validate_parents(node_id, candidate.parents, nodes)
            ordered = candidate.model_copy(update={"order": len(nodes)})
            nodes[node_id] = ordered
            await self._repository.save(connection, self.scope_id, node_id, ordered)
            active = active_nodes(nodes)
        self._runtime.lifecycle.note_suppressed(self.scope_id, nodes, active)
        await self._reconcile(nodes, active, reason=f"register:{node_id}")

    @staticmethod
    def _binding(node: _Node) -> dict[str, Any]:
        if node.kind == "source":
            return dict(node.source or {})
        payload = node.citation if node.kind == "memory" else node.artifact
        return {**(payload or {}), "parents": list(node.parents), "policy": node.policy}

    @staticmethod
    def _validate_parents(node_id: str, parents: Sequence[str], nodes: Mapping[str, _Node]) -> None:
        if not parents:
            raise SourceLifecycleError("derived nodes require at least one parent")
        seen: set[str] = set()
        for parent in parents:
            if parent == node_id:
                raise SourceLifecycleError("a node cannot depend on itself")
            if parent in seen:
                raise SourceLifecycleError("parents must be distinct")
            seen.add(parent)
            if parent not in nodes:
                raise SourceLifecycleError("parent must be registered first")
        # A new node has no descendants yet, so a cycle can only start with a
        # parent chain that already reaches this identity.
        for parent in parents:
            stack = [parent]
            visited: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current == node_id:
                    raise SourceLifecycleError("parent chain would create a cycle")
                stack.extend(nodes[current].parents if current in nodes else ())

    @staticmethod
    def _reference(node: _Node) -> Any:
        if node.kind == "source":
            from powercontext.sources.models import SourceRef

            return SourceRef.model_validate(node.source)
        if node.kind == "memory":
            from powercontext.builtin.artifacts.memory.models import MemoryCitation

            return MemoryCitation.model_validate(node.citation)
        return ArtifactRef.model_validate(node.artifact)

    async def _load(self) -> dict[str, _Node]:
        async with self._runtime._database.transaction() as connection:
            return await self._repository.load(connection, self.scope_id)

    async def _reconcile(self, nodes: Mapping[str, _Node], active: frozenset[str], *, reason: str) -> None:
        memory = self._runtime.memory.for_scope(self.scope_id)
        targets = {
            node_id: node for node_id, node in nodes.items() if node.kind == "memory" and node.citation
        }
        if not targets:
            return
        desired = {node.citation["entry_id"]: node_id in active for node_id, node in targets.items()}
        for _ in range(len(targets) + 2):
            page = await memory.list(include_inactive=True)
            current = {record.entry.entry_id: record for record in page.entries}
            pending = None
            for entry_id, want in desired.items():
                record = current.get(entry_id)
                if record is None:
                    continue
                if (record.state == "active") != want:
                    pending = (record, want)
                    break
            if pending is None:
                return
            record, want = pending
            request = RetireMemoryEntryRequest(citation=record.citation, reason=reason)
            if want:
                await memory.restore(request)
            else:
                await memory.retire(request)


class LifecycleApplication:
    """Select the scoped source lifecycle service."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._suppressed: dict[str, frozenset[tuple[str, int]]] = {}

    def for_scope(self, scope_id: str, /) -> SourceLifecycleService:
        if not isinstance(scope_id, str) or not scope_id.strip() or len(scope_id) > MAX_SCOPE_ID_LENGTH:
            raise SourceLifecycleError("scope_id must be a non-blank identifier")
        return SourceLifecycleService(self._runtime, scope_id)

    def note_suppressed(self, scope_id: str, nodes: Mapping[str, _Node], active: frozenset[str]) -> None:
        suppressed = set()
        for node_id, node in nodes.items():
            if node.kind != "topic" or node_id in active or not node.artifact:
                continue
            suppressed.add((node.artifact["artifact_id"], int(node.artifact["revision"])))
        self._suppressed[scope_id] = frozenset(suppressed)

    async def suppressed_topics(self, scope_id: str) -> frozenset[tuple[str, int]]:
        """Read Topic visibility without mutating, so search paths never block."""

        cached = self._suppressed.get(scope_id)
        if cached is not None:
            return cached
        nodes: dict[str, _Node] = {}
        try:
            async with self._runtime._database.transaction() as connection:
                nodes = await SourceLifecycleRepository().load(connection, scope_id)
        except Exception:  # noqa: BLE001 - visibility read is best effort
            nodes = {}
        self.note_suppressed(scope_id, nodes, active_nodes(nodes))
        return self._suppressed.get(scope_id, frozenset())
