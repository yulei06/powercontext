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

"""HTTP cross-encoder reranking with request batching and global selection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

if TYPE_CHECKING:
    # httpx is only ever injected by the caller, so this module imports without it.
    import httpx

__all__ = ["HTTPMemoryReranker"]

DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class _ScoreItem(BaseModel):
    """One service score addressed by its global candidate rank."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: StrictInt = Field(ge=1)
    score: float = Field(allow_inf_nan=False)


class _ScoreResponse(BaseModel):
    """The exact response envelope of one batched scoring request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: list[_ScoreItem]


class HTTPMemoryReranker:
    """Score coarse Memory candidates through one HTTP reranking service."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty URL")  # noqa: TRY003
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("rerank batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("rerank policy_id must be a non-empty string")  # noqa: TRY003
        if client is None:
            raise ValueError("rerank requires an httpx.AsyncClient")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select the globally best ranks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("rerank requires a non-empty query")  # noqa: TRY003
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scored: list[tuple[float, int]] = []
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            stop = min(start + self._batch_size, len(candidates))
            ranks = list(range(start + 1, stop + 1))
            scored.extend(await self._score_batch(query, candidates, ranks))
            requests += 1

        # Global selection: highest score first, ties broken by the original coarse rank.
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = tuple(rank for _score, rank in scored[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
            discarded_rank_count=len(candidates) - len(selected),
        )

    async def _score_batch(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        ranks: list[int],
    ) -> list[tuple[float, int]]:
        payload: dict[str, Any] = {
            "query": query,
            "candidates": [
                {"rank": rank, "text": candidates[rank - 1].text} for rank in ranks
            ],
        }
        response = await self._client.post(
            self._endpoint,
            content=_encode(payload),
            headers={"content-type": "application/json"},
        )
        response.raise_for_status()
        return _parse_scores(response, ranks)


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse_scores(response: httpx.Response, ranks: list[int]) -> list[tuple[float, int]]:
    try:
        raw = json.loads(response.content)
    except (TypeError, ValueError) as error:
        raise ValueError(f"rerank response is not valid JSON: {error}") from error
    try:
        parsed = _ScoreResponse.model_validate(raw)
    except ValueError as error:
        raise ValueError(f"rerank response does not match the score schema: {error}") from error

    expected = set(ranks)
    observed: set[int] = set()
    scores: list[tuple[float, int]] = []
    for item in parsed.scores:
        if item.rank not in expected:
            raise ValueError(f"rerank response reported a rank outside its batch: {item.rank}")  # noqa: TRY003
        if item.rank in observed:
            raise ValueError(f"rerank response repeated rank {item.rank}")  # noqa: TRY003
        observed.add(item.rank)
        scores.append((float(item.score), item.rank))
    missing = expected - observed
    if missing:
        raise ValueError(f"rerank response omitted ranks {sorted(missing)}")  # noqa: TRY003
    return scores
