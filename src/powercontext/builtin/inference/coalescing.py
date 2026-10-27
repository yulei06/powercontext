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

"""Cross-call Embedding coalescing with a bounded successful-vector cache.

One wrapper owns the pending window, the in-flight text set, and the LRU. Each
caller keeps its own original order, duplicate positions, and usage; the shared
provider request is owned by the wrapper so cancelling one caller cannot cancel
work another caller still waits on.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from powercontext.builtin.artifacts.memory import EmbeddingProfile
    from powercontext.builtin.inference.models import EmbeddingVector
    from powercontext.builtin.inference.protocols import EmbeddingModel

__all__ = ["CoalescingEmbeddingModel"]


class _Batch:
    """One underlying provider request shared by every caller that needs it."""

    __slots__ = ("texts",)

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts


class CoalescingEmbeddingModel:
    """Share identical Embedding work across concurrent callers and calls."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if _invalid_window(coalesce_ms):
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._waiting: dict[str, asyncio.Future[tuple[EmbeddingVector, object, InferenceUsage | None]]] = {}
        self._dispatched: set[str] = set()
        self._window: asyncio.Task[None] | None = None

    @property
    def profile(self) -> EmbeddingProfile:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        inputs = tuple(texts)
        if not inputs:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        wanted: dict[str, None] = {}
        for text in inputs:
            wanted.setdefault(text, None)

        settled: dict[str, tuple[EmbeddingVector, object, InferenceUsage | None]] = {}
        pending: dict[str, asyncio.Future[tuple[EmbeddingVector, object, InferenceUsage | None]]] = {}
        for text in wanted:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                settled[text] = (cached, None, None)
                continue
            future = self._waiting.get(text)
            if future is None:
                future = _new_future()
                self._waiting[text] = future
            pending[text] = future

        if pending:
            self._schedule_window()
            # Shielding keeps one caller's cancellation from cancelling the
            # shared request another caller still depends on.
            results = await asyncio.gather(*(asyncio.shield(future) for future in pending.values()))
            settled.update(zip(pending, results, strict=True))

        usage = _sum_usage(settled, wanted)
        ordered = tuple(settled[text][0] for text in inputs)
        return EmbeddingResult(vectors=ordered, usage=usage)

    def _schedule_window(self) -> None:
        if self._window is not None and not self._window.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - embed is always awaited in a loop
            return
        self._window = loop.create_task(self._flush())

    async def _flush(self) -> None:
        await asyncio.sleep(self._coalesce)
        free = [text for text in self._waiting if text not in self._dispatched]
        if not free:
            return
        batches = [free[index : index + self._max_batch_size] for index in range(0, len(free), self._max_batch_size)]
        tasks = [asyncio.create_task(self._run(tuple(batch))) for batch in batches]
        for texts in batches:
            self._dispatched.update(texts)
        # Batches are independent: one failure must not stop the others.
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, texts: tuple[str, ...]) -> None:
        try:
            result = await self._model.embed(texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(texts):
                raise ValueError("embedding vector count does not match the batch size")  # noqa: TRY003
        except Exception as error:
            self._finish(texts, error=error)
            raise
        self._finish(texts, vectors=vectors, usage=result.usage)

    def _finish(
        self,
        texts: tuple[str, ...],
        *,
        vectors: tuple[EmbeddingVector, ...] | None = None,
        usage: InferenceUsage | None = None,
        error: BaseException | None = None,
    ) -> None:
        batch = _Batch(texts)
        for index, text in enumerate(texts):
            self._dispatched.discard(text)
            future = self._waiting.pop(text, None)
            if error is not None:
                if future is not None and not future.done():
                    future.set_exception(error)
                continue
            assert vectors is not None
            vector = vectors[index]
            if self._cache_size:
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            if future is not None and not future.done():
                future.set_result((vector, batch, usage))


def _invalid_window(value: object) -> bool:
    """Report a value that cannot be a coalescing window duration."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return True
    return value != value or value < 0  # noqa: PLR0124 - explicit NaN rejection


def _new_future() -> asyncio.Future[tuple[EmbeddingVector, object, InferenceUsage | None]]:
    return asyncio.get_running_loop().create_future()


def _sum_usage(
    settled: dict[str, tuple[EmbeddingVector, object, InferenceUsage | None]],
    wanted: Iterable[str],
) -> InferenceUsage:
    """Charge only the distinct underlying batches this call actually waited on."""

    seen: dict[int, InferenceUsage] = {}
    for text in wanted:
        _, batch, usage = settled[text]
        if batch is None or usage is None:
            continue
        seen.setdefault(id(batch), usage)
    if not seen:
        return InferenceUsage(requests=0)
    tokens: int | None = 0
    outputs: int | None = 0
    for usage in seen.values():
        if usage.input_tokens is None or tokens is None:
            tokens = None
        else:
            tokens += usage.input_tokens
        if usage.output_tokens is None or outputs is None:
            outputs = None
        else:
            outputs += usage.output_tokens
    return InferenceUsage(requests=len(seen), input_tokens=tokens, output_tokens=outputs)
