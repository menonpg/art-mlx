import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator

_pinned_inference_steps: ContextVar[dict[str, int]] = ContextVar(
    "art_pinned_inference_steps",
    default={},
)


def pinned_inference_step(model_name: str) -> int | None:
    return _pinned_inference_steps.get().get(model_name)


@asynccontextmanager
async def pin_inference_step(
    model_name: str,
    step: int,
) -> AsyncIterator[None]:
    pinned_steps = dict(_pinned_inference_steps.get())
    pinned_steps[model_name] = step
    token = _pinned_inference_steps.set(pinned_steps)
    try:
        yield
    finally:
        _pinned_inference_steps.reset(token)


class AdapterLeaseManager:
    def __init__(self) -> None:
        self._counts: Counter[int] = Counter()
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, step: int) -> AsyncIterator[None]:
        async with self._condition:
            self._counts[step] += 1
        try:
            yield
        finally:
            async with self._condition:
                self._counts[step] -= 1
                if self._counts[step] <= 0:
                    del self._counts[step]
                self._condition.notify_all()

    def active_steps(self) -> set[int]:
        return set(self._counts)
