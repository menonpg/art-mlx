from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
PlanT = TypeVar("PlanT")
EstimateT = TypeVar("EstimateT")


@dataclass(frozen=True)
class _MemoryCheck:
    estimated_required_bytes: int
    available_bytes: int
    fits: bool


@dataclass(frozen=True)
class _CandidateMicroBatch(Generic[InputT, PlanT]):
    inputs: Sequence[InputT]
    indices: tuple[int, ...]
    plan: PlanT
    check: _MemoryCheck
    stats_global_count: int
    rejected_candidates: int
    cold_start: bool


def select_next_micro_batch(
    items: Sequence[InputT],
    start: int,
    *,
    dp_rank: int,
    dp_size: int,
    previous_global_micro_batch_size: int | None,
    plan_for_local_inputs: Callable[[tuple[int, ...], Sequence[InputT]], PlanT],
    estimate_for_local_inputs: Callable[
        [tuple[int, ...], Sequence[InputT]], EstimateT | None
    ],
    memory_check: Callable[[PlanT], _MemoryCheck],
    memory_check_estimate: Callable[[EstimateT], _MemoryCheck],
    has_memory_profile: Callable[[PlanT], bool],
    has_memory_profile_estimate: Callable[[EstimateT], bool],
    raise_smallest_batch_error: Callable[[PlanT, _MemoryCheck], None],
) -> _CandidateMicroBatch[InputT, PlanT]:
    remaining = len(items) - start
    min_width = min(dp_size, remaining)
    if min_width <= 0:
        raise RuntimeError("cannot select an empty microbatch window")

    cache: dict[int, _CandidateMicroBatch[InputT, PlanT]] = {}
    estimate_cache: dict[int, tuple[EstimateT, _MemoryCheck] | None] = {}
    rejected = 0

    def clamp_width(width: int) -> int:
        return max(min_width, min(width, remaining))

    def local_slice(width: int) -> tuple[tuple[int, ...], list[InputT]]:
        stop = start + clamp_width(width)
        indices = tuple(range(start + dp_rank, stop, dp_size))
        return indices, [items[index] for index in indices]

    def candidate(
        width: int,
        estimated_check: tuple[EstimateT, _MemoryCheck] | None = None,
    ) -> _CandidateMicroBatch[InputT, PlanT]:
        width = clamp_width(width)
        cached = cache.get(width)
        if cached is not None:
            return cached
        indices, local_inputs = local_slice(width)
        plan = plan_for_local_inputs(indices, local_inputs)
        check = (
            estimated_check[1] if estimated_check is not None else memory_check(plan)
        )
        item = _CandidateMicroBatch(
            inputs=local_inputs,
            indices=indices,
            plan=plan,
            check=check,
            stats_global_count=width,
            rejected_candidates=rejected,
            cold_start=not has_memory_profile(plan),
        )
        cache[width] = item
        return item

    def estimate_check(width: int) -> tuple[EstimateT, _MemoryCheck] | None:
        width = clamp_width(width)
        if width in estimate_cache:
            return estimate_cache[width]
        indices, local_inputs = local_slice(width)
        estimate = estimate_for_local_inputs(indices, local_inputs)
        if estimate is None:
            estimate_cache[width] = None
            return None
        estimate_cache[width] = estimate, memory_check_estimate(estimate)
        return estimate_cache[width]

    def probe(
        width: int,
    ) -> tuple[
        bool,
        tuple[EstimateT, _MemoryCheck] | None,
        _CandidateMicroBatch[InputT, PlanT] | None,
    ]:
        estimated = estimate_check(width)
        if estimated is not None:
            return estimated[1].fits, estimated, None
        item = candidate(width)
        return item.check.fits, None, item

    first_estimated = estimate_check(min_width)
    if first_estimated is not None and not first_estimated[1].fits:
        first = candidate(min_width, first_estimated)
        raise_smallest_batch_error(first.plan, first.check)

    if first_estimated is not None and has_memory_profile_estimate(first_estimated[0]):
        best_width = min_width
        best_estimated: tuple[EstimateT, _MemoryCheck] | None = first_estimated
        best_item: _CandidateMicroBatch[InputT, PlanT] | None = None
    else:
        first = candidate(min_width, first_estimated)
        if not first.check.fits:
            raise_smallest_batch_error(first.plan, first.check)
        if first.cold_start:
            return first
        best_width = first.stats_global_count
        best_estimated = None
        best_item = first

    def remember_fit(
        width: int,
        estimated: tuple[EstimateT, _MemoryCheck] | None,
        item: _CandidateMicroBatch[InputT, PlanT] | None,
    ) -> None:
        nonlocal best_width, best_estimated, best_item
        best_width = clamp_width(width)
        best_estimated = estimated
        best_item = item

    high_fail: int | None = None
    width = min(
        remaining,
        max(min_width, (previous_global_micro_batch_size or min_width) * 2),
    )
    while width <= remaining:
        fits, estimated, item = probe(width)
        if fits:
            remember_fit(width, estimated, item)
            if width == remaining:
                break
            width = min(remaining, max(width + 1, width * 2))
            continue
        rejected += 1
        high_fail = width
        break

    def finalize_best() -> _CandidateMicroBatch[InputT, PlanT]:
        selected = best_item or candidate(best_width, best_estimated)
        return _CandidateMicroBatch(
            inputs=selected.inputs,
            indices=selected.indices,
            plan=selected.plan,
            check=selected.check,
            stats_global_count=selected.stats_global_count,
            rejected_candidates=rejected,
            cold_start=selected.cold_start,
        )

    if high_fail is None:
        return finalize_best()

    low = best_width + 1
    high = high_fail - 1
    while low <= high:
        mid = (low + high) // 2
        fits, estimated, item = probe(mid)
        if fits:
            remember_fit(mid, estimated, item)
            low = mid + 1
        else:
            rejected += 1
            high = mid - 1

    return finalize_best()
