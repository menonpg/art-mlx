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


@dataclass(frozen=True)
class _EstimatedMemoryCheck(Generic[EstimateT]):
    estimate: EstimateT
    check: _MemoryCheck


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
    estimate_matches_plan: Callable[[EstimateT, PlanT], bool],
    has_memory_profile: Callable[[PlanT], bool],
    has_memory_profile_estimate: Callable[[EstimateT], bool],
    raise_smallest_batch_error: Callable[[PlanT, _MemoryCheck], None],
) -> _CandidateMicroBatch[InputT, PlanT]:
    remaining = len(items) - start
    min_width = min(dp_size, remaining)
    if min_width <= 0:
        raise RuntimeError("cannot select an empty microbatch window")

    cache: dict[int, _CandidateMicroBatch[InputT, PlanT]] = {}
    rejected = 0

    def clamp_width(width: int) -> int:
        return max(min_width, min(width, remaining))

    def local_slice(width: int) -> tuple[tuple[int, ...], list[InputT]]:
        stop = start + clamp_width(width)
        indices = tuple(range(start + dp_rank, stop, dp_size))
        return indices, [items[index] for index in indices]

    def candidate(
        width: int,
        estimated_check: _EstimatedMemoryCheck[EstimateT] | None = None,
    ) -> _CandidateMicroBatch[InputT, PlanT]:
        width = clamp_width(width)
        cached = cache.get(width)
        if cached is not None:
            return cached
        indices, local_inputs = local_slice(width)
        plan = plan_for_local_inputs(indices, local_inputs)
        check = (
            estimated_check.check
            if estimated_check is not None
            and estimate_matches_plan(estimated_check.estimate, plan)
            else memory_check(plan)
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

    def estimate_check(width: int) -> _EstimatedMemoryCheck[EstimateT] | None:
        indices, local_inputs = local_slice(width)
        estimate = estimate_for_local_inputs(indices, local_inputs)
        if estimate is None:
            return None
        return _EstimatedMemoryCheck(
            estimate=estimate,
            check=memory_check_estimate(estimate),
        )

    first_estimated_check = estimate_check(min_width)
    if first_estimated_check is not None:
        if not first_estimated_check.check.fits:
            first = candidate(min_width, first_estimated_check)
            raise_smallest_batch_error(first.plan, first.check)
        if has_memory_profile_estimate(first_estimated_check.estimate):
            best: _CandidateMicroBatch[InputT, PlanT] | None = None
            best_estimated_check: _EstimatedMemoryCheck[EstimateT] | None = (
                first_estimated_check
            )
            best_width = min_width
        else:
            first = candidate(min_width, first_estimated_check)
            if first.cold_start:
                return first
            best = first
            best_estimated_check = None
            best_width = first.stats_global_count
    else:
        first = candidate(min_width)
        if not first.check.fits:
            raise_smallest_batch_error(first.plan, first.check)
        if first.cold_start:
            return first
        best = first
        best_estimated_check = None
        best_width = first.stats_global_count

    high_fail: int | None = None
    width = min(
        remaining,
        max(min_width, (previous_global_micro_batch_size or min_width) * 2),
    )
    while width <= remaining:
        check = estimate_check(width)
        if check is not None and not check.check.fits:
            rejected += 1
            high_fail = width
            break
        if check is not None:
            best_width = width
            best_estimated_check = check
            best = None
            if width == remaining:
                break
            width = min(remaining, max(width + 1, width * 2))
            continue
        item = candidate(width, check)
        if item.check.fits:
            best = item
            best_width = width
            best_estimated_check = None
            if width == remaining:
                break
            width = min(remaining, max(width + 1, width * 2))
            continue
        rejected += 1
        high_fail = width
        break

    def finalize_best() -> _CandidateMicroBatch[InputT, PlanT]:
        selected = (
            candidate(best_width, best_estimated_check)
            if best is None
            or best_width != best.stats_global_count
            or best_estimated_check is not None
            else best
        )
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
        check = estimate_check(mid)
        if check is not None and not check.check.fits:
            rejected += 1
            high = mid - 1
            continue
        if check is not None:
            best_width = mid
            best_estimated_check = check
            best = None
            low = mid + 1
            continue
        item = candidate(mid, check)
        if item.check.fits:
            best = item
            best_width = mid
            best_estimated_check = None
            low = mid + 1
        else:
            rejected += 1
            high = mid - 1

    return finalize_best()
