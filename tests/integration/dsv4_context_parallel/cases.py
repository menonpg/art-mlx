from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Dsv4FamilyShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefix_length: int = Field(ge=1)
    completion_lengths: tuple[int, ...] = ()

    @field_validator("completion_lengths")
    @classmethod
    def _completion_lengths_include_context(
        cls, value: tuple[int, ...]
    ) -> tuple[int, ...]:
        if any(length < 2 for length in value):
            raise ValueError("completion lengths include a masked context token")
        return value


class Dsv4PackedRowShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: tuple[Dsv4FamilyShape, ...] = Field(min_length=1)


class Dsv4WorkloadCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    sequence_length: int = Field(ge=1)
    rows: tuple[Dsv4PackedRowShape, ...] = Field(min_length=1)
    seed: int = 0
    description: str = ""
    tags: tuple[str, ...] = ()


def dsv4_family_token_count(family: Dsv4FamilyShape) -> int:
    return int(family.prefix_length) + sum(
        int(length) for length in family.completion_lengths
    )


def dsv4_row_token_count(row: Dsv4PackedRowShape) -> int:
    return sum(dsv4_family_token_count(family) for family in row.families)


def fit_dsv4_family_to_remaining(
    family: Dsv4FamilyShape, remaining_tokens: int
) -> Dsv4FamilyShape | None:
    if int(remaining_tokens) < int(family.prefix_length) + 2:
        return None
    used = int(family.prefix_length)
    completions: list[int] = []
    for completion_length in family.completion_lengths:
        length = int(completion_length)
        if used + length > int(remaining_tokens):
            break
        completions.append(length)
        used += length
    if not completions:
        return None
    if len(completions) == len(family.completion_lengths):
        return family
    return Dsv4FamilyShape(
        prefix_length=family.prefix_length,
        completion_lengths=tuple(completions),
    )


def randomized_completion_lengths(
    *,
    seed: int,
    completion_count: int,
    mean: int,
    jitter: int,
) -> tuple[int, ...]:
    rng = random.Random(seed)
    return _randomized_completion_lengths(
        rng, completion_count=completion_count, mean=mean, jitter=jitter
    )


def randomized_repeated_case(
    *,
    name: str,
    sequence_length: int,
    seed: int,
    row_count: int = 1,
    prefix_length: int = 5000,
    completion_count: int = 16,
    completion_mean: int = 100,
    completion_jitter: int = 24,
    prefix_jitter: int = 512,
    exact_token_count: bool = False,
    tags: tuple[str, ...] = ("randomized_completions", "weak_scaling_candidate"),
) -> Dsv4WorkloadCase:
    rng = random.Random(seed)
    rows: list[Dsv4PackedRowShape] = []
    for _row_index in range(row_count):
        cursor = 0
        families: list[Dsv4FamilyShape] = []
        while cursor < sequence_length:
            remaining = sequence_length - cursor
            prefix = max(1, prefix_length + rng.randint(-prefix_jitter, prefix_jitter))
            if remaining < prefix + 2:
                if exact_token_count:
                    families.append(
                        Dsv4FamilyShape(
                            prefix_length=remaining,
                            completion_lengths=(),
                        )
                    )
                    cursor += remaining
                break
            lengths = _randomized_completion_lengths(
                rng,
                completion_count=completion_count,
                mean=completion_mean,
                jitter=completion_jitter,
            )
            family = fit_dsv4_family_to_remaining(
                Dsv4FamilyShape(
                    prefix_length=prefix,
                    completion_lengths=lengths,
                ),
                remaining,
            )
            if family is None:
                break
            families.append(family)
            cursor += dsv4_family_token_count(family)
        if exact_token_count and cursor < sequence_length and families:
            families.append(
                Dsv4FamilyShape(
                    prefix_length=sequence_length - cursor,
                    completion_lengths=(),
                )
            )
        if not families:
            families.append(
                Dsv4FamilyShape(
                    prefix_length=max(1, min(prefix_length, sequence_length - 2)),
                    completion_lengths=(2,),
                )
            )
        rows.append(Dsv4PackedRowShape(families=tuple(families)))
    return Dsv4WorkloadCase(
        name=name,
        sequence_length=sequence_length,
        rows=tuple(rows),
        seed=seed,
        description=(
            "Randomized shared-prefix workload with completion lengths varied per "
            "completion and per packed sequence."
        ),
        tags=tags,
    )


def canonical_benchmark_cases() -> tuple[Dsv4WorkloadCase, ...]:
    weak_scale = tuple(
        randomized_repeated_case(
            name=f"weak_scale_{topology}_{token_count}",
            sequence_length=token_count,
            seed=1000 + index,
            prefix_length=5000,
            completion_count=16,
            completion_mean=100,
            completion_jitter=24,
            prefix_jitter=384,
            exact_token_count=True,
            tags=(
                "benchmark",
                "weak_scaling_candidate",
                "randomized_completions",
            ),
        )
        for index, (topology, token_count) in enumerate(
            (
                ("cp1", 81920),
                ("cp2", 163840),
                ("cp4", 327680),
                ("cp8", 655360),
            )
        )
    )
    return weak_scale + (
        randomized_repeated_case(
            name="long_prefix_20k_4x120k",
            sequence_length=500000,
            seed=2001,
            prefix_length=20000,
            completion_count=4,
            completion_mean=120000,
            completion_jitter=8192,
            prefix_jitter=1024,
            exact_token_count=True,
            tags=("benchmark", "long_context", "randomized_completions"),
        ),
        randomized_repeated_case(
            name="many_small_families_81920",
            sequence_length=81920,
            seed=2002,
            prefix_length=256,
            completion_count=4,
            completion_mean=256,
            completion_jitter=96,
            prefix_jitter=64,
            exact_token_count=True,
            tags=("benchmark", "many_small_families", "randomized_completions"),
        ),
        randomized_repeated_case(
            name="dominant_family_with_background_81920",
            sequence_length=81920,
            seed=2003,
            prefix_length=20000,
            completion_count=8,
            completion_mean=4096,
            completion_jitter=1024,
            prefix_jitter=256,
            exact_token_count=True,
            tags=("benchmark", "dominant_family", "randomized_completions"),
        ),
    )


def default_validation_cases() -> tuple[Dsv4WorkloadCase, ...]:
    return (
        Dsv4WorkloadCase(
            name="single_family_two_branches",
            sequence_length=32,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=5, completion_lengths=(4, 7)),
                    )
                ),
            ),
            seed=11,
            description="One shared prefix with two sibling completions.",
        ),
        Dsv4WorkloadCase(
            name="multi_family_repeated",
            sequence_length=96,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=5, completion_lengths=(3, 4)),
                        Dsv4FamilyShape(prefix_length=6, completion_lengths=(5, 3)),
                        Dsv4FamilyShape(prefix_length=4, completion_lengths=(6, 4)),
                    )
                ),
            ),
            seed=13,
            description="Several independent prompt families in one packed row.",
        ),
        Dsv4WorkloadCase(
            name="ragged_family_mix",
            sequence_length=128,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=7, completion_lengths=(2, 6, 3)),
                        Dsv4FamilyShape(prefix_length=3, completion_lengths=(8, 2)),
                    )
                ),
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=9, completion_lengths=(4, 5)),
                        Dsv4FamilyShape(prefix_length=2, completion_lengths=(2, 3, 7)),
                    )
                ),
            ),
            seed=17,
            description="Ragged prefixes, branch counts, and completion lengths.",
            tags=("randomized_completions",),
        ),
        Dsv4WorkloadCase(
            name="dominant_family",
            sequence_length=192,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(
                            prefix_length=40, completion_lengths=(64, 9, 7)
                        ),
                        Dsv4FamilyShape(prefix_length=4, completion_lengths=(3, 3)),
                    )
                ),
            ),
            seed=19,
            description="One long family plus a small background family.",
        ),
        Dsv4WorkloadCase(
            name="long_sibling",
            sequence_length=256,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(
                            prefix_length=8,
                            completion_lengths=(160, 9, 7),
                        ),
                        Dsv4FamilyShape(prefix_length=6, completion_lengths=(4, 4)),
                    )
                ),
            ),
            seed=23,
            description="One sibling completion dominates the row.",
        ),
        Dsv4WorkloadCase(
            name="padding_tail",
            sequence_length=80,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=6, completion_lengths=(4, 4)),
                        Dsv4FamilyShape(prefix_length=5, completion_lengths=(3, 3)),
                    )
                ),
            ),
            seed=29,
            description="Real tokens followed by padding.",
        ),
        Dsv4WorkloadCase(
            name="cp_boundary_prefix",
            sequence_length=96,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=30, completion_lengths=(4, 4)),
                        Dsv4FamilyShape(prefix_length=8, completion_lengths=(5, 5)),
                    )
                ),
            ),
            seed=31,
            description="A prefix crosses a proportional CP partition boundary.",
        ),
        Dsv4WorkloadCase(
            name="cp_boundary_completion",
            sequence_length=112,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=8, completion_lengths=(35, 4)),
                        Dsv4FamilyShape(prefix_length=6, completion_lengths=(5, 5)),
                    )
                ),
            ),
            seed=37,
            description="A completion crosses a proportional CP partition boundary.",
        ),
        Dsv4WorkloadCase(
            name="family_boundary_at_partition",
            sequence_length=80,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=12, completion_lengths=(20,)),
                        Dsv4FamilyShape(prefix_length=8, completion_lengths=(24,)),
                    )
                ),
            ),
            seed=41,
            description="A whole family boundary lands exactly on CP2.",
        ),
        Dsv4WorkloadCase(
            name="empty_trailing_rank",
            sequence_length=8,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=2, completion_lengths=(2,)),
                    )
                ),
            ),
            seed=43,
            description="Tiny row leaves trailing CP ranks empty.",
        ),
        Dsv4WorkloadCase(
            name="csa_ratio_boundary",
            sequence_length=96,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=5, completion_lengths=(6, 10)),
                        Dsv4FamilyShape(prefix_length=7, completion_lengths=(9, 11)),
                    )
                ),
            ),
            seed=47,
            description="Prompt and completion boundaries are not aligned to CSA4.",
            tags=("csa_ratio_boundary",),
        ),
        Dsv4WorkloadCase(
            name="hca_ratio_boundary",
            sequence_length=320,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=129, completion_lengths=(130, 5)),
                    )
                ),
            ),
            seed=53,
            description="Prompt and completion boundaries are not aligned to HCA128.",
            tags=("hca_ratio_boundary",),
        ),
        Dsv4WorkloadCase(
            name="swa_boundary",
            sequence_length=320,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(
                            prefix_length=16,
                            completion_lengths=(127, 128, 129),
                        ),
                    )
                ),
            ),
            seed=59,
            description="Completion lengths straddle the raw SWA window.",
            tags=("swa_boundary",),
        ),
        Dsv4WorkloadCase(
            name="topk_tie_or_near_tie",
            sequence_length=96,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=8, completion_lengths=(8, 8, 8)),
                    )
                ),
            ),
            seed=61,
            description="Reserved diagnostic for indexer tie handling.",
            tags=("topk_tie_or_near_tie",),
        ),
        Dsv4WorkloadCase(
            name="no_stage_keys",
            sequence_length=64,
            rows=(
                Dsv4PackedRowShape(
                    families=(
                        Dsv4FamilyShape(prefix_length=2, completion_lengths=(2, 2)),
                    )
                ),
            ),
            seed=67,
            description="Reserved diagnostic for stages with no visible keys.",
            tags=("no_stage_keys",),
        ),
    )


def _randomized_completion_lengths(
    rng: random.Random,
    *,
    completion_count: int,
    mean: int,
    jitter: int,
) -> tuple[int, ...]:
    low = max(2, int(mean) - int(jitter))
    high = max(low, int(mean) + int(jitter))
    lengths = [rng.randint(low, high) for _ in range(completion_count)]
    if completion_count > 1 and len(set(lengths)) == 1:
        lengths[-1] += 1
    return tuple(lengths)
