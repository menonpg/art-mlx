from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GdnFamilyShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefix_length: int = Field(ge=1)
    suffix_lengths: tuple[int, ...] = Field(min_length=1)


class GdnPackedRowShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    families: tuple[GdnFamilyShape, ...] = Field(min_length=1)


class GdnPhase0Case(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    sequence_length: int = Field(ge=1)
    rows: tuple[GdnPackedRowShape, ...] = Field(min_length=1)
    seed: int = 0
    description: str = ""


def gdn_family_token_count(family: GdnFamilyShape) -> int:
    return int(family.prefix_length) + sum(
        int(length) for length in family.suffix_lengths
    )


def fit_gdn_family_to_remaining(
    family: GdnFamilyShape, remaining_tokens: int
) -> GdnFamilyShape | None:
    if int(remaining_tokens) < int(family.prefix_length):
        return None
    used = int(family.prefix_length)
    suffixes: list[int] = []
    for suffix_length in family.suffix_lengths:
        length = int(suffix_length)
        if used + length > int(remaining_tokens):
            break
        suffixes.append(length)
        used += length
    if not suffixes:
        return None
    if len(suffixes) == len(family.suffix_lengths):
        return family
    return GdnFamilyShape(
        prefix_length=family.prefix_length,
        suffix_lengths=tuple(suffixes),
    )


def default_phase0_cases(conv_width: int = 4) -> tuple[GdnPhase0Case, ...]:
    return (
        GdnPhase0Case(
            name="single_family_two_branches",
            sequence_length=24,
            rows=(
                GdnPackedRowShape(
                    families=(GdnFamilyShape(prefix_length=5, suffix_lengths=(3, 4)),)
                ),
            ),
            seed=11,
            description="One prompt family with two child completions.",
        ),
        GdnPhase0Case(
            name="multi_family_repeated",
            sequence_length=64,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=5, suffix_lengths=(3, 3)),
                        GdnFamilyShape(prefix_length=6, suffix_lengths=(2, 4)),
                        GdnFamilyShape(prefix_length=4, suffix_lengths=(5, 3)),
                    )
                ),
            ),
            seed=13,
            description="Several independent prompt families in one packed row.",
        ),
        GdnPhase0Case(
            name="ragged_family_mix",
            sequence_length=96,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=7, suffix_lengths=(2, 6, 3)),
                        GdnFamilyShape(prefix_length=3, suffix_lengths=(8, 1)),
                    )
                ),
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=9, suffix_lengths=(4, 5)),
                        GdnFamilyShape(prefix_length=2, suffix_lengths=(2, 2, 7)),
                    )
                ),
            ),
            seed=17,
            description="Ragged prefix lengths, branch counts, and suffix lengths.",
        ),
        GdnPhase0Case(
            name="dominant_family",
            sequence_length=128,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=32, suffix_lengths=(20, 7, 5)),
                        GdnFamilyShape(prefix_length=4, suffix_lengths=(3, 3)),
                    )
                ),
            ),
            seed=19,
            description="One long family plus a small background family.",
        ),
        GdnPhase0Case(
            name="conv_tail_boundary",
            sequence_length=64,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(
                            prefix_length=conv_width + 2,
                            suffix_lengths=(conv_width - 1, conv_width, conv_width + 1),
                        ),
                    )
                ),
            ),
            seed=23,
            description="Suffixes shorter than, equal to, and longer than conv width.",
        ),
        GdnPhase0Case(
            name="padding_tail",
            sequence_length=80,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=6, suffix_lengths=(4, 4)),
                        GdnFamilyShape(prefix_length=5, suffix_lengths=(3, 3)),
                    )
                ),
            ),
            seed=29,
            description="Real tokens followed by padding.",
        ),
        GdnPhase0Case(
            name="cp_boundary_prefix",
            sequence_length=96,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=30, suffix_lengths=(4, 4)),
                        GdnFamilyShape(prefix_length=8, suffix_lengths=(5, 5)),
                    )
                ),
            ),
            seed=31,
            description="A prefix crosses a proportional CP partition boundary.",
        ),
        GdnPhase0Case(
            name="cp_boundary_suffix",
            sequence_length=112,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=8, suffix_lengths=(35, 4)),
                        GdnFamilyShape(prefix_length=6, suffix_lengths=(5, 5)),
                    )
                ),
            ),
            seed=37,
            description="A suffix crosses a proportional CP partition boundary.",
        ),
        GdnPhase0Case(
            name="long_sibling",
            sequence_length=192,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=8, suffix_lengths=(96, 7, 5)),
                        GdnFamilyShape(prefix_length=6, suffix_lengths=(4, 4)),
                    )
                ),
            ),
            seed=41,
            description="One sibling completion dominates the row and crosses CP waves.",
        ),
        GdnPhase0Case(
            name="many_branches_wave",
            sequence_length=96,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(
                            prefix_length=4,
                            suffix_lengths=(2, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4),
                        ),
                    )
                ),
            ),
            seed=43,
            description="Many short siblings force multi-wave completion scheduling.",
        ),
        GdnPhase0Case(
            name="family_boundary_at_partition",
            sequence_length=80,
            rows=(
                GdnPackedRowShape(
                    families=(
                        GdnFamilyShape(prefix_length=12, suffix_lengths=(20,)),
                        GdnFamilyShape(prefix_length=8, suffix_lengths=(24,)),
                    )
                ),
            ),
            seed=47,
            description="A whole family boundary lands exactly on the CP2 partition.",
        ),
        GdnPhase0Case(
            name="empty_trailing_rank",
            sequence_length=8,
            rows=(
                GdnPackedRowShape(
                    families=(GdnFamilyShape(prefix_length=2, suffix_lengths=(2,)),)
                ),
            ),
            seed=53,
            description="Tiny row leaves trailing CP ranks empty.",
        ),
    )
