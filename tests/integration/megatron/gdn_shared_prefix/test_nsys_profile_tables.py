from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from .nsys_profile_tables import parse_nsys_sqlite


def test_parse_nsys_profile_tables_assigns_kernels_by_launch_range(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "profile.sqlite"
    _write_synthetic_nsys_sqlite(sqlite_path)

    tables = parse_nsys_sqlite(
        sqlite_path,
        tmp_path / "tables",
        expected_ranges=(
            "art_gdn_lab_forward",
            "art_gdn_in_proj",
            "art_gdn_recurrent_forward",
            "art_gdn_lab_backward",
            "art_gdn_missing_expected",
        ),
        nvtx_prefixes=("art_gdn", "autograd::", "aten::"),
        top_kernels=2,
    )

    range_by_label = {row.label: row for row in tables.nvtx_range_summary}
    kernel_by_label = {row.label: row for row in tables.kernel_by_deepest_range}
    assert range_by_label["art_gdn_lab_forward"].calls == 1
    assert range_by_label["art_gdn_lab_forward"].cpu_total_ms == pytest.approx(100.0)
    assert range_by_label["art_gdn_lab_forward"].gpu_kernel_total_ms == pytest.approx(
        12.0
    )
    assert kernel_by_label["art_gdn_in_proj"].gpu_total_ms == pytest.approx(2.0)
    assert kernel_by_label["art_gdn_recurrent_forward"].gpu_total_ms == pytest.approx(
        10.0
    )
    assert kernel_by_label[
        "autograd::engine::evaluate_function: MulBackward0"
    ].gpu_total_ms == pytest.approx(3.0)
    assert range_by_label["art_gdn_dynamic_cp_range"].calls == 1
    assert tables.top_kernels[0].kernel_name == "recurrent_kernel"
    assert tables.missing_expected_ranges == ("art_gdn_missing_expected",)
    assert (
        Path(tables.paths.markdown_path)
        .read_text(encoding="utf-8")
        .startswith("# GDN Nsys Profile Tables")
    )
    assert Path(tables.paths.nvtx_csv_path).exists()
    assert Path(tables.paths.kernel_by_range_csv_path).exists()
    assert Path(tables.paths.top_kernels_csv_path).exists()


def _write_synthetic_nsys_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("create table StringIds(id integer primary key, value text)")
        connection.executemany(
            "insert into StringIds(id, value) values(?, ?)",
            (
                (1, "in_proj_kernel"),
                (2, "recurrent_kernel"),
                (3, "backward_kernel"),
            ),
        )
        connection.execute(
            "create table NVTX_EVENTS(start integer, end integer, text text, textId integer, jsonText text, jsonTextId integer)"
        )
        connection.executemany(
            "insert into NVTX_EVENTS(start, end, text, textId, jsonText, jsonTextId) values(?, ?, ?, null, null, null)",
            (
                (0, 100_000_000, "art_gdn_lab_forward"),
                (10_000_000, 30_000_000, "art_gdn_in_proj"),
                (30_000_000, 90_000_000, "art_gdn_recurrent_forward"),
                (100_000_000, 160_000_000, "art_gdn_lab_backward"),
                (
                    105_000_000,
                    120_000_000,
                    "autograd::engine::evaluate_function: MulBackward0",
                ),
                (170_000_000, 180_000_000, "art_gdn_dynamic_cp_range"),
            ),
        )
        connection.execute(
            "create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, correlationId integer)"
        )
        connection.executemany(
            "insert into CUPTI_ACTIVITY_KIND_RUNTIME(start, end, correlationId) values(?, ?, ?)",
            (
                (12_000_000, 13_000_000, 101),
                (40_000_000, 41_000_000, 102),
                (110_000_000, 111_000_000, 103),
            ),
        )
        connection.execute(
            "create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, correlationId integer, shortName integer, demangledName integer, mangledName integer)"
        )
        connection.executemany(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL(start, end, correlationId, shortName, demangledName, mangledName) values(?, ?, ?, ?, null, null)",
            (
                (200_000_000, 202_000_000, 101, 1),
                (210_000_000, 220_000_000, 102, 2),
                (230_000_000, 233_000_000, 103, 3),
            ),
        )
