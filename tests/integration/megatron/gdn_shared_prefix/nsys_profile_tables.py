from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

NS_PER_MS = 1_000_000.0


class NsysTablePaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    sqlite_path: str
    json_path: str
    markdown_path: str
    nvtx_csv_path: str
    kernel_by_range_csv_path: str
    top_kernels_csv_path: str


class NsysNvtxRangeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    calls: int
    cpu_total_ms: float
    cpu_median_ms: float
    cpu_p90_ms: float
    cpu_max_ms: float
    cuda_api_total_ms: float
    cuda_api_calls: int
    gpu_kernel_total_ms: float
    gpu_kernel_count: int


class NsysKernelByRangeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    kernel_count: int
    gpu_total_ms: float
    gpu_median_ms: float
    gpu_p90_ms: float
    gpu_max_ms: float


class NsysTopKernelSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_name: str
    calls: int
    gpu_total_ms: float
    gpu_median_ms: float
    gpu_p90_ms: float
    gpu_max_ms: float


class NsysProfileTables(BaseModel):
    model_config = ConfigDict(frozen=True)

    paths: NsysTablePaths
    nvtx_range_summary: tuple[NsysNvtxRangeSummary, ...]
    kernel_by_deepest_range: tuple[NsysKernelByRangeSummary, ...]
    top_kernels: tuple[NsysTopKernelSummary, ...]
    missing_expected_ranges: tuple[str, ...] = Field(default_factory=tuple)


class _Range(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class _RuntimeEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_ns: int
    end_ns: int
    correlation_id: int | None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def midpoint_ns(self) -> int:
        return self.start_ns + self.duration_ns // 2


class _KernelEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_ns: int
    end_ns: int
    correlation_id: int | None
    name: str

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def export_nsys_sqlite(report_path: Path, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "-o",
            str(sqlite_path),
            str(report_path),
        ),
        check=True,
        text=True,
    )


def parse_nsys_sqlite(
    sqlite_path: Path,
    output_dir: Path,
    *,
    expected_ranges: tuple[str, ...] = (),
    nvtx_prefix: str = "art_gdn",
    nvtx_prefixes: tuple[str, ...] | None = None,
    top_kernels: int = 20,
) -> NsysProfileTables:
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        string_ids = _read_string_ids(connection)
        ranges = _read_nvtx_ranges(
            connection,
            string_ids,
            expected_ranges=expected_ranges,
            nvtx_prefixes=nvtx_prefixes or (nvtx_prefix,),
        )
        runtime_events = _read_runtime_events(connection)
        kernels = _read_kernel_events(connection, string_ids)

    runtime_by_correlation = {
        event.correlation_id: event
        for event in runtime_events
        if event.correlation_id is not None
    }
    range_summary = _summarize_ranges(
        ranges,
        runtime_events,
        kernels,
        runtime_by_correlation,
        expected_ranges=expected_ranges,
    )
    kernel_by_range = _summarize_kernels_by_deepest_range(
        ranges, kernels, runtime_by_correlation
    )
    top_kernel_rows = _summarize_top_kernels(kernels, limit=top_kernels)
    paths = NsysTablePaths(
        sqlite_path=str(sqlite_path),
        json_path=str(output_dir / "profile_tables.json"),
        markdown_path=str(output_dir / "profile_report.md"),
        nvtx_csv_path=str(output_dir / "profile_nvtx_ranges.csv"),
        kernel_by_range_csv_path=str(output_dir / "profile_kernel_by_range.csv"),
        top_kernels_csv_path=str(output_dir / "profile_top_kernels.csv"),
    )
    tables = NsysProfileTables(
        paths=paths,
        nvtx_range_summary=range_summary,
        kernel_by_deepest_range=kernel_by_range,
        top_kernels=top_kernel_rows,
        missing_expected_ranges=tuple(
            label
            for label in expected_ranges
            if all(row.label != label or row.calls == 0 for row in range_summary)
        ),
    )
    _write_json(Path(paths.json_path), tables)
    _write_csv(Path(paths.nvtx_csv_path), tables.nvtx_range_summary)
    _write_csv(Path(paths.kernel_by_range_csv_path), tables.kernel_by_deepest_range)
    _write_csv(Path(paths.top_kernels_csv_path), tables.top_kernels)
    Path(paths.markdown_path).write_text(_render_markdown(tables), encoding="utf-8")
    return tables


def _read_string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    if not _has_table(connection, "StringIds"):
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute("select id, value from StringIds")
    }


def _read_nvtx_ranges(
    connection: sqlite3.Connection,
    string_ids: dict[int, str],
    *,
    expected_ranges: tuple[str, ...],
    nvtx_prefixes: tuple[str, ...],
) -> tuple[_Range, ...]:
    if not _has_table(connection, "NVTX_EVENTS"):
        return ()
    columns = _columns(connection, "NVTX_EVENTS")
    rows = connection.execute(
        "select "
        + ", ".join(
            (
                _select_expr(columns, "start"),
                _select_expr(columns, "end"),
                _select_expr(columns, "text"),
                _select_expr(columns, "textId"),
                _select_expr(columns, "jsonText"),
                _select_expr(columns, "jsonTextId"),
            )
        )
        + " from NVTX_EVENTS where end is not null"
    )
    expected = set(expected_ranges)
    ranges = []
    for start, end, text, text_id, json_text, json_text_id in rows:
        label = _resolve_text(text, text_id, string_ids) or _resolve_text(
            json_text, json_text_id, string_ids
        )
        if label is None:
            continue
        if label in expected or label.startswith(nvtx_prefixes):
            ranges.append(_Range(label=label, start_ns=int(start), end_ns=int(end)))
    return tuple(ranges)


def _read_runtime_events(
    connection: sqlite3.Connection,
) -> tuple[_RuntimeEvent, ...]:
    if not _has_table(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return ()
    rows = connection.execute(
        "select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
    )
    return tuple(
        _RuntimeEvent(
            start_ns=int(start),
            end_ns=int(end),
            correlation_id=None if correlation_id is None else int(correlation_id),
        )
        for start, end, correlation_id in rows
    )


def _read_kernel_events(
    connection: sqlite3.Connection, string_ids: dict[int, str]
) -> tuple[_KernelEvent, ...]:
    if not _has_table(connection, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return ()
    columns = _columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    rows = connection.execute(
        "select "
        + ", ".join(
            (
                _select_expr(columns, "start"),
                _select_expr(columns, "end"),
                _select_expr(columns, "correlationId"),
                _select_expr(columns, "shortName"),
                _select_expr(columns, "demangledName"),
                _select_expr(columns, "mangledName"),
            )
        )
        + " from CUPTI_ACTIVITY_KIND_KERNEL"
    )
    kernels = []
    for start, end, correlation_id, short_name, demangled_name, mangled_name in rows:
        name = (
            _resolve_text(short_name, None, string_ids)
            or _resolve_text(demangled_name, None, string_ids)
            or _resolve_text(mangled_name, None, string_ids)
            or "[unknown]"
        )
        kernels.append(
            _KernelEvent(
                start_ns=int(start),
                end_ns=int(end),
                correlation_id=None if correlation_id is None else int(correlation_id),
                name=name,
            )
        )
    return tuple(kernels)


def _summarize_ranges(
    ranges: tuple[_Range, ...],
    runtime_events: tuple[_RuntimeEvent, ...],
    kernels: tuple[_KernelEvent, ...],
    runtime_by_correlation: dict[int | None, _RuntimeEvent],
    *,
    expected_ranges: tuple[str, ...],
) -> tuple[NsysNvtxRangeSummary, ...]:
    labels = _ordered_labels(ranges, expected_ranges)
    rows = []
    for label in labels:
        label_ranges = [event for event in ranges if event.label == label]
        runtime_inside = [
            event
            for event in runtime_events
            if any(
                _point_in_range(event.midpoint_ns, nvtx_range)
                for nvtx_range in label_ranges
            )
        ]
        kernels_inside = [
            kernel
            for kernel in kernels
            if any(
                _point_in_range(
                    _kernel_attribution_point(kernel, runtime_by_correlation),
                    nvtx_range,
                )
                for nvtx_range in label_ranges
            )
        ]
        cpu_durations = [event.duration_ns for event in label_ranges]
        runtime_durations = [event.duration_ns for event in runtime_inside]
        kernel_durations = [event.duration_ns for event in kernels_inside]
        rows.append(
            NsysNvtxRangeSummary(
                label=label,
                calls=len(label_ranges),
                cpu_total_ms=_to_ms(sum(cpu_durations)),
                cpu_median_ms=_to_ms(_median(cpu_durations)),
                cpu_p90_ms=_to_ms(_p90(cpu_durations)),
                cpu_max_ms=_to_ms(max(cpu_durations, default=0)),
                cuda_api_total_ms=_to_ms(sum(runtime_durations)),
                cuda_api_calls=len(runtime_inside),
                gpu_kernel_total_ms=_to_ms(sum(kernel_durations)),
                gpu_kernel_count=len(kernels_inside),
            )
        )
    return tuple(rows)


def _summarize_kernels_by_deepest_range(
    ranges: tuple[_Range, ...],
    kernels: tuple[_KernelEvent, ...],
    runtime_by_correlation: dict[int | None, _RuntimeEvent],
) -> tuple[NsysKernelByRangeSummary, ...]:
    by_label: dict[str, list[int]] = {}
    for kernel in kernels:
        label = _deepest_range_label(
            ranges, _kernel_attribution_point(kernel, runtime_by_correlation)
        )
        if label is not None:
            by_label.setdefault(label, []).append(kernel.duration_ns)
    rows = [
        NsysKernelByRangeSummary(
            label=label,
            kernel_count=len(durations),
            gpu_total_ms=_to_ms(sum(durations)),
            gpu_median_ms=_to_ms(_median(durations)),
            gpu_p90_ms=_to_ms(_p90(durations)),
            gpu_max_ms=_to_ms(max(durations, default=0)),
        )
        for label, durations in by_label.items()
    ]
    return tuple(sorted(rows, key=lambda row: row.gpu_total_ms, reverse=True))


def _summarize_top_kernels(
    kernels: tuple[_KernelEvent, ...], *, limit: int
) -> tuple[NsysTopKernelSummary, ...]:
    by_name: dict[str, list[int]] = {}
    for kernel in kernels:
        by_name.setdefault(kernel.name, []).append(kernel.duration_ns)
    rows = [
        NsysTopKernelSummary(
            kernel_name=name,
            calls=len(durations),
            gpu_total_ms=_to_ms(sum(durations)),
            gpu_median_ms=_to_ms(_median(durations)),
            gpu_p90_ms=_to_ms(_p90(durations)),
            gpu_max_ms=_to_ms(max(durations, default=0)),
        )
        for name, durations in by_name.items()
    ]
    return tuple(sorted(rows, key=lambda row: row.gpu_total_ms, reverse=True)[:limit])


def _ordered_labels(
    ranges: tuple[_Range, ...], expected_ranges: tuple[str, ...]
) -> tuple[str, ...]:
    seen = set[str]()
    labels = []
    for label in expected_ranges:
        labels.append(label)
        seen.add(label)
    dynamic = sorted(
        {event.label for event in ranges if event.label not in seen},
        key=lambda label: sum(
            event.duration_ns for event in ranges if event.label == label
        ),
        reverse=True,
    )
    labels.extend(dynamic)
    return tuple(labels)


def _deepest_range_label(ranges: tuple[_Range, ...], point_ns: int) -> str | None:
    matches = [event for event in ranges if _point_in_range(point_ns, event)]
    if not matches:
        return None
    return min(matches, key=lambda event: event.duration_ns).label


def _kernel_attribution_point(
    kernel: _KernelEvent, runtime_by_correlation: dict[int | None, _RuntimeEvent]
) -> int:
    runtime = runtime_by_correlation.get(kernel.correlation_id)
    if runtime is not None:
        return runtime.midpoint_ns
    return kernel.start_ns + kernel.duration_ns // 2


def _point_in_range(point_ns: int, nvtx_range: _Range) -> bool:
    return nvtx_range.start_ns <= point_ns <= nvtx_range.end_ns


def _resolve_text(
    text_or_id: object, text_id: object | None, string_ids: dict[int, str]
) -> str | None:
    if isinstance(text_or_id, str):
        return text_or_id
    if isinstance(text_or_id, int):
        return string_ids.get(text_or_id)
    if isinstance(text_id, int):
        return string_ids.get(text_id)
    return None


def _select_expr(columns: set[str], column: str) -> str:
    if column in columns:
        return column
    return f"NULL as {column}"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"pragma table_info({table})").fetchall()
    }


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()
    return row is not None


def _median(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    return sorted_values[len(sorted_values) // 2]


def _p90(values: list[int]) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    return sorted_values[
        min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))
    ]


def _to_ms(ns: int) -> float:
    return float(ns) / NS_PER_MS


def _write_json(path: Path, tables: NsysProfileTables) -> None:
    path.write_text(
        json.dumps(tables.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: tuple[BaseModel, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = tuple(type(rows[0]).model_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def _render_markdown(tables: NsysProfileTables) -> str:
    return "\n".join(
        (
            "# GDN Nsys Profile Tables",
            "",
            "Definitions:",
            "",
            "- NVTX CPU columns measure the host-side range duration from `range_push` to `range_pop`.",
            "- Inclusive CUDA kernel time assigns kernels to a range by the CUDA launch API correlation and includes child ranges.",
            "- Deepest-range kernel time counts each kernel once under the narrowest matching NVTX range, so it is the easiest table for spotting where GPU time landed.",
            "- CUDA API time is host runtime API time whose midpoint occurs inside the NVTX range.",
            "",
            f"SQLite source: `{tables.paths.sqlite_path}`",
            "",
            "## Top-Level Lab Ranges",
            "",
            _markdown_table(
                [
                    row
                    for row in tables.nvtx_range_summary
                    if row.label.startswith("art_gdn_lab_")
                ],
                (
                    "label",
                    "calls",
                    "cpu_total_ms",
                    "cpu_median_ms",
                    "gpu_kernel_total_ms",
                    "gpu_kernel_count",
                    "cuda_api_total_ms",
                ),
            ),
            "",
            "## Operator NVTX Ranges",
            "",
            _markdown_table(
                [
                    row
                    for row in tables.nvtx_range_summary
                    if not row.label.startswith("art_gdn_lab_")
                ],
                (
                    "label",
                    "calls",
                    "cpu_total_ms",
                    "cpu_median_ms",
                    "gpu_kernel_total_ms",
                    "gpu_kernel_count",
                    "cuda_api_total_ms",
                ),
            ),
            "",
            "## Kernel Time By Deepest NVTX Range",
            "",
            _markdown_table(
                tables.kernel_by_deepest_range,
                (
                    "label",
                    "kernel_count",
                    "gpu_total_ms",
                    "gpu_median_ms",
                    "gpu_p90_ms",
                    "gpu_max_ms",
                ),
            ),
            "",
            "## Top CUDA Kernels",
            "",
            _markdown_table(
                [
                    row.model_copy(
                        update={"kernel_name": _shorten(row.kernel_name, limit=96)}
                    )
                    for row in tables.top_kernels
                ],
                (
                    "kernel_name",
                    "calls",
                    "gpu_total_ms",
                    "gpu_median_ms",
                    "gpu_p90_ms",
                    "gpu_max_ms",
                ),
            ),
            "",
            "## Missing Expected NVTX Ranges",
            "",
            _markdown_table(
                [{"label": label} for label in tables.missing_expected_ranges],
                ("label",),
            ),
            "",
        )
    )


def _markdown_table(rows: list[Any] | tuple[Any, ...], fields: tuple[str, ...]) -> str:
    if not rows:
        return "_No rows._"
    normalized = [_row_dict(row) for row in rows]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in normalized:
        lines.append(
            "| "
            + " | ".join(_format_cell(row.get(field, "")) for field in fields)
            + " |"
        )
    return "\n".join(lines)


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, BaseModel):
        return row.model_dump()
    return dict(row)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _shorten(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
