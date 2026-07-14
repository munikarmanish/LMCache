#!/usr/bin/env python
"""Plot Sweep A (capacity headline) from working_set_tester's period CSVs.

Sweep A runs working_set_tester in fixed mode, which GROWS the working set over
time toward the target; each assessment period is logged with its own
``working_set_size`` and TTFT percentiles in ``sustained_periods_*.csv`` under
each run's ``kvct/`` dir. That per-period working-set is the real x-axis — so we
read those files directly (not the whole-run ``tiers.csv`` tally, which double-
counts retry log lines and can't resolve per-working-set).

For each adapter we plot TTFT (median + p95) vs per-period working set, one line
per adapter (cxl / nixl). The divergence above a single node's 64 GB L1 is the
capacity win: nixl must recompute (deleted-on-eviction) so its TTFT jumps to the
full prefill cost, while cxl still serves from the CXL tier at load latency.

The adapter is taken from the run-dir name (``A_<adapter>_<stamp>``), since the
period CSV itself carries no adapter column. Multiple periods at the same
working-set (fixed mode repeats some) are averaged.

Usage:
    python plot_sweep.py --root results/sweep [--latest] [--out results/sweep/plots]

    --latest plots only the newest run per adapter (the usual "compare my two
    most recent runs" case); without it, all runs under --root are averaged
    together per adapter.
"""
# Standard
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import re

# Third Party
import matplotlib.pyplot as plt

# This testbed's measured KV size anchor for Llama-3.1-8B-Instruct:
# 1.2188 GB per 10k tokens. Used only to mark the knees that bound the CXL band.
GB_PER_1K_TOK = 0.12188
# Single-node L1 (64 GB): above this a single node overflows local DRAM.
L1_NODE_TOK = 64 / GB_PER_1K_TOK * 1000  # ~0.525e6 tok
# CXL pool ceiling (511 x 256 MiB = 127.75 GB): the backend has no live-chunk
# eviction, so at/above this stores are dropped — sweeps stay left of it.
CXL_POOL_TOK = 127.75 / GB_PER_1K_TOK * 1000  # ~1.048e6 tok

# Run-dir name -> adapter, e.g. "A_cxl_20260708-041411" -> "cxl".
_RUN_DIR_RE = re.compile(r"^[AB]_(?P<adapter>cxl|nixl)(?:_[a-z_]+)?_\d{8}-\d{6}$")


def adapter_of(period_csv: Path) -> str:
    """Infer the adapter (cxl / nixl) from a period CSV's run-dir name.

    Args:
        period_csv: Path to a ``sustained_periods_*.csv`` inside
            ``<root>/<run-dir>/kvct/``.

    Returns:
        The adapter label, or ``"unknown"`` if the run-dir name doesn't parse.
    """
    # .../<run-dir>/kvct/sustained_periods_*.csv -> parents[1] is <run-dir>.
    run_dir = period_csv.parents[1].name
    m = _RUN_DIR_RE.match(run_dir)
    return m.group("adapter") if m else "unknown"


def find_period_csvs(root: Path, latest_only: bool) -> list[Path]:
    """Discover the period CSVs to plot under ``root``.

    Args:
        root: Results root containing ``A_<adapter>_*/kvct/`` dirs.
        latest_only: If True, keep only the newest CSV per adapter (by run-dir
            timestamp in the name, which sorts lexically); otherwise return
            every ``sustained_periods_*.csv`` found.

    Returns:
        The period CSV paths to load, sorted ascending.
    """
    all_csvs = sorted(root.rglob("sustained_periods_*.csv"))
    if not latest_only:
        return all_csvs
    # Run-dir names embed a sortable ``YYYYMMDD-HHMMSS`` stamp, so the max path
    # per adapter is the newest run. parents[1] is the ``A_<adapter>_<stamp>``
    # dir; keeping the max over that string picks the latest.
    latest: dict[str, Path] = {}
    for csv_path in all_csvs:
        adapter = adapter_of(csv_path)
        run_dir = csv_path.parents[1].name
        if adapter not in latest or run_dir > latest[adapter].parents[1].name:
            latest[adapter] = csv_path
    return sorted(latest.values())


def load_periods(csv_paths: list[Path]) -> dict[str, list[tuple[float, float, float]]]:
    """Collect (working_set, median_ttft, p95_ttft) per adapter from CSVs.

    Reads the given ``sustained_periods_*.csv`` files. Periods sharing a
    working-set value (fixed mode repeats some) are averaged so each adapter has
    one point per distinct working set.

    Args:
        csv_paths: Period CSV files to read (see :func:`find_period_csvs`).

    Returns:
        Mapping adapter -> list of (working_set_tokens, mean_median_ttft_s,
        mean_p95_ttft_s, min_median_ttft_s), sorted ascending by working set.
        The min-median is the fastest period at that working set — the warm /
        steady-state value, free of cold-transition periods that inflate the
        mean under init=min growth.
    """
    # adapter -> working_set -> list of (median, p95) across periods.
    acc: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for csv_path in csv_paths:
        adapter = adapter_of(csv_path)
        with csv_path.open() as fh:
            for r in csv.DictReader(fh):
                try:
                    ws = int(r["working_set_size"])
                    med = float(r["median_ttft"])
                    p95 = float(r["p95_ttft"])
                except (KeyError, ValueError):
                    continue
                acc[adapter][ws].append((med, p95))

    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for adapter, by_ws in acc.items():
        pts: list[tuple[float, float, float, float]] = []
        for ws, vals in by_ws.items():
            med = sum(v[0] for v in vals) / len(vals)
            p95 = sum(v[1] for v in vals) / len(vals)
            min_med = min(v[0] for v in vals)
            pts.append((float(ws), med, p95, min_med))
        pts.sort()
        out[adapter] = pts
    return out


def plot_sweep_a(
    periods: dict[str, list[tuple[float, float, float, float]]], out_dir: Path
) -> None:
    """Render TTFT-vs-working-set for each adapter.

    For each adapter three lines are drawn: the mean-across-periods median TTFT
    (solid), the min-across-periods median TTFT (dotted — the warm/steady-state
    value, free of cold-transition periods), and the mean p95 (dashed).

    Args:
        periods: Output of :func:`load_periods`.
        out_dir: Directory to write ``sweep_a.png`` into.
    """
    if not periods:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for adapter, pts in sorted(periods.items()):
        ws = [p[0] / 1e6 for p in pts]
        med = [p[1] for p in pts]
        p95 = [p[2] for p in pts]
        min_med = [p[3] for p in pts]
        line = ax.plot(ws, med, marker="o", label=f"{adapter} median TTFT")[0]
        c = line.get_color()
        ax.plot(ws, min_med, marker="^", ls=":", c=c, alpha=0.9,
                label=f"{adapter} min TTFT (warm)")
        ax.plot(ws, p95, marker=".", ls="--", c=c, alpha=0.5,
                label=f"{adapter} p95 TTFT")

    ax.axvline(L1_NODE_TOK / 1e6, ls="--", c="gray", lw=1)
    ax.axvline(CXL_POOL_TOK / 1e6, ls=":", c="crimson", lw=1)
    ax.set_xlabel("working set (M tokens)  — dashed gray: 1-node L1 (64 GB); "
                  "dotted red: CXL pool ceiling (128 GB)")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("Sweep A: TTFT vs working set — CXL holds evicted chunks, "
                 "NIXL recomputes")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "sweep_a.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="results root with A_<adapter>_*/kvct/ dirs")
    parser.add_argument("--out", default="",
                        help="plot output dir (default: <root>/plots)")
    parser.add_argument("--latest", action="store_true",
                        help="plot only the newest run per adapter (by run-dir "
                        "timestamp) instead of averaging across all runs")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_period_csvs(root, args.latest)
    if not csv_paths:
        raise SystemExit(f"no sustained_periods_*.csv found under {root}")
    for p in csv_paths:
        print(f"reading {adapter_of(p)}: {p.parents[1].name}")
    periods = load_periods(csv_paths)
    for adapter, pts in sorted(periods.items()):
        print(f"{adapter}: {len(pts)} working-set points")
    plot_sweep_a(periods, out_dir)


if __name__ == "__main__":
    main()
