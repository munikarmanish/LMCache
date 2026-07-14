#!/usr/bin/env python
"""Plot Sweep B (cross-node dedup + fan-out): CXL vs NIXL under anti_affinity.

Sweep B pins ONE working set and drives the tester through the router under
anti_affinity, which routes each repeated prompt AWAY from the node that cached
it — forcing the other node to serve it. The question is HOW the forced-remote
node serves it:

  - CXL: reads the one shared CXL copy directly (dedup) -> L2 hit, no recompute.
  - NIXL: must RDMA-fetch from the peer, and chunks the peer evicted are gone
    -> recompute. RDMA also serialises per-read, so throughput and TTFT suffer.

Two panels, each comparing the adapters side by side:

  Left  -- tier mix: stacked bars of chunks served from L1 / L2 (CXL or RDMA
           peer) / recompute, per adapter. CXL should be ~all-L2, zero recompute;
           NIXL shows a recompute slice (deleted-on-eviction) it cannot avoid.

  Right -- TTFT: median and p95 bars per adapter, from the tester's per-period
           data at the pinned working set. Quantifies the latency the tier mix
           produces (CXL's shared-load vs NIXL's peer-fetch/recompute).

Reads the newest ``B_<adapter>_anti_affinity_*`` run per adapter under --root:
``tiers.csv`` for the tier mix, ``kvct/sustained_periods_*.csv`` for TTFT.

Usage:
    python plot_sweep_b.py --root results/sweep [--out results/sweep/plots]
"""
# Standard
from pathlib import Path
import argparse
import csv
import re

# Third Party
import matplotlib.pyplot as plt

# Newest run per adapter is the max run-dir name (sortable YYYYMMDD-HHMMSS stamp).
_B_DIR_RE = re.compile(r"^B_(?P<adapter>cxl|nixl)_(?P<strategy>[a-z_]+)_\d{8}-\d{6}$")

# Consistent colours across panels.
_ADAPTER_COLOR = {"cxl": "#2a9d8f", "nixl": "#e76f51"}


def latest_run_per_adapter(root: Path) -> dict[str, Path]:
    """Find the newest Sweep-B run dir per adapter under ``root``.

    Args:
        root: Results root containing ``B_<adapter>_<strategy>_<stamp>`` dirs.

    Returns:
        Mapping adapter -> newest run dir. Adapters with no run are absent.
    """
    latest: dict[str, Path] = {}
    for d in sorted(root.glob("B_*")):
        if not d.is_dir():
            continue
        m = _B_DIR_RE.match(d.name)
        if not m:
            continue
        adapter = m.group("adapter")
        if adapter not in latest or d.name > latest[adapter].name:
            latest[adapter] = d
    return latest


def read_tiers(run_dir: Path) -> dict[str, int]:
    """Read the single tier-mix row from a run's ``tiers.csv``.

    Args:
        run_dir: A Sweep-B run directory.

    Returns:
        Dict with keys ``l1``, ``l2``, ``recompute``, ``total`` (chunk counts).
        ``recompute`` is derived from ``recompute_frac`` * chunk total, since the
        CSV records the hit mix plus the recompute fraction, not a raw miss count.
        Returns an empty dict if the CSV has no data row.
    """
    csv_path = run_dir / "tiers.csv"
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    r = rows[-1]
    l1 = int(r["chunks_l1"])
    l2 = int(r["chunks_l2"])
    hit_total = int(r["chunks_total"])
    frac = float(r["recompute_frac"])
    # chunks_total counts only served (retained) chunks; scale up by the recompute
    # fraction to recover the recompute volume: total_served = hits * 1/(1-frac).
    recompute = round(hit_total * frac / (1.0 - frac)) if frac < 1.0 else 0
    return {"l1": l1, "l2": l2, "recompute": recompute, "total": hit_total + recompute}


def read_ttft(run_dir: Path) -> dict[str, float]:
    """Read median/p95 TTFT from a run's period CSV, averaged across periods.

    Args:
        run_dir: A Sweep-B run directory.

    Returns:
        Dict with ``median`` and ``p95`` TTFT in seconds (mean across periods),
        or an empty dict if no period CSV / rows are present.
    """
    csvs = sorted((run_dir / "kvct").glob("sustained_periods_*.csv"))
    if not csvs:
        return {}
    meds: list[float] = []
    p95s: list[float] = []
    with csvs[0].open() as fh:
        for r in csv.DictReader(fh):
            try:
                meds.append(float(r["median_ttft"]))
                p95s.append(float(r["p95_ttft"]))
            except (KeyError, ValueError):
                continue
    if not meds:
        return {}
    return {"median": sum(meds) / len(meds), "p95": sum(p95s) / len(p95s)}


def plot_sweep_b(root: Path, out_dir: Path) -> None:
    """Render the two-panel CXL-vs-NIXL Sweep-B comparison.

    Args:
        root: Results root with ``B_<adapter>_*`` run dirs.
        out_dir: Directory to write ``sweep_b.png`` into.
    """
    runs = latest_run_per_adapter(root)
    if not runs:
        raise SystemExit(f"no B_<adapter>_*/ run dirs found under {root}")

    adapters = [a for a in ("cxl", "nixl") if a in runs]
    tiers = {a: read_tiers(runs[a]) for a in adapters}
    ttft = {a: read_ttft(runs[a]) for a in adapters}
    for a in adapters:
        print(f"{a}: {runs[a].name}")

    fig, (ax_tier, ax_ttft) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: tier mix (stacked chunk counts) ---------------------------
    x = range(len(adapters))
    l1 = [tiers[a].get("l1", 0) for a in adapters]
    l2 = [tiers[a].get("l2", 0) for a in adapters]
    rec = [tiers[a].get("recompute", 0) for a in adapters]
    ax_tier.bar(x, l1, label="L1 (local DRAM)", color="#8ecae6")
    ax_tier.bar(x, l2, bottom=l1, label="L2 (CXL / RDMA peer)", color="#219ebc")
    ax_tier.bar(x, rec, bottom=[a + b for a, b in zip(l1, l2)],
                label="recompute (deleted on eviction)", color="#e63946")
    ax_tier.set_xticks(list(x))
    ax_tier.set_xticklabels([a.upper() for a in adapters])
    ax_tier.set_ylabel("chunks served (whole run)")
    ax_tier.set_title("Tier mix under anti_affinity\n"
                      "CXL: all shared-L2, 0 recompute; NIXL must recompute")
    ax_tier.legend()
    # Headroom above the tallest bar so the on-bar labels don't collide with the
    # title (the CXL bar reaches the top otherwise).
    tier_max = max((tiers[a].get("total", 0) for a in adapters), default=0)
    if tier_max > 0:
        ax_tier.set_ylim(0, tier_max * 1.12)
    # Annotate the recompute fraction on each bar.
    for i, a in enumerate(adapters):
        t = tiers[a]
        frac = t.get("recompute", 0) / t["total"] if t.get("total") else 0.0
        ax_tier.text(i, t.get("total", 0), f"{frac:.1%} recompute",
                     ha="center", va="bottom", fontsize=9)

    # --- Panel 2: TTFT (median + p95) ---------------------------------------
    width = 0.35
    xs = list(range(len(adapters)))
    med = [ttft[a].get("median", 0) for a in adapters]
    p95 = [ttft[a].get("p95", 0) for a in adapters]
    ax_ttft.bar([i - width / 2 for i in xs], med, width, label="median TTFT",
                color=[_ADAPTER_COLOR[a] for a in adapters])
    ax_ttft.bar([i + width / 2 for i in xs], p95, width, label="p95 TTFT",
                color=[_ADAPTER_COLOR[a] for a in adapters], alpha=0.5)
    ax_ttft.set_xticks(xs)
    ax_ttft.set_xticklabels([a.upper() for a in adapters])
    ax_ttft.set_ylabel("TTFT (s)")
    ax_ttft.set_title("Cross-node TTFT (pinned working set)\n"
                      "CXL shared-load vs NIXL peer-fetch/recompute")
    ax_ttft.legend()
    # Headroom so the value labels above the p95 bars clear the title.
    ttft_max = max(p95 + med, default=0)
    if ttft_max > 0:
        ax_ttft.set_ylim(0, ttft_max * 1.12)
    for i, a in enumerate(adapters):
        ax_ttft.text(i - width / 2, med[i], f"{med[i]:.2f}s",
                     ha="center", va="bottom", fontsize=9)
        ax_ttft.text(i + width / 2, p95[i], f"{p95[i]:.2f}s",
                     ha="center", va="bottom", fontsize=9)

    # Speedup callout if both adapters present.
    if len(adapters) == 2 and med[0] > 0 and med[1] > 0:
        cxl_med = ttft.get("cxl", {}).get("median", 0)
        nixl_med = ttft.get("nixl", {}).get("median", 0)
        if cxl_med > 0 and nixl_med > 0:
            fig.suptitle(
                f"Sweep B — CXL serves cross-node reads "
                f"{nixl_med / cxl_med:.1f}x faster than NIXL "
                f"(median TTFT {cxl_med:.2f}s vs {nixl_med:.2f}s)",
                fontsize=13,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "sweep_b.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="results root with B_<adapter>_*/ run dirs")
    parser.add_argument("--out", default="",
                        help="plot output dir (default: <root>/plots)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_sweep_b(root, out_dir)


if __name__ == "__main__":
    main()
