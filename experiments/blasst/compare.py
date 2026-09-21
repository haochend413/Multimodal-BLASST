"""Summarize run_sweep.sh results into one table (printed + results.csv)."""

import csv
import json
import sys
from pathlib import Path


def longbench_score(run_dir: Path):
    """Mean of LongBench subgroup scores x100, same as trtllm-eval reports."""
    path = run_dir / "samples_longbench.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    subgroups = results.get("group_subtasks", {}).get("longbench", [])
    scores = [
        results["groups"][name]["score,none"]
        for name in subgroups
        if isinstance(results.get("groups", {}).get(name, {}).get("score,none"), (int, float))
    ]
    return 100 * sum(scores) / len(scores) if scores else None


def bench_metrics(run_dir: Path):
    path = run_dir / "report.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    perf = report.get("performance", {})
    streaming = report.get("streaming_metrics", {})
    return {
        "ttft_ms": streaming.get("avg_ttft_ms"),
        "tpot_ms": streaming.get("avg_tpot_ms"),
        "out_tok_s": perf.get("system_output_throughput_tok_s"),
        "req_latency_ms": perf.get("avg_request_latency_ms"),
    }


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for run_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        rows.append({"run": run_dir.name, "longbench": longbench_score(run_dir),
                     **bench_metrics(run_dir)})
    if not rows:
        sys.exit(f"No runs found under {out}")

    dense = next((r for r in rows if r["run"] == "dense"), None)
    for row in rows:
        if dense is None:
            break
        if row["longbench"] is not None and dense["longbench"] is not None:
            row["acc_delta"] = row["longbench"] - dense["longbench"]
        for key, speedup_key in (("ttft_ms", "ttft_speedup"), ("tpot_ms", "tpot_speedup")):
            if row.get(key) and dense.get(key):
                row[speedup_key] = dense[key] / row[key]
        if row.get("out_tok_s") and dense.get("out_tok_s"):
            row["tput_speedup"] = row["out_tok_s"] / dense["out_tok_s"]

    columns = ["run", "longbench", "acc_delta", "ttft_ms", "ttft_speedup", "tpot_ms",
               "tpot_speedup", "out_tok_s", "tput_speedup", "req_latency_ms"]
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for row in rows:
        print(" | ".join(row["run"] if c == "run" else fmt(row.get(c)) for c in columns))

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out / 'results.csv'}")


if __name__ == "__main__":
    main()
