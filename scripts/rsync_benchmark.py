#!/usr/bin/env python3
# scripts/rsync_benchmark.py
#
# Simulates rsync-based sync over the chunks.jsonl dataset and produces a
# metrics JSON file in the same format as predsync's metrics.py output.
#
# Analogues to ChunkMetrics fields:
#   residual_bytes   -> rsync "Literal data" (bytes that couldn't be matched)
#   total_bytes_sent -> rsync "Total bytes sent"
#   total_time_s     -> wall time for the rsync call
#   inference_time_s -> always 0.0 (rsync has no inference step)
#   network_time_s   -> same as total_time_s (rsync is pure transfer)
#
# Usage:
#   python rsync_benchmark.py [--chunks chunks.jsonl] [--out data/rsync_TIMESTAMP.json]

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ChunkMetrics:
    chunk_id:           str   = ""
    total_time_s:       float = 0.0
    inference_time_s:   float = 0.0
    network_time_s:     float = 0.0
    residual_bytes:     int   = 0
    total_bytes_sent:   int   = 0
    rules_updated:      bool  = False
    rules_score_before: float = 0.0
    rules_score_after:  float = 0.0
    rules_version:      int   = 0


def _parse_rsync_stats(output: str) -> tuple[int, int]:
    """Return (literal_data_bytes, total_bytes_sent) from rsync --stats output."""
    literal = 0
    total_sent = 0
    for line in output.splitlines():
        m = re.search(r"Literal data:\s+([\d,]+)", line)
        if m:
            literal = int(m.group(1).replace(",", ""))
        m = re.search(r"Total bytes sent:\s+([\d,]+)", line)
        if m:
            total_sent = int(m.group(1).replace(",", ""))
    return literal, total_sent


def benchmark(chunks_path: Path) -> list[ChunkMetrics]:
    records: list[ChunkMetrics] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "src.txt"
        dst_file = Path(tmpdir) / "dst.txt"

        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                chunk_id = entry["chunk_id"]
                old_text = entry["OLD"]
                new_text = entry["NEW"]

                # Seed dst with OLD so rsync has a basis to diff against
                dst_file.write_text(old_text, encoding="utf-8")
                src_file.write_text(new_text, encoding="utf-8")

                t0 = time.perf_counter()
                result = subprocess.run(
                    [
                        "rsync",
                        "--checksum",   # force content-based diff (no mtime shortcut)
                        "--stats",
                        str(src_file),
                        str(dst_file),
                    ],
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - t0

                if result.returncode != 0:
                    print(f"[rsync] ERROR on {chunk_id}: {result.stderr.strip()}")
                    continue

                literal, total_sent = _parse_rsync_stats(result.stdout)

                m = ChunkMetrics(
                    chunk_id=chunk_id,
                    total_time_s=round(elapsed, 6),
                    inference_time_s=0.0,
                    network_time_s=round(elapsed, 6),
                    residual_bytes=literal,
                    total_bytes_sent=total_sent,
                )
                records.append(m)
                print(
                    f"[rsync] {chunk_id}  literal={literal:,}B  "
                    f"total_sent={total_sent:,}B  time={elapsed:.4f}s"
                )

    return records


def save(records: list[ChunkMetrics], out_path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_path.with_stem(f"{out_path.stem}_{ts}")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(records)
    summary = {}
    if n:
        summary = {
            "chunks":           n,
            "total_time_s":     round(sum(r.total_time_s     for r in records) / n, 3),
            "inference_time_s": round(sum(r.inference_time_s for r in records) / n, 3),
            "network_time_s":   round(sum(r.network_time_s   for r in records) / n, 3),
            "residual_bytes":   sum(r.residual_bytes   for r in records) // n,
            "total_bytes_sent": sum(r.total_bytes_sent for r in records) // n,
        }

    data = {
        "summary": summary,
        "chunks":  [asdict(r) for r in records],
    }

    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[rsync_benchmark] {n} record(s) saved to {out}")
    if summary:
        print(
            f"[rsync_benchmark] averages — "
            f"total={summary['total_time_s']:.3f}s  "
            f"literal={summary['residual_bytes']:,}B  "
            f"total_sent={summary['total_bytes_sent']:,}B"
        )


def main():
    ap = argparse.ArgumentParser(description="Benchmark rsync against chunks.jsonl")
    ap.add_argument(
        "--chunks",
        default="data/chunks.jsonl",
        help="Path to chunks.jsonl (default: data/chunks.jsonl)",
    )
    ap.add_argument(
        "--out",
        default="data/rsync.json",
        help="Output metrics JSON path (timestamp appended automatically)",
    )
    ap.add_argument(
        "--stop-after",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N chunks (for quick validation)",
    )
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is not installed or not on PATH")

    print(f"[rsync_benchmark] reading {chunks_path}")

    # Apply --stop-after by slicing the JSONL
    if args.stop_after is not None:
        import tempfile as _tf
        lines = chunks_path.read_text(encoding="utf-8").splitlines()
        lines = lines[: args.stop_after]
        tmp = Path(_tf.mktemp(suffix=".jsonl"))
        tmp.write_text("\n".join(lines), encoding="utf-8")
        chunks_path = tmp
        print(f"[rsync_benchmark] --stop-after {args.stop_after}: using first {args.stop_after} chunk(s)")

    records = benchmark(chunks_path)
    save(records, Path(args.out))


if __name__ == "__main__":
    main()