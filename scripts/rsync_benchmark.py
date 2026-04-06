#!/usr/bin/env python3
# scripts/rsync_benchmark.py  (run on client)
#
# Benchmarks rsync against the full chunks.jsonl dataset concatenated into
# a single file, letting rsync determine its own block boundaries.
#
# Requires rsync_server.py to be running on node1 first.
#
# How it works:
#   1. Concatenates all OLD texts into a single old.txt
#   2. Seeds node1 with old.txt via rsync daemon (not timed)
#   3. Concatenates all NEW texts into new.txt
#   4. Times a single rsync call syncing new.txt -> node1 daemon
#   5. Saves total bytes and total time to a JSON file
#
# Usage:
#   python rsync_benchmark.py --server-host node1 --chunks chunks.jsonl

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path


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


def benchmark(
    chunks_path: Path,
    server_host: str,
    port: int,
) -> dict:
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    n = len(chunks)
    dst = f"rsync://{server_host}:{port}/bench/data.txt"

    old_text = "".join(entry["OLD"] for entry in chunks)
    new_text = "".join(entry["NEW"] for entry in chunks)

    print(f"[rsync_benchmark] dataset: {n} chunks, {len(old_text):,} bytes OLD -> {len(new_text):,} bytes NEW")
    print(f"[rsync_benchmark] target: {dst}")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "data.txt"

        # Step 1: seed server with OLD (not timed)
        print("[rsync_benchmark] seeding server with OLD...")
        src_file.write_text(old_text, encoding="utf-8")
        result = subprocess.run(
            ["rsync", "--checksum", src_file, dst],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to seed server: {result.stderr.strip()}")

        # Step 2: write NEW and run timed rsync
        print("[rsync_benchmark] running timed rsync with NEW...")
        src_file.write_text(new_text, encoding="utf-8")
        t0 = time.perf_counter()
        result = subprocess.run(
            ["rsync", "--checksum", "--stats", src_file, dst],
            capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            raise RuntimeError(f"rsync failed: {result.stderr.strip()}")

        literal, total_sent = _parse_rsync_stats(result.stdout)

    print(f"[rsync_benchmark] done in {elapsed:.3f}s")
    print(f"[rsync_benchmark] literal={literal:,}B  total_sent={total_sent:,}B")

    return {
        "total_time_s":     round(elapsed, 3),
        "residual_bytes":   literal,
        "total_bytes_sent": total_sent,
    }


def save(result: dict, out_path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_path.with_stem(f"{out_path.stem}_{ts}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[rsync_benchmark] results saved to {out}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark rsync against chunks.jsonl (run on node0)")
    ap.add_argument("--chunks", default="chunks.jsonl", help="Path to chunks.jsonl")
    ap.add_argument(
        "--server-host",
        default="node1",
        help="Hostname or IP of the server node running rsync_server.py (default: node1)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8730,
        help="Port the rsync daemon is listening on (default: 8730)",
    )
    ap.add_argument(
        "--out",
        default="data/rsync.json",
        help="Output path (timestamp appended automatically)",
    )
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync is not installed or not on PATH")

    result = benchmark(chunks_path, args.server_host, args.port)
    save(result, Path(args.out))


if __name__ == "__main__":
    main()