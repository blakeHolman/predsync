#!/usr/bin/env python3
# scripts/metrics.py
#
# Lightweight per-chunk metrics collector.
# Imported by client.py and server.py when --metrics is passed.
#
# Each process_chunk() call produces one ChunkMetrics record.
# Call save() on shutdown to write results to a JSON file.

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ChunkMetrics:
    chunk_id:          str   = ""

    # Times (seconds)
    total_time_s:      float = 0.0   # wall time for the full process_chunk() call
    inference_time_s:  float = 0.0   # time spent inside predict_new.predict()
    network_time_s:    float = 0.0   # time spent waiting on HTTP calls

    # Bytes sent over the wire (outbound only, request bodies)
    residual_bytes:    int   = 0     # just the /sync residual payload
    total_bytes_sent:  int   = 0     # residual + rules push + /prepare JSON + any other bodies

    # Bytes sent over the wire (outbound only, request bodies)
    residual_bytes:    int   = 0
    total_bytes_sent:  int   = 0

    # Rules update (only populated if this chunk triggered extraction)
    rules_updated:     bool  = False
    rules_score_before: float = 0.0
    rules_score_after:  float = 0.0
    rules_version:     int   = 0


# ── Global collector ──────────────────────────────────────────────────────────

_records: list[ChunkMetrics] = []


def record(m: ChunkMetrics) -> None:
    """Append a completed ChunkMetrics record."""
    _records.append(m)


def save(path: str | Path) -> None:
    """Write all records plus a summary to a JSON file."""
    out = Path(path)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out.with_stem(f"{out.stem}_{ts}")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(_records)

    summary = {}
    if n:
        summary = {
            "chunks":            n,
            "total_time_s":      round(sum(r.total_time_s      for r in _records) / n, 3),
            "inference_time_s":  round(sum(r.inference_time_s  for r in _records) / n, 3),
            "network_time_s":    round(sum(r.network_time_s    for r in _records) / n, 3),
            "residual_bytes":    sum(r.residual_bytes    for r in _records) // n,
            "total_bytes_sent":  sum(r.total_bytes_sent  for r in _records) // n,
        }

    data = {
        "summary": summary,
        "chunks":  [asdict(r) for r in _records],
    }

    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[metrics] {n} record(s) saved to {out}")
    if summary:
        print(
            f"[metrics] averages — "
            f"total={summary['total_time_s']:.3f}s  "
            f"infer={summary['inference_time_s']:.3f}s  "
            f"net={summary['network_time_s']:.3f}s  "
            f"residual={summary['residual_bytes']:,}B  "
            f"total_sent={summary['total_bytes_sent']:,}B"
        )