#!/usr/bin/env python3
# scripts/test.py

# Simulates document edits by randomly selecting chunks from chunks.jsonl
# and POSTing them to the running client's /process_chunk endpoint.
#
# Run from command line:
#   python test.py --n 5 --client-host 10.0.0.2 --client-port 8766
#
# Uses data/versions.json to determine which chunks have already been
# updated — chunks with a version > 0 are excluded from selection.
# Use --reset to clear versions.json and start fresh.

import argparse
import json
import random
import sys
from pathlib import Path

import httpx


# ----- Constants -----

CHUNKS_FILE   = Path("data/chunks.jsonl")
VERSIONS_FILE = Path("data/versions.json")
TIMEOUT       = 300  # generous — inference can be slow


# ----- Helpers -----

def _load_updated() -> set:
    """Return set of chunk_ids that have already been updated (version > 0)."""
    if not VERSIONS_FILE.exists():
        return set()
    try:
        data = json.loads(VERSIONS_FILE.read_text(encoding="utf-8"))
        return {cid for cid, v in data.get("chunks", {}).items() if v > 0}
    except (json.JSONDecodeError, OSError):
        return set()


def _reset(chunks: list[dict]):
    """Reset versions.json and rewrite all chunk .txt files back to OLD text."""
    if VERSIONS_FILE.exists():
        VERSIONS_FILE.write_text(json.dumps({"chunks": {}}, indent=2), encoding="utf-8")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk_path = chunks_dir / f"{chunk['chunk_id']}.txt"
        chunk_path.write_text(chunk["old"], encoding="utf-8")
    print(f"[test] reset — versions.json cleared, {len(chunks)} chunk(s) restored to OLD text")


def _load_chunks() -> list[dict]:
    """Load all records from chunks.jsonl."""
    if not CHUNKS_FILE.exists():
        print(f"[test] chunks file not found: {CHUNKS_FILE}")
        sys.exit(1)

    chunks = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = rec.get("chunk_id")
            old      = rec.get("OLD")
            new      = rec.get("NEW")
            if chunk_id and old and new:
                chunks.append({"chunk_id": chunk_id, "old": old, "new": new})
    return chunks


# ----- CLI -----

def _parse_args():
    ap = argparse.ArgumentParser(description="Simulate chunk edits against a running client")
    ap.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of chunks to update this run (default: 1)",
    )
    ap.add_argument(
        "--client-host",
        default="127.0.0.1",
        help="IP of the client machine (default: 127.0.0.1)",
    )
    ap.add_argument(
        "--client-port",
        type=int,
        default=8766,
        help="Port the client is listening on (default: 8766)",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Reset versions.json and restore all chunk files to OLD text",
    )
    return ap.parse_args()


def main():
    args = _parse_args()

    chunks = _load_chunks()

    if args.reset:
        _reset(chunks)
        return

    updated = _load_updated()

    # Filter to chunks not yet updated
    pending = [c for c in chunks if c["chunk_id"] not in updated]

    if not pending:
        print("[test] all chunks have already been updated — use --reset to start over")
        return

    # Clamp n to available pool
    n = min(args.n, len(pending))
    if n < args.n:
        print(f"[test] only {len(pending)} chunk(s) remaining — sending {n}")

    selected = random.sample(pending, n)
    base_url = f"http://{args.client_host}:{args.client_port}"

    print(f"[test] sending {n} chunk(s) to {base_url}")

    for chunk in selected:
        chunk_id = chunk["chunk_id"]
        print(f"[test] → {chunk_id}")
        try:
            with httpx.Client(timeout=TIMEOUT) as http:
                resp = http.post(
                    f"{base_url}/process_chunk",
                    json={
                        "chunk_id": chunk["chunk_id"],
                        "old":      chunk["old"],
                        "new":      chunk["new"],
                    },
                )
            resp.raise_for_status()
            result = resp.json()
            if result.get("ok"):
                print(f"[test] ✓ {chunk_id} committed")
            else:
                print(f"[test] ✗ {chunk_id} rejected: {result}")
        except httpx.HTTPError as e:
            print(f"[test] ✗ {chunk_id} failed: {e}")

    # Report final state from versions.json
    updated = _load_updated()
    print(f"[test] done — {len(updated)}/{len(chunks)} chunk(s) updated total")


if __name__ == "__main__":
    main()