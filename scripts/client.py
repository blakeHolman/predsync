#!/usr/bin/env python3
# scripts/client.py
 
# Client node in the distributed pipeline.
#
# Run from command line:
#   python client.py --server-host 10.0.0.1 --server-port 8765 --client-port 8766
#
# server-host and server-port point to the server machine.
# client-port is the port this machine listens on for inbound server pushes.
# If client-port is omitted, it defaults to DEFAULT_CLIENT_PORT.
# The client's own IP is detected automatically from the machine's hostname.
#
# test.py simulates edits by POSTing to /process_chunk on this client.
 
import argparse
import asyncio
import json
import socket
import time
from pathlib import Path
 
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
 
import initialize
import metrics as mx
from transport import (
    ClientTransport,
    ClientListener,
    PrepareRequest,
    VerifyRulesRequest,
    register_client_routes,
    DEFAULT_CLIENT_PORT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)
from residuals import get_residual
from pick_best_example import check, save_rules, load_rules, update_from_server, bump_version
import predict_new
 
 
# ----- Constants -----
 
CHUNKS_DIR = Path("data/chunks")
 
 
# ----- Module-level state -----
 
transport = None
listener  = ClientListener()
_app      = FastAPI()
_server   = None
 
_metrics_enabled: bool = False
_metrics_path: str = "data/metrics.json"

register_client_routes(_app, listener)
 
 
class ProcessChunkRequest(BaseModel):
    chunk_id: str
    old:      str
    new:      str
 
@_app.post("/process_chunk")
async def _process_chunk_endpoint(req: ProcessChunkRequest):
    """Called by test.py to simulate a chunk edit."""
    await process_chunk(req.chunk_id, req.old, req.new)
    return {"ok": True}
 
 
# ----- Startup / shutdown -----
 
async def start(
    server_host: str = DEFAULT_SERVER_HOST,
    server_port: int = DEFAULT_SERVER_PORT,
    client_port: int = DEFAULT_CLIENT_PORT,
):
    """Register with server, negotiate rules, start inbound listener."""
    global transport, _server
 
    transport = ClientTransport(server_host, server_port, client_port)
    listener.on_prepare       = _on_prepare
    listener.on_write_new     = _on_write_new
    listener.on_rules_updated = _on_rules_updated
 
    reg = await transport.register()
    print(f"[client] registered — server rules_hash={reg.rules_hash} "
          f"score={reg.rules_score:.4f} v{reg.rules_version}")
 
    await _negotiate_rules()
 
    config = uvicorn.Config(
        _app,
        host="0.0.0.0",
        port=client_port,
        log_level="warning",
    )
    _server = uvicorn.Server(config)
    asyncio.create_task(_server.serve())
    print(f"[client] listener started on port {client_port}")
 
 
async def stop():
    if _metrics_enabled:
        mx.save(_metrics_path)
    if _server is not None:
        _server.should_exit = True
 
 
# ----- Rules negotiation -----
 
async def _negotiate_rules():
    """
    Sync local rules with server on startup via transport.verify_rules().
 
    Cases:
      - Hashes match            → already in sync, nothing to do
      - Server has better rules → adopt them locally
      - Server needs our rules  → push local rules up
      - Neither has rules yet   → stay empty, extraction on first chunk
    """
    local = load_rules()
 
    resp = await transport.verify_rules(
        rules_hash=local["rules_hash"] or "",
        rules_score=local["rules_score"],
        rules_version=local["rules_version"],
    )
 
    if resp.ok:
        print("[client] rules in sync with server")
 
    elif resp.needs_rules:
        print("[client] server requested our rules — pushing")
        await transport.verify_rules(
            rules_hash=local["rules_hash"] or "",
            rules_score=local["rules_score"],
            rules_version=local["rules_version"],
            rules=local["rules"],
            client_id=transport.client_id,
        )
 
    elif resp.rules is not None:
        update_from_server(
            rules=resp.rules,
            rules_hash=resp.rules_hash,
            rules_score=resp.rules_score,
            rules_version=resp.rules_version,
        )
        local = load_rules()
 
    # Seed predict_new and listener with whatever rules we have now
    if local["rules"]:
        predict_new.PREFIX_TEXT  = local["rules"]
        listener.rules_hash      = local["rules_hash"] or ""
        print(f"[client] PREFIX_TEXT loaded ({len(local['rules'])} rules)")
    else:
        print("[client] no rules yet — will extract on first chunk")
 
 
# ----- Outbound: called by test.py -----
 
async def process_chunk(chunk_id: str, old: str, new: str):
    """
    Handle a locally-known (OLD, NEW) chunk update:
      1. Check if this pair is a better rule exemplar; extract if so
      2. Predict new from old
      3. Compute residual(new, predicted)
      4. transport.prepare_sync → transport.sync
      5. Write new_text locally
    """
    if transport is None:
        raise RuntimeError("[client] not started — call await start() first.")

    if _metrics_enabled:
        m = mx.ChunkMetrics(chunk_id=chunk_id)
        t_total = time.perf_counter()
        t_infer = time.perf_counter()

    # Step 1: check exemplar, extract rules if needed
    _rules, needs_extraction, candidate_score = check(
        old, new, tokenizer=predict_new.TOKENIZER
    )

    if needs_extraction:
        print(f"[client] new best exemplar (score={candidate_score:.4f}) — extracting rules")
        predict_new.init_prefix_kv(old, new)
        prompt = save_rules(predict_new.PREFIX_TEXT, candidate_score)
        listener.rules_hash = prompt["rules_hash"]

        if _metrics_enabled:
            rules_body = json.dumps(VerifyRulesRequest(
                client_id=transport.client_id,
                rules_hash=prompt["rules_hash"],
                rules_score=prompt["rules_score"],
                rules_version=prompt["rules_version"],
                rules=prompt["rules"],
            ).model_dump()).encode()
            m.total_bytes_sent += len(rules_body)
            t_net = time.perf_counter()

        await transport.verify_rules(
            rules_hash=prompt["rules_hash"],
            rules_score=prompt["rules_score"],
            rules_version=prompt["rules_version"],
            rules=prompt["rules"],
            client_id=transport.client_id,
        )

        if _metrics_enabled:
            m.network_time_s += time.perf_counter() - t_net

    elif predict_new.PREFIX_TEXT is None:
        raise RuntimeError(
            f"[client] PREFIX_TEXT is None for chunk={chunk_id} — "
            "no rules available and no extraction triggered."
        )

    # Step 2: predict
    predicted = predict_new.predict(old)

    if _metrics_enabled:
        m.inference_time_s = time.perf_counter() - t_infer

    # Step 3: compute residual
    residual = get_residual(new, predicted)
    print(f"[client] chunk={chunk_id} residual={len(residual)}B "
          f"(predicted={len(predicted)}B new={len(new)}B)")

    # Step 4: prepare + sync via transport
    rules_hash = load_rules()["rules_hash"] or ""

    if _metrics_enabled:
        prepare_body = json.dumps(PrepareRequest(
            client_id=transport.client_id,
            chunk_id=chunk_id,
            rules_hash=rules_hash,
        ).model_dump()).encode()
        m.total_bytes_sent += len(prepare_body)
        t_net = time.perf_counter()

    prep = await transport.prepare_sync(chunk_id, rules_hash)
    if not prep.ok:
        raise RuntimeError(f"[client] /prepare rejected for chunk={chunk_id}: {prep.error}")

    sync_resp = await transport.sync(chunk_id, residual, new)
    if not sync_resp.ok:
        raise RuntimeError(f"[client] /sync failed for chunk={chunk_id}: {sync_resp.error}")

    if _metrics_enabled:
        m.network_time_s   += time.perf_counter() - t_net - m.inference_time_s
        m.residual_bytes    = len(residual)
        m.total_bytes_sent += len(residual)
        m.total_time_s      = time.perf_counter() - t_total
        mx.record(m)

    # Step 5: write locally
    await _on_write_new(chunk_id, new)
    print(f"[client] chunk={chunk_id} committed")
 
 
# ----- Inbound callbacks (server-pushed updates from other clients) -----
 
async def _on_rules_updated(rules: list, rules_hash: str, rules_score: float, rules_version: int):
    """Persist new rules to prompt.json and update predict_new.PREFIX_TEXT."""
    update_from_server(
        rules=rules,
        rules_hash=rules_hash,
        rules_score=rules_score,
        rules_version=rules_version,
    )
    predict_new.PREFIX_TEXT  = rules
    listener.rules_hash      = rules_hash
    print(f"[client] rules updated v{rules_version} hash={rules_hash} score={rules_score:.4f}")
 
 
async def _on_prepare(chunk_id: str) -> str:
    """Read current OLD text from disk and run predict()."""
    chunk_path = CHUNKS_DIR / f"{chunk_id}.txt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk not found: {chunk_path}")
    old = chunk_path.read_text(encoding="utf-8")
    return await asyncio.get_event_loop().run_in_executor(
        None, predict_new.predict, old
    )
 
 
async def _on_write_new(chunk_id: str, new_text: str):
    """Write accepted new_text to disk and increment chunk version."""
    chunk_path = CHUNKS_DIR / f"{chunk_id}.txt"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(new_text, encoding="utf-8")
    version = bump_version(chunk_id)
    print(f"[client] {chunk_id} written (v{version})")
 
 
# ----- CLI -----
 
def _parse_args():
    ap = argparse.ArgumentParser(description="Start the client node")
    ap.add_argument(
        "--server-host",
        default=DEFAULT_SERVER_HOST,
        help=f"IP of the server machine (default: {DEFAULT_SERVER_HOST})",
    )
    ap.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port the server is listening on (default: {DEFAULT_SERVER_PORT})",
    )
    ap.add_argument(
        "--client-port",
        type=int,
        default=DEFAULT_CLIENT_PORT,
        help=f"Port this client listens on for inbound pushes (default: {DEFAULT_CLIENT_PORT})",
    )
    ap.add_argument(
        "--metrics",
        metavar="FILE",
        nargs="?",
        const="data/metrics.json",
        default=None,
        help="Enable metrics collection (default output: data/metrics.json)",
    )
    return ap.parse_args()
 
 
async def _main():
    global _metrics_enabled, _metrics_path
    args = _parse_args()
 
    _metrics_enabled = args.metrics is not None
    _metrics_path    = args.metrics
 
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"[client] machine hostname={hostname} ip={local_ip}")
    print(f"[client] connecting to server at {args.server_host}:{args.server_port}")
    if _metrics_enabled:
        print(f"[client] metrics enabled — will save to {_metrics_path}")
 
    initialize.run_client()
    await start(args.server_host, args.server_port, args.client_port)
 
    print("[client] ready — run test.py to simulate edits")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await stop()
 
 
if __name__ == "__main__":
    asyncio.run(_main())