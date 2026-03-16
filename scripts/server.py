#!/usr/bin/env python3
# scripts/server.py

# Server node in the distributed pipeline.
#
# Run from command line:
#   python server.py --host 0.0.0.0 --port 8765
#
# host is the interface to bind to (default: 0.0.0.0 for all interfaces).
# port is the port to listen on (default: 8765).
# The server's IP is detected automatically from the machine's hostname.
#
# test.py simulates server-initiated updates by POSTing to /process_chunk
# on this server, which then pushes the update to all registered clients.

import argparse
import asyncio
import socket
from pathlib import Path

import uvicorn

import initialize
from transport import (
    ServerTransport,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)
from pick_best_example import load_rules, update_from_server, bump_version
import predict_new


# ----- Constants -----

CHUNKS_DIR = Path("data/chunks")


# ----- Module-level state -----

server_transport = None
_server          = None


# ----- Startup / shutdown -----

async def start(
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
):
    """Initialise transport, inject callbacks, seed rules, start uvicorn."""
    global server_transport, _server

    server_transport = ServerTransport(host, port)
    server_transport.on_prepare       = _on_prepare
    server_transport.on_write_new     = _on_write_new
    server_transport.on_rules_updated = _on_rules_updated

    # Seed rules from prompt.json if available
    prompt = load_rules()
    if prompt["rules"]:
        server_transport.rules         = prompt["rules"]
        server_transport.rules_hash    = prompt["rules_hash"] or ""
        server_transport.rules_score   = prompt["rules_score"]
        server_transport.rules_version = prompt["rules_version"]
        print(f"[server] loaded rules v{prompt['rules_version']} "
              f"hash={prompt['rules_hash']} score={prompt['rules_score']:.4f}")
    else:
        print("[server] no rules yet — waiting for first client registration")

    # Seed predict_new so the server can run predict() for /prepare requests
    if prompt["rules"]:
        predict_new.PREFIX_TEXT = prompt["rules"]

    config = uvicorn.Config(
        server_transport.app,
        host=host,
        port=port,
        log_level="warning",
    )
    _server = uvicorn.Server(config)
    asyncio.create_task(_server.serve())
    print(f"[server] listening on {host}:{port}")


async def stop():
    if _server is not None:
        _server.should_exit = True


# ----- Callbacks injected into ServerTransport -----

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
    print(f"[server] {chunk_id} written (v{version})")


async def _on_rules_updated(rules: list, rules_hash: str, rules_score: float, rules_version: int):
    """Persist new rules to prompt.json, update predict_new, and push to all clients."""
    update_from_server(
        rules=rules,
        rules_hash=rules_hash,
        rules_score=rules_score,
        rules_version=rules_version,
    )
    predict_new.PREFIX_TEXT = rules
    print(f"[server] rules updated v{rules_version} hash={rules_hash} score={rules_score:.4f}")

    await server_transport.push_rules(
        rules=rules,
        rules_hash=rules_hash,
        rules_score=rules_score,
        rules_version=rules_version,
    )


# ----- /process_chunk endpoint (for test.py) -----

from fastapi import FastAPI
from pydantic import BaseModel

_test_app = FastAPI()

class ProcessChunkRequest(BaseModel):
    chunk_id: str
    old:      str
    new:      str

@_test_app.post("/process_chunk")
async def _process_chunk_endpoint(req: ProcessChunkRequest):
    """
    Called by test.py to simulate a server-initiated chunk update.
    Server runs predict(), computes residual, writes locally,
    then pushes the update to all registered clients.
    """
    await process_chunk(req.chunk_id, req.old, req.new)
    return {"ok": True}


async def process_chunk(chunk_id: str, old: str, new: str):
    """
    Handle a server-initiated (OLD, NEW) chunk update:
      1. Predict new from old
      2. Compute residual(new, predicted)
      3. Write new_text locally
      4. Push residual to all registered clients
    """
    if server_transport is None:
        raise RuntimeError("[server] not started — call await start() first.")

    if predict_new.PREFIX_TEXT is None:
        raise RuntimeError(
            f"[server] PREFIX_TEXT is None for chunk={chunk_id} — "
            "no rules available yet."
        )

    # Step 1: predict
    predicted = predict_new.predict(old)

    # Step 2: compute residual
    from residuals import get_residual
    from transport import checksum
    residual = get_residual(new, predicted)
    crc      = checksum(new)
    print(f"[server] chunk={chunk_id} residual={len(residual)}B "
          f"(predicted={len(predicted)}B new={len(new)}B)")

    # Step 3: write locally
    await _on_write_new(chunk_id, new)

    # Step 4: push to all registered clients
    await server_transport.push_update(
        chunk_id=chunk_id,
        residual=residual,
        checksum=crc,
        rules_hash=server_transport.rules_hash,
        rules_score=server_transport.rules_score,
        rules_version=server_transport.rules_version,
    )
    print(f"[server] chunk={chunk_id} pushed to {server_transport.client_count} client(s)")


# ----- CLI -----

def _parse_args():
    ap = argparse.ArgumentParser(description="Start the server node")
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind to (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port to listen on (default: {DEFAULT_SERVER_PORT})",
    )
    ap.add_argument(
        "--test-port",
        type=int,
        default=8764,
        help="Port for the test.py /process_chunk endpoint (default: 8764)",
    )
    return ap.parse_args()


async def _main():
    args = _parse_args()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"[server] machine hostname={hostname} ip={local_ip}")

    initialize.run_server()
    await start(args.host, args.port)

    # Start test endpoint on a separate port so it doesn't conflict with
    # the main server transport app
    test_config = uvicorn.Config(
        _test_app,
        host="0.0.0.0",
        port=args.test_port,
        log_level="warning",
    )
    test_server = uvicorn.Server(test_config)
    asyncio.create_task(test_server.serve())
    print(f"[server] test endpoint on port {args.test_port}")

    print("[server] ready — run test.py to simulate server-pushed updates")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await stop()


if __name__ == "__main__":
    asyncio.run(_main())