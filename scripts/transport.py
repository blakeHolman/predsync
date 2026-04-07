#!/usr/bin/env python3
# scripts/transport.py

# Handles all network communication between client and server.

import asyncio
import hashlib
import json
import socket
import zlib
from typing import Callable, Awaitable

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, Request
from pydantic import BaseModel

from residuals import apply_residual


# ----- Constants -----

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765
DEFAULT_CLIENT_PORT = 8766
TIMEOUT             = 120

ERR_RULES_MISMATCH  = "rules_mismatch"
ERR_CHECKSUM_FAIL   = "checksum_fail"
ERR_CHUNK_NOT_FOUND = "chunk_not_found"


# ----- FastAPI App -----

class RegisterRequest(BaseModel):
    client_id: str
    client_port: int

class RegisterResponse(BaseModel):
    ok:            bool
    rules_hash:    str
    rules_score:   float
    rules_version: int

class VerifyRulesRequest(BaseModel):
    client_id:     str | None = None   # set when client is pushing
    rules_hash:    str
    rules_score:   float
    rules_version: int
    rules:         list | None = None   # only present when client is pushing

class RulesResponse(BaseModel):
    ok:            bool
    rules_hash:    str | None  = None
    rules_score:   float | None = None
    rules_version: int | None  = None
    rules:         list | None = None
    needs_rules:   bool        = False  # server signals client should push

class PrepareRequest(BaseModel):
    chunk_id:   str
    rules_hash: str
    client_id:  str | None = None   # only needed for client→server (pending key)

class PrepareResponse(BaseModel):
    ok:    bool
    error: str | None = None   # rules_mismatch, chunk_not_found

class SyncResponse(BaseModel):
    ok:    bool
    error: str | None = None   # checksum_fail, chunk_not_found, inference_timeout

# ----- Helper Functions -----

def generate_client_id(port):
    # sha256(hostname + port)[:16]
    raw = socket.gethostname() + str(port)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def hash_rules(rules):
    # sha256 of json.dumps(rules) — no sort_keys, order matters
    serialized = json.dumps(rules)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

def checksum(text):
    # crc32 of utf-8 encoded text, unsigned 32-bit
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF

    
# ----- Client Transport -----

class ClientTransport:
    def __init__(self, server_host=DEFAULT_SERVER_HOST, server_port=DEFAULT_SERVER_PORT, client_port=DEFAULT_CLIENT_PORT):
        self.server_host = server_host
        self.server_port = server_port
        self.client_port = client_port
        self.client_id   = generate_client_id(client_port)
        self._base_url    = f"http://{server_host}:{server_port}"

    async def register(self) -> RegisterResponse:
        """POST /register with client_id and client_port.
        Returns rules_hash, rules_score, rules_version from server."""
        payload = RegisterRequest(
            client_id=self.client_id,
            client_port=self.client_port,
        )
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/register",
                json=payload.model_dump(),
            )
            resp.raise_for_status()
            return RegisterResponse(**resp.json())
        

    async def verify_rules(
        self,
        rules_hash:    str,
        rules_score:   float,
        rules_version: int,
        rules:         list | None = None,
        client_id:     str | None = None,
    ) -> RulesResponse:
        """POST /rules with client_id, rules_hash, rules_score, rules_version,
        and optionally rules (if pushing).
        Returns ok and optionally needs_rules (if server wants client to push)."""
        payload = VerifyRulesRequest(
            client_id=client_id or self.client_id,
            rules_hash=rules_hash,
            rules_score=rules_score,
            rules_version=rules_version,
            rules=rules,
        )
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/rules",
                json=payload.model_dump(),
            )
            resp.raise_for_status()
            return RulesResponse(**resp.json())
        

    async def prepare_sync(self, chunk_id: str, rules_hash: str) -> PrepareResponse:
        """POST /prepare with client_id, chunk_id, and rules_hash.
        Returns ok or error (rules_mismatch or chunk_not_found)."""
        payload = PrepareRequest(
            client_id=self.client_id,
            chunk_id=chunk_id,
            rules_hash=rules_hash,
        )
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/prepare",
                json=payload.model_dump(),
            )
            resp.raise_for_status()
            return PrepareResponse(**resp.json())
        

    async def sync(
        self,
        chunk_id:   str,
        residual:   bytes,
        new_text:   str,
    ) -> SyncResponse:
        """POST /sync with residual bytes in the body and checksum of new_text in headers.

        Caller must have already run:
            predicted = predict(old)
            residual  = get_residual(new_text, predicted)

        The server mirrors predict(old) in its own background task (started by /prepare),
        then applies apply_residual(predicted, residual) to reconstruct new_text and
        verifies the checksum. Only the small residual delta is sent over the wire.
        """
        crc = checksum(new_text)

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{self._base_url}/sync",
                content=residual,
                headers={
                    "x-client-id":  self.client_id,
                    "x-chunk-id":   chunk_id,
                    "x-checksum":   str(crc),
                    "Content-Type": "application/octet-stream",
                },
            )
            resp.raise_for_status()
            return SyncResponse(**resp.json())
        

# ----- Client Listener -----

class ClientListener:
    """
    Runs on each client to receive inbound prepare/sync calls from the server.
    Mirrors the server-side prepare/sync flow but in reverse direction.
    """
    def __init__(self, rules_hash: str = ""):
        self.rules_hash = rules_hash

        # Pending inference futures: chunk_id -> asyncio.Future[str]
        # Keyed on chunk_id only — only one server ever pushes to this client
        self._pending: dict[str, asyncio.Future] = {}

        # Callbacks injected by client.py
        # on_prepare(chunk_id) -> predicted_text
        self.on_prepare:       Callable[[str], Awaitable[str]] | None = None
        # on_write_new(chunk_id, new_text) -> None
        self.on_write_new:     Callable[[str, str], Awaitable[None]] | None = None
        # on_rules_updated(rules, rules_hash, rules_score, rules_version) -> None
        self.on_rules_updated: Callable[[list, str, float, int], Awaitable[None]] | None = None


# ----- Server Transport -----

class ServerTransport:
    def __init__(self, host: str = DEFAULT_SERVER_HOST, port: int = DEFAULT_SERVER_PORT):
        self.host = host
        self.port = port
        self.app  = FastAPI()

        # Client registry: client_id -> {host, port}
        self._clients: dict[str, dict] = {}

        # Current best rules state (set by server.py after init_prefix_kv)
        self.rules:         list | None = None
        self.rules_hash:    str         = ""
        self.rules_score:   float       = 0.0
        self.rules_version: int         = 0

        # Pending inference futures: (chunk_id, client_id) -> asyncio.Future[str]
        # Populated by /prepare background task, awaited by /sync
        self._pending: dict[tuple[str, str], asyncio.Future] = {}

        # Callbacks injected by server.py — keeps transport logic-free
        # on_prepare(chunk_id) -> predicted_text
        self.on_prepare:       Callable[[str], Awaitable[str]] | None = None
        # on_write_new(chunk_id, new_text) -> None
        self.on_write_new:     Callable[[str, str], Awaitable[None]] | None = None
        # on_rules_updated(rules, rules_hash, rules_score, rules_version) -> None
        self.on_rules_updated: Callable[[list, str, float, int], Awaitable[None]] | None = None

        # Register endpoints
        register_routes(self.app, self)

    def register_client(self, client_id: str, host: str, client_port: int) -> None:
        """Add or update client in internal registry."""
        self._clients[client_id] = {"host": host, "port": client_port}

    async def push_update(
        self,
        chunk_id:      str,
        residual:      bytes,
        checksum:      int,
        rules_hash:    str,
        rules_score:   float,
        rules_version: int,
        exclude_id:    str | None = None,
    ) -> None:
        """Broadcast a residual delta to all registered clients except the sender.

        Mirrors the client→server prepare/sync flow in reverse:
            1. POST /prepare to all target clients concurrently
               → each client starts predict(old) in background
            2. POST /sync (residual) to all clients that acked prepare
               → each client applies residual against its prediction
        """
        targets = {
            cid: info
            for cid, info in self._clients.items()
            if cid != exclude_id
        }
        if not targets:
            return

        # Step 1: /prepare all target clients concurrently
        prepare_payload = PrepareRequest(chunk_id=chunk_id, rules_hash=rules_hash)
        async with httpx.AsyncClient(timeout=TIMEOUT) as http:
            prepare_results = await asyncio.gather(*[
                http.post(
                    f"http://{info['host']}:{info['port']}/prepare",
                    json=prepare_payload.model_dump(),
                )
                for info in targets.values()
            ], return_exceptions=True)

        # Only proceed with clients that acked prepare successfully
        ready = [
            cid
            for cid, result in zip(targets, prepare_results)
            if not isinstance(result, Exception)
            and PrepareResponse(**result.json()).ok
        ]
        if not ready:
            print(f"[push_update] no clients ready for chunk={chunk_id}")
            return

        # Step 2: /sync residual to all ready clients concurrently
        async with httpx.AsyncClient(timeout=TIMEOUT) as http:
            sync_results = await asyncio.gather(*[
                http.post(
                    f"http://{targets[cid]['host']}:{targets[cid]['port']}/sync",
                    content=residual,
                    headers={
                        "x-chunk-id":   chunk_id,
                        "x-checksum":   str(checksum),
                        "Content-Type": "application/octet-stream",
                    },
                )
                for cid in ready
            ], return_exceptions=True)

        for cid, result in zip(ready, sync_results):
            if isinstance(result, Exception):
                print(f"[push_update] sync failed for {cid}: {result}")
            elif not SyncResponse(**result.json()).ok:
                print(f"[push_update] sync rejected by {cid}: {result.json().get('error')}")

    async def push_rules(
        self,
        rules:         list,
        rules_hash:    str,
        rules_score:   float,
        rules_version: int,
        exclude_id:    str | None = None,
    ) -> None:
        """Broadcast updated rules to all registered clients except the sender."""
        targets = {
            cid: info
            for cid, info in self._clients.items()
            if cid != exclude_id
        }
        if not targets:
            return

        payload = VerifyRulesRequest(
            rules_hash=rules_hash,
            rules_score=rules_score,
            rules_version=rules_version,
            rules=rules,
        )
        async with httpx.AsyncClient(timeout=TIMEOUT) as http:
            results = await asyncio.gather(*[
                http.post(
                    f"http://{info['host']}:{info['port']}/rules",
                    json=payload.model_dump(),
                )
                for info in targets.values()
            ], return_exceptions=True)

        for cid, result in zip(targets, results):
            if isinstance(result, Exception):
                print(f"[push_rules] failed for {cid}: {result}")

    @property
    def client_count(self) -> int:
        """Return number of registered clients."""
        return len(self._clients)


# ----- FastAPI routes -----

def register_routes(app: FastAPI, server_transport: ServerTransport) -> None:
    """
    Attach all server routes to the FastAPI app.
    Called once in server.py at startup.
    Business logic callbacks injected from server.py — transport stays clean.
    """

    @app.post("/register", response_model=RegisterResponse)
    async def register(req: RegisterRequest, request: Request) -> RegisterResponse:
        """Register a new client. Returns server's current rules metadata."""
        client_host = request.client.host
        server_transport.register_client(req.client_id, client_host, req.client_port)
        print(f"[register] client={req.client_id} host={client_host} port={req.client_port}")
        return RegisterResponse(
            ok=True,
            rules_hash=server_transport.rules_hash,
            rules_score=server_transport.rules_score,
            rules_version=server_transport.rules_version,
        )

    @app.post("/rules", response_model=RulesResponse)
    async def rules(req: VerifyRulesRequest) -> RulesResponse:
        """Rules negotiation endpoint.
        - Client pushing rules (req.rules present): compare scores, adopt if better.
        - Client pulling rules (req.rules absent):  compare hashes, return rules or
          signal needs_rules if client is ahead."""
        if req.rules is not None:
            # Client is pushing — adopt if score is strictly better
            if req.rules_score > server_transport.rules_score or not server_transport.rules:
                server_transport.rules         = req.rules
                server_transport.rules_hash    = hash_rules(req.rules)
                server_transport.rules_score   = req.rules_score
                server_transport.rules_version = req.rules_version
                print(f"[rules] Adopted rules from client "
                      f"(score={req.rules_score:.4f} v{req.rules_version})")
                if server_transport.on_rules_updated is not None:
                    await server_transport.on_rules_updated(
                        server_transport.rules,
                        server_transport.rules_hash,
                        server_transport.rules_score,
                        server_transport.rules_version,
                        req.client_id,
                    )
            return RulesResponse(
                ok=True,
                rules_hash=server_transport.rules_hash,
                rules_score=server_transport.rules_score,
                rules_version=server_transport.rules_version,
            )
        else:
            # Client is pulling — check if hashes match
            if req.rules_hash == server_transport.rules_hash:
                return RulesResponse(ok=True, rules_hash=server_transport.rules_hash)

            # Client has a higher-scoring version the server doesn't know about
            if req.rules_score > server_transport.rules_score:
                return RulesResponse(ok=False, needs_rules=True)

            # Server has better rules — send them down
            return RulesResponse(
                ok=False,
                rules_hash=server_transport.rules_hash,
                rules_score=server_transport.rules_score,
                rules_version=server_transport.rules_version,
                rules=server_transport.rules,
            )

    @app.post("/prepare", response_model=PrepareResponse)
    async def prepare(
        req: PrepareRequest,
        background_tasks: BackgroundTasks,
    ) -> PrepareResponse:
        """Verify rules hash and kick off background inference.
        Returns immediately; /sync will block until inference is ready."""
        if req.rules_hash != server_transport.rules_hash:
            return PrepareResponse(ok=False, error=ERR_RULES_MISMATCH)

        if server_transport.on_prepare is None:
            return PrepareResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)

        # Create a Future that /sync will await, keyed on (chunk_id, client_id)
        # to prevent collisions when multiple clients sync the same chunk
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        server_transport._pending[(req.chunk_id, req.client_id)] = future

        async def _run_inference(chunk_id: str) -> None:
            try:
                predicted = await server_transport.on_prepare(chunk_id)
                future.set_result(predicted)
            except FileNotFoundError:
                future.set_exception(
                    FileNotFoundError(f"Chunk not found: {chunk_id}")
                )
            except Exception as exc:
                future.set_exception(exc)

        background_tasks.add_task(_run_inference, req.chunk_id)
        print(f"[prepare] chunk={req.chunk_id} client={req.client_id} inference queued")
        return PrepareResponse(ok=True)

    @app.post("/sync", response_model=SyncResponse)
    async def sync(
        request:       Request,
        x_client_id:   str = Header(...),
        x_chunk_id:    str = Header(...),
        x_checksum:    int = Header(...),
    ) -> SyncResponse:
        """Receive residual bytes from client, reconstruct new_text, verify checksum, write, fan-out.

        Flow:
            1. Read raw residual bytes from body
            2. Await server's predicted text (set by /prepare background task)
            3. apply_residual(predicted, residual)  →  new_text
            4. Verify checksum(new_text) == x_checksum
            5. Write new_text, broadcast to other clients
        """
        # Read raw residual bytes from body
        residual = await request.body()

        # Wait for background inference from /prepare to finish
        future = server_transport._pending.get((x_chunk_id, x_client_id))
        if future is None:
            return SyncResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)

        try:
            predicted = await asyncio.wait_for(
                asyncio.shield(future), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            return SyncResponse(ok=False, error="inference_timeout")
        except FileNotFoundError:
            return SyncResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)
        finally:
            server_transport._pending.pop((x_chunk_id, x_client_id), None)

        # Reconstruct new_text from server's own predicted + client's residual delta
        new_text = apply_residual(predicted, residual).decode("utf-8")

        # Verify checksum against reconstructed text
        if checksum(new_text) != x_checksum:
            return SyncResponse(ok=False, error=ERR_CHECKSUM_FAIL)

        # Write accepted new_text to disk
        if server_transport.on_write_new is not None:
            await server_transport.on_write_new(x_chunk_id, new_text)

        print(f"[sync] chunk={x_chunk_id} from={x_client_id} accepted")

        # Broadcast residual delta to all other clients asynchronously
        asyncio.create_task(
            server_transport.push_update(
                chunk_id=x_chunk_id,
                residual=residual,
                checksum=x_checksum,
                rules_hash=server_transport.rules_hash,
                rules_score=server_transport.rules_score,
                rules_version=server_transport.rules_version,
                exclude_id=x_client_id,
            )
        )
        return SyncResponse(ok=True)


def register_client_routes(app: FastAPI, listener: ClientListener) -> None:
    """
    Attach inbound routes to the client's FastAPI app.
    Called once in client.py at startup.
    Mirrors register_routes but in the server→client direction.
    """

    @app.post("/rules", response_model=RulesResponse)
    async def rules(req: VerifyRulesRequest) -> RulesResponse:
        """Server pushes better rules to this client mid-session."""
        if req.rules is not None and req.rules_score > 0.0:
            listener.rules_hash = hash_rules(req.rules)
            print(f"[client rules] adopted server rules v{req.rules_version} "
                  f"score={req.rules_score:.4f}")
            if listener.on_rules_updated is not None:
                await listener.on_rules_updated(
                    req.rules,
                    listener.rules_hash,
                    req.rules_score,
                    req.rules_version,
                )
        return RulesResponse(ok=True, rules_hash=listener.rules_hash)

    @app.post("/prepare", response_model=PrepareResponse)
    async def prepare(
        req: PrepareRequest,
        background_tasks: BackgroundTasks,
    ) -> PrepareResponse:
        """Server signals client to start predict(old) in background.
        Returns immediately; /sync will block until inference is ready."""
        if req.rules_hash != listener.rules_hash:
            return PrepareResponse(ok=False, error=ERR_RULES_MISMATCH)

        if listener.on_prepare is None:
            return PrepareResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        listener._pending[req.chunk_id] = future

        async def _run_inference(chunk_id: str) -> None:
            try:
                predicted = await listener.on_prepare(chunk_id)
                future.set_result(predicted)
            except FileNotFoundError:
                future.set_exception(FileNotFoundError(f"Chunk not found: {chunk_id}"))
            except Exception as exc:
                future.set_exception(exc)

        background_tasks.add_task(_run_inference, req.chunk_id)
        print(f"[client prepare] chunk={req.chunk_id} inference queued")
        return PrepareResponse(ok=True)

    @app.post("/sync", response_model=SyncResponse)
    async def sync(
        request:     Request,
        x_chunk_id:  str = Header(...),
        x_checksum:  int = Header(...),
    ) -> SyncResponse:
        """Server sends residual bytes; client applies them against its prediction.

        Flow:
            1. Await predicted text (set by /prepare background task)
            2. apply_residual(predicted, residual)  →  new_text
            3. Verify checksum(new_text) == x_checksum
            4. Write new_text locally
        """
        residual = await request.body()

        future = listener._pending.get(x_chunk_id)
        if future is None:
            return SyncResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)

        try:
            predicted = await asyncio.wait_for(
                asyncio.shield(future), timeout=TIMEOUT
            )
        except asyncio.TimeoutError:
            return SyncResponse(ok=False, error="inference_timeout")
        except FileNotFoundError:
            return SyncResponse(ok=False, error=ERR_CHUNK_NOT_FOUND)
        finally:
            listener._pending.pop(x_chunk_id, None)

        new_text = apply_residual(predicted, residual).decode("utf-8")

        if checksum(new_text) != x_checksum:
            return SyncResponse(ok=False, error=ERR_CHECKSUM_FAIL)

        if listener.on_write_new is not None:
            await listener.on_write_new(x_chunk_id, new_text)

        print(f"[client sync] chunk={x_chunk_id} written locally")
        return SyncResponse(ok=True)