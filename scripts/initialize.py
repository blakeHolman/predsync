#!/usr/bin/env python3
# scripts/initialize.py

# Verifies the environment is ready before client or server starts.
# Call initialize.run_client() or initialize.run_server() at the top of each.

import os
import json
import hashlib
import sys
from pathlib import Path


# ----- Constants -----

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

REQUIRED_SCRIPTS = [
    "initialize.py",
    "pick_best_example.py",
    "predict_new.py",
    "residuals.py",
    "transport.py",
    "client.py",
    "server.py",
    "test.py",
]

# ----- Helper functions -----

def _check(condition: bool, message: str):
    """Assert a condition, print status, exit on failure."""
    if condition:
        print(f"  [ok] {message}")
    else:
        print(f"  [FAIL] {message}")
        sys.exit(1)


def _ensure_dir(path: str):
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"  [ok] directory exists: {path}")


def hash_rules(rules: list):
    content = json.dumps(rules)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


# ----- Initialization checks ------

def _verify_directories():
    """
    Ensures required directories exist.
    """
    print("\n[initialize] checking directories ...")
    _ensure_dir("data")
    _ensure_dir("scripts")
    _ensure_dir("data/chunks")


def _verify_prompt():
    """
    Verifies data/prompt.json exists and is well-formed.
    Creates an empty prompt.json if missing.
    The static prompt text is hardcoded in predict_new.py —
    only the extracted rules and their hash are stored here.
    """
    print("\n[initialize] checking data/prompt.json ...")
    path = Path("data/prompt.json")

    if not path.exists():
        print("  [warn] data/prompt.json not found — creating empty")
        payload = {
            "rules_hash": None,
            "rules_score": 0.0,
            "rules_version": 0,
            "rules": []
        }
        path.write_text(json.dumps(payload, indent=2))
        print("  [ok] created data/prompt.json (no rules yet — will populate on first exemplar pick)")
        return

    # Validate structure
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  [FAIL] data/prompt.json is malformed: {e}")
        sys.exit(1)

    _check("rules_hash" in data, "prompt.json has 'rules_hash' field")
    _check("rules"      in data, "prompt.json has 'rules' field")

    # If rules are present, verify hash matches
    if data["rules"]:
        expected = hash_rules(data["rules"])
        if data["rules_hash"] != expected:
            print("  [warn] rules_hash mismatch — recomputing from rules")
            data["rules_hash"] = expected
            path.write_text(json.dumps(data, indent=2))
        print(f"  [ok] data/prompt.json ({len(data['rules'])} rules, hash={data['rules_hash']})")
    else:
        print("  [ok] data/prompt.json (empty — awaiting first exemplar)")


def _verify_versions():
    """
    Verifies data/versions.json exists and is well-formed.
    Creates an empty versions.json if missing.
    """
    print("\n[initialize] checking data/versions.json ...")
    path = Path("data/versions.json")

    if not path.exists():
        print("  [warn] data/versions.json not found — creating empty")
        payload = {"chunks": {}}
        path.write_text(json.dumps(payload, indent=2))
        print("  [ok] created data/versions.json")
        return

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"  [FAIL] data/versions.json is malformed: {e}")
        sys.exit(1)

    _check("chunks" in data, "versions.json has 'chunks' field")
    print(f"  [ok] data/versions.json ({len(data.get('chunks', {}))} chunks tracked)")


def _verify_scripts():
    """
    Verifies that all required scripts are present in the scripts/ directory.
    """
    print("\n[initialize] checking required scripts ...")
    for script in REQUIRED_SCRIPTS:
        _check(os.path.isfile(f"scripts/{script}"), f"script exists: {script}")


def _download_model():
    """
    Checks if the model is cached locally, and downloads it if not.
    """
    print(f"\n[initialize] checking model ({MODEL_NAME}) ...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Check if model is already cached
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(MODEL_NAME, "config.json")
        if cached is not None:
            print(f"  [ok] model already cached locally")
            return

        # Not cached — download now
        print(f"  [warn] model not found locally — downloading (~8GB, this may take a while)")
        AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print(f"  [ok] model downloaded and cached")

    except ImportError as e:
        print(f"  [FAIL] missing dependency: {e}")
        print("         run: pip install transformers torch huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] model download failed: {e}")
        sys.exit(1)


def _verify_gpu():
    print("\n[initialize] checking GPU ...")
    try:
        import torch
        if torch.cuda.is_available():
            name   = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [ok] GPU found: {name} ({vram:.1f}GB VRAM)")
            if vram < 4.0:
                print(f"  [warn] less than 4GB VRAM — consider 4-bit quantization")
        else:
            print("  [warn] no GPU found — inference will be slow on CPU")
    except ImportError:
        print("  [warn] torch not installed, skipping GPU check")


# ----- Client Specific Checks -----

def _verify_chunks_jsonl():
    """Checks that data/chunks.jsonl exists and is non-empty."""
    print("\n[initialize] checking data/chunks.jsonl ...")
    path = Path("data/chunks.jsonl")
    _check(path.exists(), "data/chunks.jsonl exists")
    if path.exists():
        count = sum(1 for line in path.read_text().splitlines() if line.strip())
        print(f"  [ok] {count} records in chunks.jsonl")


# ----- Server and Client Specific Checks -----

def _verify_chunks_populated():
    """Checks that data/chunks/ contains .txt files corresponding to chunks.jsonl."""
    print("\n[initialize] checking data/chunks/ is populated ...")
    chunk_files = list(Path("data/chunks").glob("*.txt"))
    if not chunk_files:
        print("  [warn] data/chunks/ is empty — populate from chunks.jsonl")
    else:
        print(f"  [ok] {len(chunk_files)} chunk files present")


# ----- Public entrypoints -----

def run_common():
    """Shared checks for both nodes."""
    print("\n=== initialize (common) ===")
    _verify_directories()
    _verify_prompt()
    _verify_versions()
    _verify_scripts()
    _verify_gpu()
    _download_model()
    print("\n[initialize] common checks passed\n")


def run_client():
    """Client-specific initialization."""
    run_common()
    print("=== initialize (client) ===")
    _verify_chunks_jsonl()
    _verify_chunks_populated()
    print("\n[initialize] client ready\n")


def run_server():
    """Server-specific initialization."""
    run_common()
    print("=== initialize (server) ===")
    _verify_chunks_populated()
    print("\n[initialize] server ready\n")


# ----- Main entrypoint for testing -----

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verify environment is ready")
    ap.add_argument("--mode",        choices=["common", "client", "server"], default="common")

    args = ap.parse_args()

    if args.mode == "client":
        run_client()
    elif args.mode == "server":
        run_server()
    else:
        run_common()