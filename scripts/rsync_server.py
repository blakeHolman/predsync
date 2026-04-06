#!/usr/bin/env python3
# scripts/rsync_server.py  (run on server)
#
# Starts an rsync daemon that listens for incoming syncs from node0.
# No SSH required — node0 connects directly over TCP.
#
# Usage:
#   python rsync_server.py
#   python rsync_server.py --port 8730 --path /tmp/rsync_bench

import argparse
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Start rsync daemon on node1 (server)")
    ap.add_argument(
        "--port",
        type=int,
        default=8730,
        help="Port to listen on (default: 8730)",
    )
    ap.add_argument(
        "--path",
        default="/tmp/rsync_bench",
        help="Directory to receive files into (default: /tmp/rsync_bench)",
    )
    args = ap.parse_args()

    recv_path = Path(args.path)
    recv_path.mkdir(parents=True, exist_ok=True)

    # Write rsyncd.conf to a temp file
    conf = tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False)
    conf.write(f"[bench]\n")
    conf.write(f"path = {recv_path}\n")
    conf.write(f"read only = no\n")
    conf.close()

    print(f"[rsync_server] listening on port {args.port}")
    print(f"[rsync_server] receiving files into {recv_path}")
    print(f"[rsync_server] press Ctrl+C to stop")

    proc = subprocess.Popen([
        "rsync",
        "--daemon",
        "--no-detach",
        f"--config={conf.name}",
        f"--port={args.port}",
    ])

    def _shutdown(sig, frame):
        print("\n[rsync_server] shutting down")
        proc.terminate()
        os.unlink(conf.name)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    proc.wait()
    os.unlink(conf.name)


if __name__ == "__main__":
    main()