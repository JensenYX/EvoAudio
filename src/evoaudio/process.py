from __future__ import annotations

import argparse
import os
import select
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A worker command is required")
    child = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    started = heartbeat = time.monotonic()
    code = 1
    try:
        while child.poll() is None:
            ready, _, _ = select.select([sys.stdin], [], [], 1)
            if ready:
                if not os.read(sys.stdin.fileno(), 4096):
                    break
                heartbeat = time.monotonic()
            if time.monotonic() - heartbeat > 30 or time.monotonic() - started > args.timeout:
                break
        else:
            code = child.returncode
    except KeyboardInterrupt:
        code = 130
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
