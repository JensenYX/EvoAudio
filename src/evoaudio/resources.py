from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

from evoaudio.io import log


def local_host(host: str) -> bool:
    names = {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}
    try:
        names.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return host in names


def command_on(host: str, command: list[str], env: dict[str, str]) -> list[str]:
    if local_host(host):
        return ["env", *[f"{k}={v}" for k, v in env.items()], *command]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        host,
        shlex.join(["env", *[f"{k}={v}" for k, v in env.items()], *command]),
    ]


def run_worker(
    cfg,
    module: str,
    args: list[str],
    output: Path,
    *,
    python: str | None = None,
    devices: list[int] | None = None,
    host: str | None = None,
    extra_env: dict | None = None,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices or cfg.resources.devices[:1])),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    env.update(extra_env or {})
    worker_python = python or cfg.resources.train_python
    command = supervised(cfg, worker_python, [worker_python, "-m", module, *args])
    cmd = command_on(host or cfg.resources.hosts[0], command, env)
    log(f"Running {module}; log: {output}")
    with output.open("a") as stream:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT)
        wait_workers([process], cfg.resources.worker_timeout, output)


def supervised(cfg, python: str, command: list[str]) -> list[str]:
    return [python, "-m", "evoaudio.process", "--timeout", str(cfg.resources.worker_timeout), "--", *command]


def wait_workers(processes, timeout: int, logs: Path):
    started = time.monotonic()
    try:
        while True:
            codes = [p.poll() for p in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Worker failed. Read {logs}")
            if all(code == 0 for code in codes):
                return
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"Worker exceeded worker_timeout. Read {logs}")
            for process in processes:
                if process.poll() is None:
                    try:
                        process.stdin.write(b".\n")
                        process.stdin.flush()
                    except BrokenPipeError:
                        pass
            time.sleep(2)
    finally:
        for process in processes:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)


def launch_training(cfg, command: list[str], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    hosts = cfg.resources.hosts
    devices = cfg.resources.devices
    processes = []
    streams = []
    try:
        for rank, host in enumerate(hosts):
            env = {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
                "NPROC_PER_NODE": str(len(devices)),
                "NNODES": str(len(hosts)),
                "NODE_RANK": str(rank),
                "MASTER_ADDR": hosts[0],
                "MASTER_PORT": str(cfg.resources.master_port),
                "OMP_NUM_THREADS": "2",
                "ENABLE_AUDIO_OUTPUT": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "EVOAUDIO_REWARD_LOG": str((root.parent / "rewards").resolve()),
            }
            for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_IB_DISABLE"):
                if key in os.environ:
                    env[key] = os.environ[key]
            stream = (root / f"node{rank}.log").open("a")
            streams.append(stream)
            proc = subprocess.Popen(
                command_on(host, supervised(cfg, cfg.resources.train_python, command), env),
                stdin=subprocess.PIPE,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(proc)
        wait_workers(processes, cfg.resources.worker_timeout, root)
    finally:
        for p in processes:
            if p.stdin and not p.stdin.closed:
                p.stdin.close()
        for stream in streams:
            stream.close()
