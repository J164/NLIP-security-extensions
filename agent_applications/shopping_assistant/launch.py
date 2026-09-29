"""Start all four shopping-assistant servers and stop them together on Ctrl-C."""

from __future__ import annotations

import subprocess
import sys
import time

import httpx

from ._config import CONFIG

PACKAGE = __package__
SERVERS = {
    "search_agent": f"{PACKAGE}.agents.search_agent",
    "review_agent": f"{PACKAGE}.agents.review_agent",
    "checkout_agent": f"{PACKAGE}.agents.checkout_agent",
    "orchestrator": f"{PACKAGE}.orchestrator",
}


def health_url(entity: str) -> str:
    settings = CONFIG["entities"][entity]
    return f"http://{settings['host']}:{settings['port']}/health"


def wait_healthy(entity: str, process: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{entity} exited with code {process.returncode}.")
        try:
            if httpx.get(health_url(entity), timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"{entity} did not become healthy within {timeout:.0f}s.")


def main() -> None:
    processes: dict[str, subprocess.Popen] = {}
    try:
        for entity, module in SERVERS.items():
            processes[entity] = subprocess.Popen([sys.executable, "-m", module])
            wait_healthy(entity, processes[entity])
            print(f"[launch] {entity} ready at {health_url(entity).removesuffix('/health')}", flush=True)
        orchestrator = CONFIG["entities"]["orchestrator"]
        print(
            f"[launch] Open http://localhost:{orchestrator['port']}/ "
            "(from another machine: ssh -L "
            f"{orchestrator['port']}:localhost:{orchestrator['port']} <vm>). Ctrl-C to stop.",
            flush=True,
        )
        while all(process.poll() is None for process in processes.values()):
            time.sleep(0.5)
        print("[launch] A server exited; stopping the rest.", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
