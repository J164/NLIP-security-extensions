"""Replay benchmark cases end to end against real shopping-assistant stacks.

    python -m adversarial_tests.agentspec_benchmark.harness \
        --models gpt-oss-120b --subset smoke --reps 1 --defense off

For each (model, defense) pair the harness writes a config, starts the four servers plus a
pass-through proxy in front of the search and review agents, checks the model with one
tool-call request, then replays each case over real NLIP/HTTP: mock-OAuth login, then the
case's turns with the user client from ``client_cli.py``. Every trial appends one row to
``<out>/results.jsonl`` (format in ``report.py``) and saves its full observation under
``<out>/trials/``. ``--resume`` skips trials already recorded as ok or error.

Run layout::

    <out>/manifest.json        git commit, config hashes, endpoints, preflight, times
    <out>/results.jsonl        one row per trial
    <out>/configs/             generated config per (model, defense)
    <out>/stacks/<name>/       per-stack logs, ledger, usage log, SE log, overlay dir
    <out>/trials/<model>/<defense>/<case>-r<rep>.json
    <out>/summary.md, summary.csv
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from agent_applications.shopping_assistant import protocol
from agent_applications.shopping_assistant._config import APP_DIR
from agent_applications.shopping_assistant.client_cli import ShoppingClient, mock_login
from agent_applications.shopping_assistant.launch import SERVERS
from nlip_sdk import NLIPClient, NLIPMessage, NLIPSubMessage

from . import cases as case_lib
from . import oracles, report

BENCH_DIR = Path(__file__).parent
REPO_DIR = BENCH_DIR.parent.parent
BASE_CONFIG = APP_DIR / "config.toml"
DEFENSE_OVERLAY = BENCH_DIR / "configs" / "defense_on.toml"

MODELS = {
    "gpt-oss-120b": "https://llm.jetstream-cloud.org/gpt-oss-120b/v1",
    "muse-glimmer": "https://llm.jetstream-cloud.org/v1",
    "llama-4-scout": "https://llm.jetstream-cloud.org/llama-4-scout/v1",
    "gemma-4-31b-it": "https://llm.jetstream-cloud.org/v1",
    "qwen-3.8-27b": "https://llm.jetstream-cloud.org/v1",
    "qwen-3.8-flash-next": "https://llm.jetstream-cloud.org/v1",
    "gemma-4-31b-it": "https://llm.jetstream-cloud.org/v1",
    "qwen-3.8-27b": "https://llm.jetstream-cloud.org/v1",
    "qwen-3.8-flash-next": "https://llm.jetstream-cloud.org/v1",
}
BENCH_USERS = {"carol@example.com": {"name": "Carol Example"}}

# Port offsets inside a stack's block of 100.
PORTS = {
    "orchestrator": 0, "search_agent": 1, "review_agent": 2, "checkout_agent": 3,
    "search_agent_real": 11, "review_agent_real": 12,
}
PROXIED = ("search_agent", "review_agent")
BLOCK_RE = re.compile(r"^(?P<extension>\S+) blocked (?P<point>ingress|egress): (?P<reason>.*)$", re.S)


class TrialError(RuntimeError):
    """The trial could not be completed (timeout, HTTP 5xx, server failure)."""


# --- configs ------------------------------------------------------------------------------

def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def dump_toml(data: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> str:
    """Minimal TOML writer for the app's config shape (tables, scalars, lists)."""
    lines = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    if prefix:
        lines.append("[" + ".".join(_toml_key(p) for p in prefix) + "]")
    lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in scalars.items()]
    out = "\n".join(lines) + ("\n\n" if lines else "")
    for key, table in tables.items():
        out += dump_toml(table, (*prefix, key))
    return out


def deep_merge(base: dict, overlay: Mapping) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def defense_enabled(config: Mapping[str, Any]) -> bool:
    entities = config.get("entities", {})
    return any(
        entity.get("security", {}).get("ingress") or entity.get("security", {}).get("egress")
        or entity.get("tool_guards")
        for entity in entities.values()
    )


def stack_config(model: str, defense: str, port_base: int) -> dict[str, Any]:
    """The app config for one stack: its ports, model, benchmark users and defense."""
    with BASE_CONFIG.open("rb") as file:
        config = tomllib.load(file)
    if defense == "on":
        with DEFENSE_OVERLAY.open("rb") as file:
            config = deep_merge(config, tomllib.load(file))
        if not defense_enabled(config):
            raise SystemExit(f"--defense on: {DEFENSE_OVERLAY} enables no security extension or tool guard yet.")
    for entity in ("orchestrator", "search_agent", "review_agent", "checkout_agent"):
        port = port_base + PORTS[entity]
        settings = config["entities"][entity]
        settings.update(host="127.0.0.1", port=port, endpoint=f"http://127.0.0.1:{port}/nlip")
    config["llm"].update(base_url=MODELS[model], model=model)
    config["auth"].update(provider="mock", redirect_uri=f"http://localhost:{port_base}/auth/callback")
    config.setdefault("mock_users", {}).update(BENCH_USERS)
    return config


# --- stacks -------------------------------------------------------------------------------

@dataclass
class Stack:
    """Four servers and two proxies for one (model, defense), on one block of ports."""

    model: str
    defense: str
    port_base: int
    directory: Path
    config_path: Path
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port_base}"

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port_base + PORTS[name]}"

    @property
    def orders(self) -> Path:
        return self.directory / "orders.jsonl"

    @property
    def usage_log(self) -> Path:
        return self.directory / "usage.jsonl"

    @property
    def security_log(self) -> Path:
        return self.directory / "security.jsonl"

    @property
    def overlay(self) -> Path:
        return self.directory / "overlay"

    def _env(self, entity: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("SHOP_")}
        env.update(
            PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_DIR), env.get("PYTHONPATH")])),
            SHOP_CONFIG=str(self.config_path),
            SHOP_LLM_MODEL=self.model,
            SHOP_LLM_BASE_URL=MODELS[self.model],
            SHOP_ORDERS_PATH=str(self.orders),
            SHOP_OVERLAY_DIR=str(self.overlay),
            SHOP_LLM_USAGE_LOG=str(self.usage_log),
            SHOP_SECURITY_LOG=str(self.security_log),
            SHOP_ENTITY=entity,
            AUTH_PROVIDER="mock",
        )
        return env

    def _spawn(self, name: str, args: list[str], entity: str) -> None:
        logs = self.directory / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = (logs / f"{name}.log").open("a")
        self.processes[name] = subprocess.Popen(
            [sys.executable, *args], cwd=REPO_DIR, env=self._env(entity), stdout=log, stderr=subprocess.STDOUT
        )

    def _uvicorn(self, name: str, target: str, port_name: str, *extra: str) -> None:
        port = str(self.port_base + PORTS[port_name])
        self._spawn(name, ["-m", "uvicorn", target, *extra, "--host", "127.0.0.1", "--port", port,
                           "--log-level", "warning"], name)

    async def start(self) -> None:
        self.overlay.mkdir(parents=True, exist_ok=True)
        for entity in PROXIED:
            self._uvicorn(entity, f"{SERVERS[entity]}:app", f"{entity}_real")
            self._spawn(f"{entity}_proxy", [
                "-m", f"{__package__}.proxy", "--port", str(self.port_base + PORTS[entity]),
                "--upstream", self.url(f"{entity}_real"), "--target", entity,
            ], f"{entity}_proxy")
        self._uvicorn("checkout_agent", f"{SERVERS['checkout_agent']}:app", "checkout_agent")
        self._uvicorn("orchestrator", f"{SERVERS['orchestrator']}:create_app", "orchestrator", "--factory")
        for name in ("search_agent", "review_agent", "checkout_agent", "orchestrator"):
            await self._wait_healthy(name)

    async def _wait_healthy(self, name: str, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                dead = [n for n, p in self.processes.items() if p.poll() is not None]
                if dead:
                    raise TrialError(f"{', '.join(dead)} exited; see {self.directory / 'logs'}")
                try:
                    if (await client.get(f"{self.url(name)}/health")).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.3)
        raise TrialError(f"{name} not healthy after {timeout:.0f}s; see {self.directory / 'logs'}")

    def stop(self) -> None:
        for process in self.processes.values():
            process.terminate()
        for process in self.processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()

    async def restart(self) -> None:
        self.stop()
        await self.start()


# --- preflight ----------------------------------------------------------------------------

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_catalog",
        "description": "Search the shop catalog for products.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}


async def preflight(model: str, timeout: float = 120.0) -> dict[str, Any]:
    """One tool-call request. ok means the model answered with a tool call."""
    started = time.monotonic()
    body = {
        "model": model, "temperature": 0, "max_tokens": 1024, "tools": [PROBE_TOOL],
        "messages": [{"role": "user", "content": "Find wireless headphones in the catalog."}],
    }
    key = os.getenv("SHOP_LLM_API_KEY", "empty")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{MODELS[model]}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
            )
        latency = round(time.monotonic() - started, 2)
        if response.status_code != 200:
            return {"ok": False, "latency_s": latency, "error": f"HTTP {response.status_code}: {response.text[:300]}"}
        calls = (response.json().get("choices") or [{}])[0].get("message", {}).get("tool_calls")
        if not calls:
            return {"ok": False, "latency_s": latency, "error": "no tool call in the reply"}
        return {"ok": True, "latency_s": latency, "error": None}
    except (httpx.HTTPError, ValueError) as error:
        return {"ok": False, "latency_s": round(time.monotonic() - started, 2), "error": repr(error)}


# --- one trial ----------------------------------------------------------------------------

class BenchClient(ShoppingClient):
    """``ShoppingClient`` with a per-request timeout long enough for slow models."""

    def __init__(self, base: str, ticket: str, timeout: float) -> None:
        self.ticket = ticket
        self.session = NLIPClient(
            {"orchestrator": f"{base}/nlip"}, identity="bench_user", timeout=timeout
        ).create_session("orchestrator")


def _read_lines(path: Path, start: int = 0) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as file:
        return [json.loads(line) for line in file.read().splitlines()[start:] if line.strip()]


def _count_lines(path: Path) -> int:
    return len(_read_lines(path))


def write_overlays(directory: Path, overlays: Mapping[str, Any] | None) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    (directory / "pages").mkdir(parents=True)
    overlays = overlays or {}
    for pid, text in (overlays.get("pages") or {}).items():
        (directory / "pages" / f"{pid}.txt").write_text(text)
    if overlays.get("reviews"):
        (directory / "reviews.json").write_text(json.dumps(overlays["reviews"]))


def _turn_message(turn: Mapping[str, Any]) -> NLIPMessage:
    message = NLIPMessage.text(turn["say"])
    for part in turn.get("attach", []):
        message.add_submessage(NLIPSubMessage(**part))
    return message


def _record(kind: str, response: NLIPMessage) -> dict[str, Any]:
    fields = dict(protocol.structured_fields(response))
    return {
        "kind": kind, "status": 200,
        "text": response.content if isinstance(response.content, str) else "",
        "order_status": fields.get(protocol.ORDER_STATUS), "trace": fields.get(protocol.TRACE) or [],
        "detail": None,
    }


def defense_events(turns: list[dict], security_rows: list[dict]) -> list[dict]:
    """SE and guard decisions from 403 block responses, guard trace entries and the SE log."""
    events = []
    for turn in turns:
        match = BLOCK_RE.match(turn.get("detail") or "") if turn["status"] == 403 else None
        if match:
            events.append({"source": "http", "action": "block", **match.groupdict(),
                           "rule": oracles.rule_from_reason(match["reason"])})
        for step in turn.get("trace") or []:
            if "blocked_by" in step:
                events.append({"source": "guard", "action": "block", "extension": step["blocked_by"],
                               "reason": step.get("reason"), "rule": oracles.rule_from_reason(step.get("reason"))})
    for row in security_rows:
        events.append({"source": "se_log", **row, "rule": row.get("rule") or oracles.rule_from_reason(row.get("reason"))})
    return events


async def run_turns(stack: Stack, case: Mapping, timeout: float) -> tuple[list[dict], list[dict], list[float]]:
    """Log in and replay the case. Returns (turn records, new ledger rows, turn latencies)."""
    ticket = await mock_login(stack.base_url, case["user"])
    client = BenchClient(stack.base_url, ticket, timeout)
    turns, orders, latencies = [], [], []
    seen = _count_lines(stack.orders)
    for index, turn in enumerate(case["turns"]):
        kind = next(k for k in ("say", "confirm", "cancel") if k in turn)
        started = time.monotonic()
        try:
            if kind == "say":
                response = await client.send(_turn_message(turn))
            else:
                response = await client.confirm(kind == "confirm")
            record = _record(kind, response)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status >= 500 or status == 429:
                raise TrialError(f"turn {index}: HTTP {status}: {error.response.text[:300]}") from error
            try:
                detail = error.response.json().get("detail")
            except ValueError:
                detail = error.response.text
            record = {"kind": kind, "status": status, "text": "", "order_status": None, "trace": [],
                      "detail": str(detail)}
        except httpx.HTTPError as error:
            raise TrialError(f"turn {index}: {error!r}") from error
        latencies.append(round(time.monotonic() - started, 3))
        turns.append(record)
        new = _read_lines(stack.orders, seen)
        seen += len(new)
        orders += [{**row, "turn": index} for row in new]
    return turns, orders, latencies


async def run_trial(stack: Stack, case: Mapping, timeout: float) -> dict[str, Any]:
    """One attempt at one case. Returns the observation plus cost data."""
    write_overlays(stack.overlay, case.get("overlays"))
    proxy = case.get("proxy") or {}
    async with httpx.AsyncClient(timeout=10) as client:
        for target in PROXIED:
            ops = proxy.get("ops", []) if proxy.get("target") == target else []
            (await client.post(f"{stack.url(target)}/_bench/arm", json={"case_id": case["id"], "ops": ops})).raise_for_status()
    usage_start, security_start = _count_lines(stack.usage_log), _count_lines(stack.security_log)
    budget = timeout * len(case["turns"]) + 30
    try:
        turns, orders, latencies = await asyncio.wait_for(run_turns(stack, case, timeout), budget)
    except asyncio.TimeoutError as error:
        raise TrialError(f"trial exceeded {budget:.0f}s") from error
    proxy_log, proxy_applied = {}, {}
    async with httpx.AsyncClient(timeout=10) as client:
        for target in PROXIED:
            log = (await client.get(f"{stack.url(target)}/_bench/log")).json()
            proxy_log[target], proxy_applied[target] = log["received"], log["applied"]
    usage = _read_lines(stack.usage_log, usage_start)
    security = _read_lines(stack.security_log, security_start)
    observation = {
        "user": case["user"], "turns": turns, "new_orders": orders, "proxy_log": proxy_log,
        "defense_events": defense_events(turns, security),
    }
    return {"observation": observation, "latencies": latencies, "usage": usage, "security": security,
            "proxy_applied": proxy_applied}


# --- the run ------------------------------------------------------------------------------

@dataclass
class Run:
    args: argparse.Namespace
    out: Path
    cases: list[dict]
    done: set[tuple] = field(default_factory=set)

    @property
    def results(self) -> Path:
        return self.out / "results.jsonl"

    def append(self, row: dict) -> None:
        with self.results.open("a") as file:
            file.write(json.dumps(row) + "\n")

    def base_row(self, case: Mapping, model: str, defense: str, rep: int) -> dict[str, Any]:
        return {
            "case_id": case["id"], "kind": case["kind"], "family": case["family"],
            "mechanism": case.get("mechanism"), "model": model,
            "defense": defense, "rep": rep, "status": "ok", "attack_goal": None, "user_task": None,
            "attack_checks": {}, "se_fired": False, "se_rule": None,
            "expected_rule": case.get("expected_rule"), "turn_latencies_s": [], "se_ms": [],
            "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "error": None, "attempts": 0,
        }

    async def trial(self, stack: Stack, case: Mapping, rep: int) -> dict[str, Any]:
        row = self.base_row(case, stack.model, stack.defense, rep)
        started = time.time()
        for attempt in range(self.args.retries + 1):
            row["attempts"] = attempt + 1
            try:
                result = await run_trial(stack, case, self.args.turn_timeout)
                break
            except (TrialError, httpx.HTTPError) as error:
                row["error"] = str(error)
                print(f"  [{stack.model}/{stack.defense}] {case['id']} r{rep} attempt {attempt + 1}: {error}", flush=True)
                # A request still running inside the stack could write to the next case's
                # ledger, so restart before the next attempt or the next case.
                if attempt < self.args.retries:
                    await asyncio.sleep(10 * 3**attempt)
                await stack.restart()
        else:
            row["status"] = "error"
            return row
        observation = result["observation"]
        row.update(oracles.evaluate(case, observation), error=None)
        row.update(
            turn_latencies_s=result["latencies"],
            se_ms=[r["ms"] for r in result["security"] if isinstance(r.get("ms"), (int, float))],
            llm_calls=sum(r.get("llm_calls", 0) for r in result["usage"]),
            input_tokens=sum(r.get("input_tokens", 0) for r in result["usage"]),
            output_tokens=sum(r.get("output_tokens", 0) for r in result["usage"]),
            duration_s=round(time.time() - started, 2),
        )
        trial_dir = self.out / "trials" / stack.model / stack.defense
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / f"{case['id']}-r{rep}.json").write_text(json.dumps(
            {"case": case, "row": row, **result}, indent=1, ensure_ascii=False
        ))
        return row

    async def run_stack(self, stack: Stack, queue: asyncio.Queue) -> None:
        await stack.start()
        try:
            while not queue.empty():
                case, rep = queue.get_nowait()
                row = await self.trial(stack, case, rep)
                self.append(row)
                verdict = ("error" if row["status"] == "error" else
                           f"attack={row['attack_goal']} task={row['user_task']} se={row['se_fired']}")
                print(f"  [{stack.model}/{stack.defense}] {case['id']} r{rep}: {verdict}", flush=True)
        finally:
            stack.stop()

    async def run_model(self, index: int, model: str, defense: str, preflight_result: dict) -> None:
        todo = [(c, rep) for rep in range(self.args.reps) for c in self.cases
                if (c["id"], model, defense, rep) not in self.done]
        if not todo:
            return
        if not preflight_result["ok"]:
            for case, rep in todo:
                self.append({**self.base_row(case, model, defense, rep), "status": "unavailable",
                             "error": preflight_result["error"]})
            print(f"[{model}/{defense}] unavailable: {preflight_result['error']}", flush=True)
            return
        queue: asyncio.Queue = asyncio.Queue()
        for item in todo:
            queue.put_nowait(item)
        stacks = []
        for k in range(self.args.concurrency):
            port_base = self.args.port_base + 100 * (index * self.args.concurrency + k)
            name = f"{model}-{defense}-{k}"
            config_path = self.out / "configs" / f"{name}.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(dump_toml(stack_config(model, defense, port_base)))
            stacks.append(Stack(model, defense, port_base, self.out / "stacks" / name, config_path))
        print(f"[{model}/{defense}] {len(todo)} trials on {len(stacks)} stack(s)", flush=True)
        await asyncio.gather(*(self.run_stack(s, queue) for s in stacks))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def load_done(results: Path) -> set[tuple]:
    """Keep ok/error rows from an earlier run; drop the rest so they are retried."""
    if not results.is_file():
        return set()
    rows = [r for r in report.load([results]) if r["status"] in ("ok", "error")]
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {(r["case_id"], r["model"], r["defense"], r["rep"]) for r in rows}


async def main_async(args: argparse.Namespace) -> int:
    all_cases = case_lib.load_cases()
    selected = case_lib.select(all_cases, args.cases)
    if args.subset == "smoke":
        selected = [c for c in selected if c.get("smoke")]
    if not selected:
        print("No cases selected.", file=sys.stderr)
        return 1
    out = args.out or BENCH_DIR / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out.exists() and not args.resume and any(out.iterdir()):
        print(f"{out} exists; pass --resume to continue it.", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    defenses = ["off", "on"] if args.defense == "both" else [args.defense]
    for defense in defenses:  # fail early if the defense overlay is empty
        stack_config(args.models[0], defense, args.port_base)

    run = Run(args, out, selected, load_done(out / "results.jsonl") if args.resume else set())
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"runs": []}
    print(f"Preflight: {', '.join(args.models)}", flush=True)
    checks = await asyncio.gather(*(preflight(m) for m in args.models))
    pre = dict(zip(args.models, checks))
    for model, result in pre.items():
        print(f"  {model}: {'ok' if result['ok'] else 'unavailable'} ({result['latency_s']}s) {result['error'] or ''}")
    entry = {
        "started_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv[1:],
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "models": {m: {"base_url": MODELS[m], "preflight": pre[m]} for m in args.models},
        "defenses": defenses, "cases": [c["id"] for c in selected], "reps": args.reps,
        "hashes": {
            "base_config": _sha256(BASE_CONFIG), "defense_overlay": _sha256(DEFENSE_OVERLAY),
            **{p.name: _sha256(p) for p in case_lib.DEFAULT_FILES},
        },
    }
    manifest["runs"].append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=1))

    for defense in defenses:
        await asyncio.gather(*(run.run_model(i, m, defense, pre[m]) for i, m in enumerate(args.models)))

    entry["finished_at"] = datetime.now(timezone.utc).isoformat()
    entry["config_hashes"] = {p.name: _sha256(p) for p in sorted((out / "configs").glob("*.toml"))}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    summary = report.summarize(report.load([run.results]))
    text = report.markdown(summary)
    (out / "summary.md").write_text(text)
    report.write_csv(summary, out / "summary.csv")
    print(text)
    print(f"Results in {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(MODELS),
                        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
                        help=f"comma-separated, from {', '.join(MODELS)}")
    parser.add_argument("--defense", choices=("off", "on", "both"), default="off")
    parser.add_argument("--cases", default="all", help="attacks, benign, all, or comma-separated ids")
    parser.add_argument("--subset", choices=("smoke",), help="only cases marked smoke")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", type=Path, help="run directory (default results/<UTC time>)")
    parser.add_argument("--resume", action="store_true", help="continue the run in --out")
    parser.add_argument("--concurrency", type=int, default=1, help="stacks per model")
    parser.add_argument("--turn-timeout", type=float, default=180.0, help="seconds per user turn")
    parser.add_argument("--retries", type=int, default=2, help="retries per trial after an error")
    parser.add_argument("--port-base", type=int, default=18000)
    args = parser.parse_args(argv)
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    if args.resume and not args.out:
        parser.error("--resume needs --out")
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
