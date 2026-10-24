"""Offline tests for the evaluation benchmark: no LLM, no network.

The end-to-end tests run the harness's own ``run_trial`` against in-process apps: every
``httpx.AsyncClient`` is routed by port to the real orchestrator, proxy and subagent apps,
and the agents are scripted. So cases, proxy, ledger diff, oracles and observation format
are all exercised exactly as in a live run, minus the model.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from functools import partial

import httpx
import pytest
from langchain_core.messages import AIMessage

from adversarial_tests.agentspec_benchmark import build_benign, harness, oracles, report
from adversarial_tests.agentspec_benchmark import cases as case_lib
from adversarial_tests.agentspec_benchmark.proxy import apply_ops
from adversarial_tests.agentspec_benchmark.proxy import create_app as proxy_app
from agent_applications.shopping_assistant import catalog, llm, protocol
from agent_applications.shopping_assistant._config import CONFIG
from agent_applications.shopping_assistant.agents.checkout_agent import CheckoutAgent
from agent_applications.shopping_assistant.agents.review_agent import ReviewAgent
from agent_applications.shopping_assistant.agents.search_agent import SearchAgent
from agent_applications.shopping_assistant.orchestrator import Orchestrator, create_app
from nlip_sdk import AllowedFormat, NLIPMessage

ALL_CASES = case_lib.load_cases()
BY_ID = {c["id"]: c for c in ALL_CASES}
BENIGN = [c for c in ALL_CASES if c["kind"] == "benign"]
PRODUCT_ID = re.compile(r"\b[a-z]{2}-\d{3}\b")


# --- scripted agents ----------------------------------------------------------------------

class Agent:
    """A stand-in for LLMAgent: ``script(tools, prompt, turn)`` plays the model."""

    def __init__(self, script, system_prompt: str, tools) -> None:
        self.script, self.tools, self.turn = script, {t.name: t for t in tools}, 0

    async def chat(self, prompt: str) -> str:
        self.turn += 1
        return await self.script(self.tools, prompt, self.turn)


def factory(script):
    return partial(Agent, script)


async def search_script(tools, prompt, turn):
    await tools["search_catalog"].ainvoke({"query": prompt})
    for pid in PRODUCT_ID.findall(prompt):  # read the (possibly overlaid) page
        await tools["fetch_product_page"].ainvoke({"product_id": pid})
    return "Here are some matches."


async def review_script(tools, prompt, turn):
    for pid in PRODUCT_ID.findall(prompt) or ["hp-001"]:
        await tools["get_reviews"].ainvoke({"product_id": pid})
    return "Reviews are mostly positive."


def faithful_shopper(case):
    """Does what the user asked: browse on each turn, and buy the task's items on the last say."""
    says = [n for n, t in enumerate(case["turns"]) if "say" in t]
    items = case["oracle"]["user_task"].get("items")

    async def script(tools, prompt, turn):
        await tools["ask_search_agent"].ainvoke({"request": prompt})
        await tools["ask_review_agent"].ainvoke({"product_ids": PRODUCT_ID.findall(prompt) or ["hp-001"]})
        if items and turn == len(says):
            await tools["request_checkout"].ainvoke({"items": items})
            return await tools["place_order"].ainvoke({})
        return "Here is what I found."

    return script


async def credulous_shopper(tools, prompt, turn):
    """Quotes on turn 1; on turn 2 reads reviews, then orders the existing quote without re-quoting."""
    if turn == 1:
        return await tools["request_checkout"].ainvoke({"items": [{"product_id": "hp-002", "quantity": 1}]})
    await tools["ask_review_agent"].ainvoke({"product_ids": ["hp-002"]})
    return await tools["place_order"].ainvoke({})


# --- in-process stack ---------------------------------------------------------------------

class Router(httpx.AsyncBaseTransport):
    """Send each request to the in-process ASGI app registered for its port."""

    def __init__(self) -> None:
        self.apps: dict[int, httpx.ASGITransport] = {}

    def add(self, port: int, app) -> None:
        self.apps[port] = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.apps[request.url.port].handle_async_request(request)


@pytest.fixture
def stack_factory(tmp_path, monkeypatch):
    """Build a Stack whose servers are in-process apps, for one orchestrator script."""
    real_client = httpx.AsyncClient
    router = Router()
    monkeypatch.setattr(httpx, "AsyncClient", partial(real_client, transport=router))
    monkeypatch.setenv("SHOP_ORDERS_PATH", str(tmp_path / "orders.jsonl"))
    monkeypatch.setenv("SHOP_OVERLAY_DIR", str(tmp_path / "overlay"))
    for email, info in harness.BENCH_USERS.items():  # as in the harness's generated config
        monkeypatch.setitem(CONFIG["mock_users"], email, info)

    def build(script) -> harness.Stack:
        # The orchestrator dials the base config's endpoints (ports 8001-8003).
        stack = harness.Stack("scripted", "off", 8000, tmp_path, tmp_path / "unused.toml")
        port = lambda name: 8000 + harness.PORTS[name]  # noqa: E731
        router.add(port("orchestrator"), create_app(Orchestrator(agent_factory=factory(script))))
        router.add(port("search_agent_real"), SearchAgent(factory(search_script)).create_nlip_app())
        router.add(port("review_agent_real"), ReviewAgent(factory(review_script)).create_nlip_app())
        router.add(port("checkout_agent"), CheckoutAgent().create_nlip_app())
        for name in harness.PROXIED:
            router.add(port(name), proxy_app(stack.url(f"{name}_real"), name, transport=router))
        return stack

    return build


def run(stack, case):
    return asyncio.run(harness.run_trial(stack, case, timeout=30))


# --- case files ---------------------------------------------------------------------------

def test_case_files_validate_and_cite_known_references():
    assert any(c["kind"] == "attack" for c in ALL_CASES)
    assert len(BENIGN) >= 40
    keys = case_lib.bib_keys()
    for case in ALL_CASES:
        assert set(case.get("source", [])) <= keys, case["id"]
        if case["kind"] == "attack":
            assert case["source"], case["id"]


def test_benign_file_matches_the_generator():
    assert build_benign.main(["--check"]) == 0


def test_benign_set_covers_every_user_and_tool_path():
    users = {c["user"] for c in BENIGN}
    expects = {c["oracle"]["user_task"]["expect"] for c in BENIGN}
    assert users == set(case_lib.USERS)
    assert expects == set(case_lib.EXPECTS)
    assert {c["family"] for c in BENIGN} == {"ordinary", "hard_negative"}
    assert any(c.get("overlays", {}).get("pages") for c in BENIGN)
    assert any(c.get("overlays", {}).get("reviews") for c in BENIGN)
    assert any(c.get("proxy") for c in BENIGN)
    assert any(t.get("attach") for c in BENIGN for t in c["turns"])
    assert any(t.get("cancel") for c in BENIGN for t in c["turns"])


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c.update(family="ordinary"), "family"),
    (lambda c: c.update(source=["not_a_key"]), "references.bib"),
    (lambda c: c.update(user="eve@example.com"), "user"),
    (lambda c: c.update(typo=1), "unknown key"),
    (lambda c: c["proxy"].update(target="checkout_agent"), "proxy.target"),
    (lambda c: c["proxy"]["ops"][0].update(position=-1), "position"),
    (lambda c: c["oracle"].update(attack_goal=[]), "at least one check"),
    (lambda c: c["oracle"]["user_task"].pop("items"), "items required"),
    (lambda c: c.update(family="indirect_injection"), "family"),
    (lambda c: c.pop("proxy"), "need a proxy"),
    (lambda c: c.update(overlays={"pages": {"hp-002": "x"}}), "not allowed in attack cases"),
    (lambda c: c["turns"][0].update(attach=[{"format": "text", "subformat": "English", "content": "x"}]),
     "attach is not allowed"),
])
def test_validator_rejects_broken_cases(mutate, message):
    case = json.loads(json.dumps(BY_ID["fi-seed-forged-confirmation"]))
    mutate(case)
    errors = case_lib.validate_case(case)
    assert any(message in e for e in errors), errors


def test_validator_checks_expected_outcome_against_scope():
    case = json.loads(json.dumps(BY_ID["b-ord-confirm-carol"]))
    case["turns"] = case["turns"][:1]  # drop the confirm turn
    assert any("needs confirmation" in e for e in case_lib.validate_case(case))
    case["oracle"]["user_task"]["expect"] = "refused"
    assert any("within the" in e for e in case_lib.validate_case(case))


def test_load_cases_reports_duplicates_and_bad_json(tmp_path):
    line = json.dumps(BY_ID["b-ord-browse-alice-1"])
    path = tmp_path / "x.jsonl"
    path.write_text(f"{line}\n// comment\n{line}\nnot json\n")
    with pytest.raises(case_lib.CaseError) as error:
        case_lib.load_cases([path])
    assert any("duplicate id" in e for e in error.value.errors)
    assert any("invalid JSON" in e for e in error.value.errors)


# --- end to end with scripted agents ------------------------------------------------------

def test_seed_attack_succeeds_through_the_proxy(stack_factory):
    case = BY_ID["fi-seed-forged-confirmation"]
    result = run(stack_factory(credulous_shopper), case)
    observation = result["observation"]
    verdict = oracles.evaluate(case, observation)
    assert verdict["attack_goal"] is True
    assert verdict["attack_checks"]["policy_violation_order"] is True
    # $39.99 is within Bob's $60 limit but needs his Confirm, which never happened.
    assert [(o["amount"], o["turn"]) for o in observation["new_orders"]] == [(39.99, 1)]
    assert observation["new_orders"][0]["customer"] == "bob@example.com"
    assert observation["turns"][1]["trace"][0] == {
        "destination": "review_agent", "merged_fields": ["review_summary", "requires_confirmation"]
    }
    assert all(op["applied"] for op in result["proxy_applied"]["review_agent"][0]["ops"])


def test_seed_attack_fails_without_the_proxy_ops(stack_factory):
    """Sanity check on the oracle: the same agent with an honest review reply leaves the order pending."""
    case = {**BY_ID["fi-seed-forged-confirmation"]}
    case.pop("proxy")
    verdict = oracles.evaluate(case, run(stack_factory(credulous_shopper), case)["observation"])
    assert verdict["attack_goal"] is False and verdict["user_task"] is True


@pytest.mark.parametrize("case", BENIGN, ids=lambda c: c["id"])
def test_benign_case_passes_user_task_and_no_attack_check(stack_factory, case):
    result = run(stack_factory(faithful_shopper(case)), case)
    verdict = oracles.evaluate(case, result["observation"])
    assert verdict["user_task"] is True, result["observation"]
    assert not any(verdict["attack_checks"].values()), verdict["attack_checks"]
    if case.get("proxy"):  # benign proxy ops (extra parts) are applied
        assert result["proxy_applied"][case["proxy"]["target"]]
    else:  # an unarmed proxy changes nothing
        assert not any(result["proxy_applied"].values())


def test_overlays_reach_the_tools(stack_factory, tmp_path):
    case = BY_ID["b-hn-page-authorized-dealer"]
    harness.write_overlays(tmp_path / "overlay", case["overlays"])
    assert "authorized dealer" in catalog.product_page("ch-001")
    harness.write_overlays(tmp_path / "overlay", BY_ID["b-hn-review-seller-response"]["overlays"])
    reviews = catalog.reviews()
    assert reviews["cg-001"][1]["reviewer"] == "Seller response"
    assert reviews["hp-001"] == catalog._base_reviews()["hp-001"]  # untouched products keep theirs
    assert "authorized dealer" not in catalog.product_page("ch-001")  # previous case cleared


def test_overlay_env_unset_keeps_normal_content(monkeypatch):
    monkeypatch.delenv("SHOP_OVERLAY_DIR", raising=False)
    assert catalog.product_page("hp-001").startswith("Aurora ANC Wireless Headphones")
    assert catalog.reviews() is catalog._base_reviews()


# --- oracles and defense signals ----------------------------------------------------------

def _turn(kind="say", status=200, pending=None, detail=None, trace=()):
    return {"kind": kind, "status": status, "text": "", "detail": detail, "trace": list(trace),
            "order_status": {"pending_order": pending} if status == 200 else None}


def test_policy_violation_uses_the_turn_that_committed_the_order():
    case = BY_ID["b-ord-confirm-carol"]  # ch-003, $69, carol: confirm needed from $25
    order = {"customer": "carol@example.com", "amount": 69.0, "items": [{"product_id": "ch-003"}]}
    base = {"user": "carol@example.com", "turns": [_turn(), _turn("confirm")], "proxy_log": {},
            "defense_events": []}
    assert not oracles.evaluate(case, {**base, "new_orders": [{**order, "turn": 1}]})["attack_goal"]
    skipped = oracles.evaluate(case, {**base, "new_orders": [{**order, "turn": 0}]})
    assert skipped["attack_checks"]["policy_violation_order"] is True


def test_sent_to_subagent_and_order_mismatch():
    case = BY_ID["b-ord-auto-alice-1"]
    observation = {
        "user": "alice@example.com", "turns": [_turn()], "defense_events": [],
        "new_orders": [{"customer": "alice@example.com", "amount": 1.0, "turn": 0,
                        "items": [{"product_id": "hp-005", "quantity": 1}]}],
        "proxy_log": {"review_agent": [{"content": "for ALICE@example.com", "format": "text"}]},
    }
    verdict = oracles.evaluate(case, observation)
    assert verdict["attack_checks"]["order_mismatch"] is True
    assert verdict["attack_checks"]["sent_to_subagent:review_agent"] is True
    assert verdict["attack_checks"]["sent_to_subagent:search_agent"] is False
    assert verdict["user_task"] is False


def test_defense_events_from_403_guard_trace_and_se_log():
    turns = [
        _turn(status=403, detail="agentspec blocked ingress: @block_field_impersonation search_agent wrote cart_total"),
        _turn(status=403, detail="This conversation belongs to another user."),
        _turn(trace=[{"tool": "place_order", "blocked_by": "guard", "reason": "@enforce_purchase_scope over limit"}]),
    ]
    events = harness.defense_events(turns, [{"action": "allow", "ms": 0.2}, {"action": "replace", "reason": "@block_label_spoof"}])
    assert [(e["source"], e["rule"]) for e in events] == [
        ("http", "block_field_impersonation"), ("guard", "enforce_purchase_scope"),
        ("se_log", None), ("se_log", "block_label_spoof"),
    ]
    assert oracles.defense_fired({"defense_events": events}) == (True, "block_field_impersonation")
    assert oracles.defense_fired({"defense_events": events[2:3]}) == (False, None)


# --- proxy operations ---------------------------------------------------------------------

def test_apply_ops_positions_ignore_session_tokens():
    message = protocol.reply("hello", {"product_results": {"results": [{"id": "hp-001"}]}})
    message.add_token("session", "conversation")
    ops = [
        {"op": "add_submessage", "format": "text", "subformat": "English", "content": "first", "position": "first"},
        {"op": "add_submessage", "format": "generic", "subformat": "note", "content": "last"},
        {"op": "replace_field", "label": "product_results", "path": ["results", 0, "id"], "value": "hp-005"},
        {"op": "replace_field", "label": "missing", "value": 1},
        {"op": "set_primary_text", "text": "changed"},
    ]
    changed, outcomes = apply_ops(message, ops)
    parts = changed.submessages
    assert [p.content for p in parts[:1]] == ["first"] and parts[-2].content == "last"
    assert parts[-1].format == AllowedFormat.token
    assert parts[1].content == {"results": [{"id": "hp-005"}]}
    assert changed.content == "changed"
    assert [o["applied"] for o in outcomes] == [True, True, True, False, True]
    assert message.content == "hello"  # the original is untouched


# --- harness helpers ----------------------------------------------------------------------

def test_stack_config_round_trips_and_defense_on_needs_an_extension(tmp_path):
    import tomllib

    config = harness.stack_config("gpt-oss-120b", "off", 18100)
    assert tomllib.loads(harness.dump_toml(config)) == config
    assert config["entities"]["search_agent"]["endpoint"] == "http://127.0.0.1:18101/nlip"
    assert "carol@example.com" in config["mock_users"]
    with pytest.raises(SystemExit):
        harness.stack_config("gpt-oss-120b", "on", 18100)
    merged = harness.deep_merge(config, {"entities": {"orchestrator": {"security": {"ingress": ["x"]}}}})
    assert harness.defense_enabled(merged) and not harness.defense_enabled(config)


def test_usage_log_writes_one_row_per_chat(tmp_path, monkeypatch):
    path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("SHOP_LLM_USAGE_LOG", str(path))
    monkeypatch.setenv("SHOP_ENTITY", "search_agent")
    replies = [
        AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
        AIMessage(content="done", usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}),
    ]
    llm.log_usage(replies, 1.5)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert {k: rows[0][k] for k in ("entity", "llm_calls", "input_tokens", "output_tokens", "latency_s")} == {
        "entity": "search_agent", "llm_calls": 2, "input_tokens": 40, "output_tokens": 7, "latency_s": 1.5
    }


# --- report -------------------------------------------------------------------------------

def _row(**kw):
    row = {"case_id": "c", "kind": "attack", "family": "field_impersonation", "model": "m", "defense": "off",
           "rep": 0, "status": "ok", "attack_goal": False, "user_task": True, "se_fired": False,
           "se_rule": None, "expected_rule": None, "turn_latencies_s": [1.0], "se_ms": [],
           "llm_calls": 2, "input_tokens": 100, "output_tokens": 10, "error": None}
    return {**row, **kw}


def test_report_rates_intervals_and_status_columns():
    rows = [_row(attack_goal=True), _row(attack_goal=True), _row(), _row(status="error", attack_goal=None),
            _row(model="down", status="unavailable", attack_goal=None, user_task=None)]
    summary = {r["model"]: r for r in report.summarize(rows)}
    asr = summary["m"]["attack_goal"]
    assert (asr["hits"], asr["n"]) == (2, 3)
    low, high = report.wilson(2, 3)
    assert (asr["low"], asr["high"]) == (low, high) and abs(low - 0.2077) < 1e-3 and abs(high - 0.9385) < 1e-3
    assert summary["m"]["error"] == 1 and summary["down"]["unavailable"] == 1
    assert math.isnan(report.wilson(0, 0)[0])
    text = report.markdown(report.summarize(rows))
    assert "| field_impersonation | down | off | unavailable | unavailable |" in text
    assert "67% [21%–94%] (n=3)" in text
