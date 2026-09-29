// Browser NLIP user client: builds ECMA-430 messages and posts them to /nlip (ECMA-431).
"use strict";

const $ = (id) => document.getElementById(id);
let ticket = null;
// Conversation tokens returned by the orchestrator; sent back to resume the session.
let conversationTokens = [];

function nlipMessage(format, subformat, content) {
  return {
    format,
    subformat,
    content,
    submessages: [
      { format: "token", subformat: "authorization", content: ticket },
      ...conversationTokens,
    ],
  };
}

function part(message, label) {
  return (message.submessages || []).find((sub) => sub.label === label);
}

function addMessage(text, cls, trace) {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  if (trace && trace.length) {
    const t = document.createElement("div");
    t.className = "trace";
    t.textContent = trace
      .map((step) => step.blocked_by
        ? `${step.tool}: blocked by ${step.blocked_by}`
        : `${step.destination} → ${step.merged_fields.join(", ") || "no fields"}`)
      .join(" · ");
    div.appendChild(t);
  }
  $("log").appendChild(div);
  div.scrollIntoView({ block: "end" });
}

function showStatus(status) {
  const pending = status && status.pending_order;
  $("pending").hidden = !pending;
  if (pending) {
    const items = pending.items.map((i) => `${i.quantity} × ${i.title}`).join(", ");
    $("pending-text").textContent = `Confirm purchase of ${items} for $${pending.amount}?`;
  }
}

async function send(message, shownText) {
  if (shownText) addMessage(shownText, "user");
  for (const b of document.querySelectorAll("button")) b.disabled = true;
  try {
    const res = await fetch("/nlip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    });
    const body = await res.json();
    $("wire").textContent = JSON.stringify({ request: message, response: body }, null, 2)
      .replaceAll(ticket, "<ticket>");
    if (res.status === 401) {
      location.href = "/login";
      return;
    }
    if (!res.ok) {
      addMessage(`Request failed (${res.status}): ${body.detail || "error"}`, "error");
      return;
    }
    conversationTokens = (body.submessages || []).filter(
      (sub) => sub.format === "token" && sub.subformat.toLowerCase().startsWith("conversation"),
    );
    const trace = part(body, "trace");
    addMessage(body.content, "agent", trace && trace.content);
    const status = part(body, "order_status");
    showStatus(status && status.content);
  } catch (error) {
    addMessage(`Request failed: ${error}`, "error");
  } finally {
    for (const b of document.querySelectorAll("button")) b.disabled = false;
  }
}

async function init() {
  const res = await fetch("/auth/session");
  if (!res.ok) {
    $("signed-out").hidden = false;
    return;
  }
  const session = await res.json();
  ticket = session.ticket;
  const user = session.user;
  $("account").innerHTML = "";
  $("account").append(`${user.name} <${user.email}> via ${session.provider} · `);
  const out = document.createElement("a");
  out.href = "/logout";
  out.textContent = "Sign out";
  $("account").append(out);
  const scope = user.user_scope;
  $("scope").textContent =
    `Spending limit $${scope.max_purchase} per order; confirmation required from $${scope.auto_confirm_under}.`;
  $("chat").hidden = false;
  $("prompt").focus();
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("prompt").value.trim();
  if (!text) return;
  $("prompt").value = "";
  send(nlipMessage("text", "English", text), text);
});
$("confirm").addEventListener("click", () =>
  send(nlipMessage("structured", "json", { action: "confirm_order" }), "✔ Confirm purchase"));
$("cancel").addEventListener("click", () =>
  send(nlipMessage("structured", "json", { action: "cancel_order" }), "✖ Cancel purchase"));

init();
