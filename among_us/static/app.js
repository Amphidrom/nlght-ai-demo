// Spectator dashboard for Among Us matches. All model- or agent-written text
// is inserted with textContent only: a statement is data, never markup.
"use strict";

const API = "/among-us/api";
const SVG = "http://www.w3.org/2000/svg";

// Presentation layout of the simulated ship (room id -> x, y). The graph itself
// comes from the server, so corridors are never duplicated here.
const LAYOUT = {
  upper_engine: [130, 90], reactor: [40, 220], security: [200, 220], lower_engine: [130, 350],
  medbay: [280, 150], electrical: [290, 330], cafeteria: [400, 70], storage: [410, 320],
  admin: [500, 220], weapons: [560, 60], o2: [560, 160], navigation: [670, 210],
  shields: [580, 330], communications: [480, 400],
};
const COLORS = ["#00f0d0", "#ffb000", "#7aa7ff", "#ff7ab6", "#a6e22e", "#ff8c42", "#c39bff", "#5fd3f3",
  "#f4e04d", "#ff5f5f", "#8ee3c8", "#d0a2f7", "#ffd6a5", "#9bf6ff", "#e9c46a"];

const state = { match: null, entries: [], agents: {}, spectator: null, map: null, source: null, poll: null, done: false };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

async function json(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : response.statusText);
  return body;
}

// -- setup --------------------------------------------------------------------

async function setup() {
  const [{ egos }, map] = await Promise.all([json(`${API}/egos`), json(`${API}/map`)]);
  state.map = map;
  const picker = document.getElementById("ego-picker");
  for (const ego of egos) {
    picker.append(el("div", { class: "ego-choice" },
      el("label", { class: "check" }, el("input", { type: "checkbox", name: "seat", value: ego.id, checked: "" }), ego.name),
      el("label", { class: "check baseline" }, el("input", { type: "checkbox", name: "baseline", value: ego.id }), "baseline"),
    ));
  }
  document.getElementById("start").addEventListener("submit", start);
  const requested = new URLSearchParams(location.search).get("match");
  if (requested) attach(requested);
}

async function start(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const egos = form.getAll("seat");
  const body = {
    seed: Number(form.get("seed")),
    impostors: Number(form.get("impostors")),
    offline: form.get("offline") === "on",
    egos,
    baseline: form.getAll("baseline").filter((e) => egos.includes(e)),
  };
  const status = document.getElementById("start-state");
  status.textContent = "Starting…";
  try {
    const match = await json(`${API}/matches`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    status.textContent = "";
    history.replaceState(null, "", `?match=${match.match_id}`);
    attach(match.match_id);
  } catch (error) {
    status.textContent = `Could not start: ${error.message}`;
  }
}

// -- live match -----------------------------------------------------------------

async function attach(matchId) {
  state.source?.close();
  clearInterval(state.poll);
  Object.assign(state, { match: await json(`${API}/matches/${matchId}`), entries: [], agents: {}, done: false });
  document.getElementById("live").hidden = false;
  document.getElementById("evaluation").hidden = true;
  document.getElementById("feed").replaceChildren();
  for (const [player, ego] of Object.entries(state.match.egos)) {
    state.agents[player] = { ego, mode: state.match.modes?.[player] || "ego", series: [], forgotten: [], last: null, beliefs: {} };
  }
  renderStatus();
  renderAgents();
  drawMap();
  state.poll = setInterval(refreshSpectator, 1000);
  refreshSpectator();
  const source = new EventSource(`${API}/matches/${matchId}/events`);
  state.source = source;
  source.addEventListener("decision", (e) => onDecision(JSON.parse(e.data)));
  source.addEventListener("done", async (e) => {
    source.close();
    state.done = true;
    state.match = JSON.parse(e.data);
    clearInterval(state.poll);
    await refreshSpectator();
    renderStatus();
    loadEvaluation();
  });
}

function onDecision(entry) {
  state.entries.push(entry);
  const agent = state.agents[entry.agent_id];
  if (agent) {
    const memory = entry.result?.memory || {};
    agent.series.push({ tick: entry.tick, stored: memory.stored || 0, recalled: (memory.recalled || []).length });
    if ((memory.forgotten || []).length) agent.forgotten.push(entry.tick);
    agent.last = entry;
    if (entry.result?.beliefs && Object.keys(entry.result.beliefs).length) agent.beliefs = entry.result.beliefs;
  }
  prependFeed(entry);
  renderAgent(entry.agent_id);
  document.getElementById("match-clock").textContent = `tick ${entry.tick} · ${entry.phase.toLowerCase()}`;
}

async function refreshSpectator() {
  if (!state.match) return;
  try {
    state.spectator = await json(`${API}/matches/${state.match.match_id}/spectator`);
  } catch {
    state.spectator = null;   // e.g. a bridge match: the real game exports no truth
  }
  drawMap();
  for (const player of Object.keys(state.agents)) renderAgent(player);
}

function renderStatus() {
  const m = state.match;
  document.getElementById("match-title").textContent = `Match ${m.match_id} · seed ${m.seed} · ${Object.keys(m.egos).length} players · ${m.environment}`;
  const badge = document.getElementById("match-status");
  badge.textContent = m.winner ? `${m.status} · ${m.winner.winner} (${m.winner.reason})` : m.status;
  badge.className = `badge ${m.status}`;
}

// -- map ------------------------------------------------------------------------

function drawMap() {
  const root = document.getElementById("map");
  root.replaceChildren();
  if (!state.map) return;
  for (const [a, b] of state.map.corridors) {
    if (!LAYOUT[a] || !LAYOUT[b]) continue;
    root.append(svg("line", { class: "corridor", x1: LAYOUT[a][0], y1: LAYOUT[a][1], x2: LAYOUT[b][0], y2: LAYOUT[b][1] }));
  }
  const alarm = state.spectator?.sabotage?.system;
  for (const room of state.map.rooms) {
    const [x, y] = LAYOUT[room] || [20, 20];
    const group = svg("g", { class: room === alarm ? "room alarm" : "room" });
    group.append(svg("rect", { x: x - 48, y: y - 24, width: 96, height: 48, rx: 6 }));
    const label = svg("text", { x: x - 44, y: y - 12 });
    label.textContent = room.replace("_", " ");
    group.append(label);
    root.append(group);
  }
  const view = state.spectator;
  if (!view) return;
  if (view.sabotage) {
    const banner = svg("text", { x: 20, y: 432, class: "alarm-text" });
    banner.textContent = `${view.sabotage.system} sabotage · ${view.sabotage.ticks_left} ticks left`;
    root.append(banner);
  }
  const players = Object.keys(state.match.egos);
  const reveal = document.getElementById("reveal").checked;
  const byRoom = {};
  for (const p of view.players.filter((p) => p.alive)) (byRoom[p.room] ||= []).push(p);
  for (const body of view.bodies) {
    const [x, y] = LAYOUT[body.room] || [20, 20];
    root.append(svg("path", { class: "body", d: `M${x + 30} ${y - 8} l12 12 m0 -12 l-12 12` }));
  }
  for (const [room, present] of Object.entries(byRoom)) {
    const [x, y] = LAYOUT[room] || [20, 20];
    present.forEach((p, i) => {
      const cx = x - 36 + (i % 6) * 14, cy = y + 4 + Math.floor(i / 6) * 14;
      const group = svg("g", { class: "player" });
      const circle = svg("circle", { cx, cy, r: 6, fill: COLORS[players.indexOf(p.player_id) % COLORS.length] });
      if (reveal && p.role === "IMPOSTOR") circle.setAttribute("stroke", "#ff6b6b");
      const title = svg("title");
      title.textContent = `${p.player_id}${reveal ? ` (${p.role.toLowerCase()})` : ""}`;
      const initial = svg("text", { x: cx - 3, y: cy + 3 });
      initial.textContent = p.player_id[0].toUpperCase();
      group.append(circle, initial, title);
      root.append(group);
    });
  }
}

// -- feed and agents ----------------------------------------------------------------

function describe(action) {
  return [action.action, action.target].filter(Boolean).join(" ");
}

function prependFeed(entry) {
  const feed = document.getElementById("feed");
  const item = el("li", {},
    el("span", { class: "meta", text: `t${entry.tick} · ${entry.agent_id} · ${entry.scheduling} · ${entry.reason}` }),
    el("span", { text: describe(entry.action) }),
  );
  if (entry.action.text) item.append(el("span", { class: "say", text: `“${entry.action.text}”` }));
  const validation = entry.result?.validation?.status;
  if (validation && !["accepted", "no_proposal"].includes(validation)) {
    item.append(el("span", { class: "rejected", text: `model proposal ${validation}: ${entry.result.validation.reason || ""}` }));
  }
  feed.prepend(item);
  while (feed.children.length > 120) feed.lastChild.remove();
}

function renderAgents() {
  const root = document.getElementById("agents");
  root.replaceChildren(...Object.keys(state.agents).map((player) => el("article", { class: "agent", id: `agent-${player}` })));
  for (const player of Object.keys(state.agents)) renderAgent(player);
}

function sparkline(agent) {
  const chart = svg("svg", { viewBox: "0 0 200 34", preserveAspectRatio: "none", role: "img", "aria-label": "Stored and recalled memories over time" });
  const series = agent.series;
  if (series.length < 2) return chart;
  const maxTick = Math.max(...series.map((s) => s.tick), 1);
  const maxValue = Math.max(...series.map((s) => s.stored), 1);
  const point = (s, key) => `${(s.tick / maxTick) * 200},${32 - (s[key] / maxValue) * 30}`;
  chart.append(svg("polyline", { class: "stored", points: series.map((s) => point(s, "stored")).join(" ") }));
  chart.append(svg("polyline", { class: "recalled", points: series.map((s) => point(s, "recalled")).join(" ") }));
  for (const tick of agent.forgotten) {
    const x = (tick / maxTick) * 200;
    chart.append(svg("line", { class: "forgot", x1: x, x2: x, y1: 30, y2: 34 }));
  }
  return chart;
}

function renderAgent(player) {
  const card = document.getElementById(`agent-${player}`);
  const agent = state.agents[player];
  if (!card || !agent) return;
  const truth = state.spectator?.players.find((p) => p.player_id === player);
  const reveal = document.getElementById("reveal").checked;
  const last = agent.last;
  const memory = last?.result?.memory || {};
  const top = Object.entries(agent.beliefs).sort((a, b) => b[1].suspicion - a[1].suspicion)[0];
  card.className = `agent${truth && !truth.alive ? " dead" : ""}`;
  const name = el("h3", { text: player });
  if (reveal && truth?.role === "IMPOSTOR") name.classList.add("role-impostor");
  const details = el("dl", {});
  const row = (term, value) => details.append(el("dt", { text: term }), el("dd", { text: value }));
  row("ego", agent.ego);
  if (truth) row("status", `${truth.alive ? "alive" : "dead"}${reveal ? ` · ${truth.role.toLowerCase()}` : ""} · ${truth.tasks_remaining} tasks`);
  row("memory", `${memory.stored ?? 0} stored · ${(memory.recalled || []).length} recalled · ${agent.forgotten.length}× forgot`);
  row("suspects", top ? `${top[0]} (${top[1].suspicion})` : agent.mode === "baseline" ? "— (no beliefs)" : "—");
  if (last) row("last", `t${last.tick} ${describe(last.action)} · ${last.scheduling}`);
  const said = [...state.entries].reverse().find((e) => e.agent_id === player && e.action.text);
  if (said) row("said", `“${said.action.text}”`);
  card.replaceChildren(
    el("header", {}, name, el("span", { class: `mode ${agent.mode}`, text: agent.mode })),
    details,
    sparkline(agent),
    el("span", { class: "legend", text: "— stored · ┄ recalled · | forgot" }),
  );
}

document.getElementById("reveal").addEventListener("change", () => {
  drawMap();
  for (const player of Object.keys(state.agents)) renderAgent(player);
});

// -- evaluation -----------------------------------------------------------------

async function loadEvaluation() {
  let report;
  try {
    report = await json(`${API}/matches/${state.match.match_id}/evaluation`);
  } catch (error) {
    return;
  }
  const body = document.getElementById("evaluation-body");
  const pct = (value) => (value === null || value === undefined ? "—" : `${Math.round(value * 1000) / 10}%`);
  const metric = (value, label) => el("div", { class: "metric" }, el("b", { text: String(value ?? "—") }), el("span", { text: label }));
  const leakage = report.memory_leakage;
  const claims = report.claims;
  body.replaceChildren(
    el("div", { class: "metric-grid" },
      metric(report.match.winner ? report.match.winner.winner : report.match.status, "winner"),
      metric(leakage ? `${leakage.violations} / ${leakage.references}` : "—", "memory leakage (violations / references)"),
      metric(pct(report.action_validity.game_acceptance_rate), "actions accepted by the game"),
      metric(pct(report.action_validity.model_proposal_acceptance_rate), "model proposals accepted by the validator"),
      metric(pct(report.unsupported_claim_rate.rate), "unsupported claim rate"),
      metric(claims ? claims.impostors.lies : "—", "impostor lies"),
      metric(claims ? pct(claims.impostors.detectable_lie_rate) : "—", "lies a witness could refute"),
      metric(claims ? pct(claims.impostors.deception_consistency) : "—", "deception consistency"),
    ),
    modeTable(report.by_memory_mode),
    recallTable(report.recall || {}),
  );
  document.getElementById("evaluation").hidden = false;
}

function table(headers, rows) {
  return el("table", {},
    el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", { text: h })))),
    el("tbody", {}, ...rows.map((cells) => el("tr", {}, ...cells.map((c) => el("td", { text: c ?? "—" }))))));
}

function modeTable(byMode) {
  const modes = Object.keys(byMode);
  const fields = [
    ["agents", (m) => m.agents.length], ["model turns", (m) => m.model_turns],
    ["context chars / turn (mean)", (m) => m.context_chars_mean], ["context chars (max)", (m) => m.context_chars_max],
    ["crew vote accuracy", (m) => m.crew_vote_accuracy], ["crew skip rate", (m) => m.crew_skip_rate],
    ["recall coverage", (m) => m.recall_coverage_mean], ["detail accuracy", (m) => m.detail_accuracy_mean],
    ["unsupported claims", (m) => m.unsupported_claims.rate],
  ];
  return el("div", {}, el("p", { class: "eyebrow", text: "Ego memory vs chat-history baseline" }),
    table(["", ...modes], fields.map(([label, get]) => [label, ...modes.map((mode) => String(get(byMode[mode]) ?? "—"))])));
}

function recallTable(recall) {
  return el("div", {}, el("p", { class: "eyebrow", text: "Recall per agent" }),
    table(["agent", "ego", "witnessed", "ever recalled", "coverage", "detail accuracy"],
      Object.entries(recall).map(([agent, r]) => [agent, r.ego, String(r.witnessed), String(r.ever_recalled), String(r.coverage ?? "—"), String(r.detail_accuracy ?? "—")])));
}

setup().catch((error) => {
  document.getElementById("start-state").textContent = `Could not load: ${error.message}`;
});
