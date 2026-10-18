# Among Us agents (experimental)

nlght-ai agents with an ego and an imperfect memory play Among Us. The first vertical slice
runs against a deterministic simulation; a C# bridge (`among_us/bridge`) connects the same
protocol to the real game and is not built or tested yet.

## Architecture

```
GameEnvironment ── SimEnvironment  (among_us/sim, seed-deterministic)
                └─ BridgeEnvironment (NDJSON over 127.0.0.1:47100 → C# BepInEx/Reactor mod)
      │ Observation per agent (visibility-filtered) · SemanticAction · ActionResult
Match orchestrator (among_us/match.py, HTTP adapter among_us/adapter.py)
      │ one workflow run per agent and decision point, session "among-us:{match}:{agent}"
Workflow among_us_agent_turn (seed/08_among_us.sql)
      perceive → remember → believe → decide (model) → validate
Objective event log → replay + evaluation only, never agent input
```

The model is not in a frame loop. The environment decides when an agent is at a **decision
point** — goal reached, a salient observation (kill, body, meeting), a kill opportunity, a phase
change, a meeting turn or a vote. In between, the game executes the last semantic action
(walking takes one corridor per tick, a task three ticks).

## Scheduling

The environment decides *which* agents are at a decision point; `among_us/scheduler.py` decides
*how* each one decides:

| Rule | Setting (default) | Effect |
|---|---|---|
| Priority | — | meetings and votes, then salient observations (kill, body), then kill opportunities, then routine goals |
| Concurrency | `max_concurrent_model_turns` (4) | model turns running at once; the rest queue in priority order |
| Budget | `model_turns_per_tick` (8) | model turns per task-phase tick; agents beyond it decide by policy this tick. Meetings and votes are exempt |
| Deadline | `turn_timeout_seconds` (90) | a slower model turn is replaced by a policy turn |

A policy turn runs the same workflow with the model off: the agent still perceives, remembers and
believes. Every decision records its `scheduling` mode (`model`, `policy:budget`, `policy:timeout`,
`policy:offline`), and the match summary counts them. Settings go into the match request:

```json
{"seed": 7, "scheduler": {"max_concurrent_model_turns": 2, "model_turns_per_tick": 4}}
```

## The agent turn

| Step | What it does | Where state lives |
|---|---|---|
| `among_us_perceive` | Strictly parses the observation and refuses leaks (`perception.py`) | `ctx.metadata` |
| `among_us_remember` | Attention, encoding, retention per ego (`memory.py`); forgotten traces are deleted | slot `among_us:episodic` |
| `among_us_believe` | Deterministic suspicion from recallable memories only (`beliefs.py`) | slot `among_us:beliefs` |
| `among_us_decide` | Ego persona + rules as trusted instructions; observation, memories, beliefs as untrusted context; buffered `call_llm` | turn via `record_turn` |
| `among_us_validate` | Legal action? Evidence only from the agent's own recallable memories? Otherwise the deterministic fallback (`policy.py`) | result signal |

Memory holds only perceptions. A model's proposal or statement is recorded as a turn summary,
never written back as a fact. Every context record carries an untrusted content origin; other
players' chat is labelled `game_chat`, because an impostor's job is to lie.

## Egos

Fifteen profiles, one per seat of a full lobby, ship as `among_us/egos/<ego_id>.toml`. The file
name is the id; a request can only select a profile, never supply one.

`EgoProfile` in `among_us/ego.py` is the single definition of an ego file: strict types, ranges,
units, no unknown keys. Each parameter feeds exactly one formula, and no code branches on which ego
it runs:

| Section · parameter | Range | Consumer | Effect |
|---|---|---|---|
| `attention.default`, `attention.by_kind` | 0–1 | `memory.encode` | probability of noticing an event (kills, bodies, meetings: always) |
| `encoding.fidelity` | 0–1 | `memory.encode` | probability of keeping each detail; share of chat words kept |
| `encoding.detail_noise` | 0–1 | `memory.encode` | probability of remembering a neighbouring room instead |
| `retention.half_life_ticks` | > 0 ticks | `memory.strength` | exponential forgetting |
| `retention.recall_threshold` | 0–1 | `memory.recall` | minimum strength to recall; below half of it the trace is deleted |
| `judgement.suspicion_bias` | 0–3 | `beliefs.update_beliefs` | multiplier on incriminating evidence |
| `voice.max_words` | 5–120 | decide prompt, validator | statement length |
| `persona` | text | decide step | character, part of the trusted instructions |

`among_us/egos/ego.schema.json` is generated from the model (`python -m among_us.ego`) and
referenced by every file (`#:schema`), so TOML-aware editors validate while you edit. A test fails
if the committed schema drifts from the model.

| Ego | Attention | Fidelity | Noise | Half-life | Suspicion bias | Character |
|---|---|---|---|---|---|---|
| `archive` | 0.7 | 1.0 | 0.0 | 400 | 0.9 | Archive, a dry record keeper |
| `bubbles` | 0.45 | 0.6 | 0.15 | 15 | 0.7 | Bubbles, cheerful, chatty and easily distracted |
| `drifter` | 0.3 | 0.7 | 0.1 | 30 | 0.8 | Drifter, a daydreamer |
| `echo` | 0.6 | 0.85 | 0.05 | 50 | 0.8 | Echo, agreeable and cautious |
| `gossip` | 0.4 | 0.65 | 0.2 | 35 | 1.1 | Gossip |
| `guardian` | 0.7 | 0.85 | 0.04 | 90 | 1.1 | Guardian |
| `hawk` | 0.95 | 0.85 | 0.05 | 12 | 1.3 | Hawk, sharp-eyed and quick |
| `ledger` | 0.7 | 0.9 | 0.02 | 120 | 1.0 | Ledger, a timekeeper |
| `maverick` | 0.65 | 0.8 | 0.05 | 40 | 1.5 | Maverick, bold and impatient |
| `nova` | 0.6 | 0.75 | 0.08 | 45 | 1.4 | Nova, energetic and theatrical |
| `rookie` | 0.5 | 0.55 | 0.25 | 25 | 0.9 | Rookie, new to the game and a bit nervous |
| `sage` | 0.6 | 0.9 | 0.02 | 250 | 0.6 | Sage, calm and patient |
| `sherlock` | 0.9 | 0.95 | 0.02 | 80 | 1.2 | Sherlock, a meticulous investigator |
| `skeptic` | 0.75 | 0.9 | 0.03 | 70 | 0.6 | Skeptic |
| `tinker` | 0.4 | 0.85 | 0.05 | 60 | 0.8 | Tinker, absorbed in your tasks |

Every random choice draws from `sha256(match_seed | agent | event | purpose)`, so a match
replays exactly and differences between egos come from their profiles alone.

## Running a match

```powershell
psql $env:DATABASE_URL -f demo/seed/08_among_us.sql
uv run python main.py

# Simulated match with the model
Invoke-RestMethod -Method Post http://localhost:9000/among-us/api/matches `
  -ContentType application/json -Body '{"seed":7,"egos":["archive","bubbles","echo","maverick","sherlock"]}'

# The same without a model or database: every agent uses the deterministic policy
Invoke-RestMethod -Method Post http://localhost:9000/among-us/api/matches `
  -ContentType application/json -Body '{"seed":7,"offline":true}'
```

Then follow `GET /among-us/api/matches/{id}/events` (server-sent events) and, once finished,
`GET /among-us/api/matches/{id}/evaluation`. Replays are written to `.among-us-runs/`.

## Baseline: plain chat history

`among_us_agent_turn_baseline` is the same turn with one difference: `among_us_transcript` keeps
everything the agent perceived verbatim and never forgets (bounded at 300 events, as a real chat
history is bounded by the context window), and there are no beliefs. Perception, persona, rules,
validator, model provider and scheduling are identical, so a comparison isolates the memory
architecture. Ego and baseline agents can sit in the same match:

```json
{"seed": 7, "baseline": ["archive", "echo", "gossip"]}
```

The evaluation reports both groups side by side under `by_memory_mode`: decisions, model turns,
context size per model turn, crewmate vote accuracy and skip rate, recall coverage and detail
accuracy, unsupported claims. A baseline recalls everything exactly by construction; the question
the comparison answers is what the ego architecture buys in context size, and how it changes votes
and claims — which only a run with a real model shows.

## Location claims

A meeting statement can carry structured claims — `{"player": "echo", "room": "admin", "tick": 12}`
— next to its text. The validator accepts them only with a statement, for known players and past
ticks, and a claim about someone else only with `evidence_ids` the agent can recall; a claim about
oneself needs none. Claims stay on the orchestrator side (the bridge protocol is unchanged), and only
the evaluation compares them with where everyone really was. That is what makes deception
measurable without parsing free text: an impostor's false claim is a lie, a crewmate's false claim
is a mistake — usually a memory error the ego is allowed to make.

## Evaluation

| Metric | Measured as |
|---|---|
| Memory leakage | memory or cited ids that are not events the agent itself perceived (must be 0) |
| Recall | per agent: share of its perceived events it ever recalled, and detail accuracy against its own view |
| Action validity | actions the game accepted; model proposals the validator accepted, by status |
| Unsupported claim rate | model claims refused for citing memories the agent cannot recall |
| By memory mode | ego vs baseline: context size, crew vote accuracy and skip rate, recall, unsupported claims |
| Ego differentiation | `differentiation_probe`: one identical event stream, pairwise recall distance per ego |
| Claims | structured location claims (`player`, `room`, `tick`, ±1 tick) against true positions: impostor lies, the share of lies a living witness could refute, self-contradictions and deception consistency; crewmate mistakes per memory mode |

## Not in this slice

Dashboard and memory timeline, an Impostor server plugin, and any use of Among Us artwork.
