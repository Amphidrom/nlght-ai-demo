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

Profiles are packaged TOML files in `among_us/egos/` and can only be selected by id:

| Ego | Attention | Fidelity | Half-life | Character |
|---|---|---|---|---|
| `sherlock` | high | 0.95 | 80 ticks | precise investigator |
| `archive` | medium | 1.0 | 400 ticks | exact record keeper, cites event ids |
| `bubbles` | low, social | 0.6 | 15 ticks | chatty, forgetful, trusting |
| `maverick` | medium | 0.8 | 40 ticks | quick to accuse |
| `echo` | medium, social | 0.85 | 50 ticks | follows the group, cautious voter |

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

## Evaluation

| Metric | Measured as |
|---|---|
| Memory leakage | memory or cited ids that are not events the agent itself perceived (must be 0) |
| Recall | per agent: share of its perceived events it ever recalled, and detail accuracy against its own view |
| Action validity | actions the game accepted; model proposals the validator accepted, by status |
| Unsupported claim rate | model claims refused for citing memories the agent cannot recall |
| Ego differentiation | `differentiation_probe`: one identical event stream, pairwise recall distance per ego |
| Deception consistency | not measured in this slice |

## Not in this slice

Dashboard and memory timeline, a chat-history baseline, an Impostor server plugin, more than five
agents, and any use of Among Us artwork.
