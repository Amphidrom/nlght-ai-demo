# Among Us agents (experimental)

nlght-ai agents with an ego and an imperfect memory play Among Us. The first vertical slice
runs against a deterministic simulation; the nlght Impostor server (repository `nlght-ai-impostor`)
brings the same agents into the real game, one registered lobby at a time.

## Architecture

```
SimEnvironment (among_us/sim, seed-deterministic) ── run_match (among_us/match.py)
nlght Impostor server ── one session per lobby ──── LobbyRegistry (among_us/lobby.py)
      │ Observation per agent (visibility-filtered) · SemanticAction · ActionResult
HTTP adapter among_us/adapter.py
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

## The real game: nlght Impostor server

The real game is reached through `nlght-ai-impostor`, a fork of the open-source Among Us server
[Impostor](https://github.com/Impostor/Impostor) with the bot layer of
[among-us-llm](https://github.com/realmrhigh/among-us-llm) (both GPL-3.0; a separate repository and
process, so this demo is not affected by the license).

- **Every agent is a player.** In Among Us a player is a client connection, and the game logic runs
  on the host's client. The server therefore runs one *in-process* bot client per agent that speaks
  the client protocol into Impostor's message handler — no game installation per agent. A human
  hosts the lobby with the normal game and adds agents by chat command.
- **The body stays in the server, the mind is here.** Walking (path finding on the map graph),
  tasks at their consoles, kills, reports, the emergency button, chat and votes are executed by the
  bot. Every decision point — goal reached, a kill or body seen, a kill opportunity, each meeting
  turn and the vote — becomes a `decision_request`; the agent's `SemanticAction` comes back over the
  same protocol. Without an answer in time, the bot's built-in heuristic decides.
- **Vision is filtered at the source.** A network client receives everyone's position; the bot
  reports only what its own character could see, and `perception.py` re-checks it here.
- **The orchestrator names the bots.** When the host types `!bots N`, the server registers the lobby
  here before a single bot exists; the registry picks N egos that are free in this lobby and whose
  names no player in it uses, and the server spawns exactly those bots under exactly those names
  (`agent_id` is always the lower-cased name). Skeld rooms use the simulation's room names, so the
  dashboard map and location claims work unchanged.
- `among_us/protocol_fixtures/server_decision_request.json` is a message the server produced in its
  test suite; `tests/test_among_us.py` holds the contract.

### Lobby sessions (protocol v2)

Nothing is started by hand here: a lobby exists once its host asks for bots, and it lives across
its games.

| Step | Request | Effect |
|---|---|---|
| `!bots N` | `POST /among-us/api/lobbies` `{protocol, game_code, game_version, source, count, players}` | `{lobby_id, token, agents: [{agent_id, name, ego}]}` |
| more bots | `POST /among-us/api/lobbies/{id}/agents` `{protocol, count, players}` | further agents for the same lobby |
| game messages | `POST /among-us/api/lobbies/{id}/messages`, one message per request | `202`; agent turns run in the background |
| actions | `GET /among-us/api/lobbies/{id}/stream` | server-sent events, one `action` per event |
| `!bots off` | `DELETE /among-us/api/lobbies/{id}` | the lobby and a running match end |

Every request except the registration carries `Authorization: Bearer <token>`. A token belongs to
exactly one lobby: it can neither read nor write another, and one slow or broken lobby does not
hold up the rest.

The server posts:

- `game_started` `{map, players}` — a new match begins; its agents start with an empty memory.
- `observation` `{observation}` — what an agent perceived since its last message; in the task phase
  with the actions the game would allow right now. The turn runs perceive, remember and believe and
  stops before decide: memory grows with every event, not only at decision points. The game is not
  answered — but the agent may act on its own (see below).
- `decision_request` `{request_id, deadline_seconds, observation}` — the agent is at a decision
  point. Its turn starts at once, in parallel with every other agent's, and runs for as long as its
  model needs: lobbies have no concurrency cap, budget or timeout. `deadline_seconds` is how long the
  server waits (`Nlght:DecisionTimeoutSeconds`, 120 s by default) before its heuristic takes over. A
  newer request of the same agent cancels an unanswered older one.
- `action_result` `{agent_id, accepted, reason, request_id}` — the game's verdict, recorded on the
  decision.
- `agent_left` `{agent_id, reason}` and `game_ended` `{winner, reason, roles}` — `roles` is every
  player's team role, humans included, kept on the match for evaluation only.

The orchestrator streams `action` `{request_id, …SemanticAction}`: an answer to a request, or with
`request_id: null` an action on the agent's own initiative, which the server checks against the game
as it is at that moment and judges with an `action_result` without `request_id`.

### Two kinds of turns, one stance

A game turn has to be fast, a meeting turn may take its time. Both start from the agent's
**stance**, which the decide step assesses deterministically (`among_us/assess.py`), without the
model:

- **Situation.** Every observation carries `situation` — the crew's task bar (null when the lobby
  hides it or comms are down), the agent's own emergency meetings left, its kill cooldown in ticks,
  ticks since the last meeting — and the assessment adds who is alive and dead as far as the agent
  knows, and the stage of the game (early, mid, late).
- **Suspicion and trust.** Beliefs carry `trust` next to `suspicion`: seeing someone do tasks builds
  it, and so do the agent's own conclusions.
- **Plan.** A tactic is held across turns with its reason and the condition that ends it, in the
  Hive Mind slot `among_us:plan`: crew — report, fix a sabotage, call a meeting when sure enough,
  escort someone trusted late in the game, do tasks, then shadow a suspect, stay with someone
  trusted or watch with the group; impostor — escape after a kill, strike when alone with a
  victim, stalk a lone crewmate, sabotage the lights before striking, fake tasks where others see,
  blend in. Only something urgent overrides a valid plan. How much an agent leans which way is its
  ego's: `judgement.trust_at`, `group_affinity` and `boldness`.

**Game turn** (`among_us/turn.py`): the context is a few lines of plain text — who the agent is,
where the game stands, its plan, what it sees, its six most important memories, its suspects and
whom it trusts — followed by the legal moves as numbered options, the plan's first, each explained
in a few words (`VENT electrical (anyone in the room sees you)`). The model answers with a number.
`FOLLOW <player>` keeps someone in sight; `GUARD` stays and watches.

**Meeting turn**: the whole chat, every memory the agent can still recall, the contradictions
between what was said and what it saw (`hawk says they were in medbay; you saw hawk in electrical at
t20 (m5)`), its suspects and trust with reasons. The model may think first; only the last lines
count: `SAY … | m2 m5`, `VOTE <player> | …`, `SKIP` or `WAIT`, and optionally `MIND hawk+ nova- echo=`
— how its view of the others changed. A contradiction and a `MIND` verdict both become **conclusions**:
memories of their own (`JUDGED`), weighed in the beliefs and forgotten like any other memory, so
what the meeting settled carries into the game turns that follow.

No JSON goes into or comes out of the model. An answer that is not understood (`no_proposal`) or not
legal (`illegal`) is replaced by the plan's move, else the policy (`decided_by`: model, plan or
policy). Where a statement puts whom is read from its text (`assess.room_claims`) for the evaluation.
Every turn records how long each step took (`timings`: perceive, remember, believe, decide, model,
validate); the admin view shows them per agent.

**Meetings as a conversation.** In meetings and votes every observation carries `meeting_chat`: the
meeting's chat so far, word for word, as it stands on everyone's screen (what the agent *keeps* of it
is still its memory's business). The decide step then adds meeting rules to the instructions —
answer whoever addressed or accused you, weigh others' claims against your memories, agree or push
back with a reason, ask pointed questions, change your mind openly, never repeat yourself (the
context lists `you_said`), and WAIT when there is nothing new. Without the model, the policy does
the same in small: it defends itself by name when accused, accuses with evidence, sides with or
questions someone else's accusation (an impostor never piles onto its partner), asks, and falls
silent instead of repeating a line.

**Initiative.** After every observation, `among_us/initiative.py` asks — without the model — whether a
belief has grown strong enough to act on before the game asks. A crewmate whose suspicion of a
living player reaches its ego's `judgement.call_meeting_at` calls an emergency meeting, citing the
memories behind it; evidence it already acted on never triggers a second call (Hive Mind slot
`among_us:initiative`, per match). The thresholds differ by temperament — Maverick 0.45, Hawk 0.5,
Sherlock 0.75, Sage 0.9, Echo and Rookie 0.95 — so the same evidence makes one agent press the button
and leaves another waiting for the next meeting. Each initiative is an `initiative` entry in the match
log with its suspect, suspicion and the game's verdict.

`GET /among-us/api/lobbies` and `GET /among-us/api/lobbies/{id}` list the lobbies with their state,
agents (ego, role, alive, room, last action, latency, decisions by scheduling mode, refused answers,
initiatives sent, accepted and refused) and matches. Each match of a lobby is an ordinary match under `/among-us/api/matches/{id}`.

The registry needs the seeded workflow (`seed/08_among_us.sql`, loaded once): without it a
registration is refused with `503`, so no lobby ever plays without its agents. The same happens with
an older seed whose decide step still pins `model_provider` — it would override every agent's own
model; loading the file again updates it.

### Admin view

The dashboard at `/among-us` lists the registered lobbies above the simulation controls: game code,
state, matches (each opens the live match view), and per agent its role (always shown — no player has
access to this page), alive or dead (dead as soon as any agent perceived the death), room, last
action, latency, decisions by scheduling mode, refused answers, initiatives and its model. A click on
an agent shows what it suspects and what it recalls after its latest turn.

### One model per agent

Every agent decides with its own provider and model, resolved in this order
(`among_us/models.py`):

1. the operator's choice in the admin view — `PUT /among-us/api/lobbies/{id}/agents/{agent}/model`
   with `{"provider", "model"}`, or `{"provider": null}` to go back; it holds from the agent's next
   turn,
2. the ego file's optional `[model]` table — `provider` (a provider name from `platform.yaml`) and
   `name` (omitted: that provider's default model),
3. the platform default: the first configured provider and its default model.

Meeting turns may use a model of their own — `[model.meeting]` in the ego file, a meeting choice in
the admin view (`"phase": "meeting"`); without one the game model decides there too.

```toml
[model]
provider = "ollama"
name = "qwen3-coder:30b"

[model.meeting]
provider = "anthropic"
```

Which models exist comes from the providers themselves: `GET /among-us/api/models` asks each
provider's nlght backend (`ModelProviderBackend.list_models()`, cached for a minute), and an
operator's choice is checked against that list. Every turn carries `model_provider` and `model`,
which the nlght executor binds for the decide step. An agent whose provider is not configured in this
process decides by policy, counted as `policy:no-model`, and the admin view says so.

nlght-ai 0.1.4 does not expose the provider backends on its container; `catalog_from_container` reads
them from the workflow executor (`_model_clients`) until it does.

Everything in *Playing the impostor* works on the server too: lights and comms sabotage (in the
real game lights out shrink a crewmate's sight rather than hiding names), doors, and the Shapeshifter
and Phantom roles, with perception following appearance there as well.

There is no spectator ground truth for the real game yet: `game_ended` reports the roles, but not
who saw what, so leakage, recall and claims cannot be scored against the truth there.

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

## Playing the impostor

Any agent can be dealt the impostor role; the game, not the agent, decides. An impostor agent knows
its partners and gets these extra legal actions, in the simulation and on the server alike:

| Action | Effect | What others perceive |
|---|---|---|
| `KILL <player>` | only a crewmate it can see; it remembers the kill as `KILL_COMMITTED` (only the killer perceives that kind) | `KILL_WITNESSED` for anyone in sight |
| `VENT <room>` | travel through the vent network to a reachable room | `VENT_SEEN` for anyone at the entry or exit — nearly as damning as a witnessed kill |
| `SABOTAGE reactor\|o2` | a critical alarm; the crew loses unless someone reaches the room in time (fixing is a reflex) | `SABOTAGE_ALARM` for everyone, and `alarm` in every observation; nobody perceives who pulled it |
| `REPORT <own victim>` | self-reporting as cover is allowed | an ordinary report |
| `SABOTAGE lights` | crewmates still see that someone is there, but not who — sightings, kills and vents in the dark have no actor; fixed in electrical | `SABOTAGE_ALARM`, `outages: ["lights"]` |
| `SABOTAGE comms` | crewmates cannot see their task list and cannot do tasks; fixed in communications | `SABOTAGE_ALARM`, `outages: ["comms"]` |
| `CLOSE_DOORS <room>` | nobody walks in or out for a while (vents still work) | `DOORS_CLOSED`, heard in the room and next to it, nobody perceives who |

Reactor and O2 are critical: unfixed, the impostors win, and the emergency button is disabled.
Lights and comms are not critical. Only one sabotage runs at a time; doors have their own cooldown.

### Special roles

`impostor_roles` (e.g. `["SHAPESHIFTER"]`) hands special roles to impostors; an agent knows only its
own (`self_state.special_role`).

- **Shapeshifter** — `SHAPESHIFT <player>`: for a while it *looks like* that player. Everyone else
  perceives the disguise, so a witness of a kill blames the copied player (`KILL_WITNESSED` names
  the disguise). Changing shape and the shape wearing off are visible (`SHAPESHIFT_SEEN`); the
  wearing off shows the true face.
- **Phantom** — `VANISH` / `APPEAR`: while invisible it is in nobody's perception — no sightings, no
  vent sightings — and it cannot kill. Vanishing and appearing in front of others are visible
  (`VANISH_SEEN`, `APPEAR_SEEN`).

Perception follows appearance throughout; at a meeting every disguise drops and every phantom is
back. `impostor_play` adds misattributed and unidentified kill sightings, shapeshifts, vanishes,
doors closed and sabotages per system.

Fake tasks look like real ones to everyone else. After a kill the impostor is at a decision point
again (`observed: KILL_COMMITTED`). The evaluation adds `impostor_play`: kills and how many were
witnessed, vent uses and sightings, sabotages started, fixed and won, self-reports.

## Location claims

A meeting statement can carry structured claims — `{"player": "echo", "room": "admin", "tick": 12}`
— next to its text. The validator accepts them only with a statement, for known players and past
ticks, and a claim about someone else only with `evidence_ids` the agent can recall; a claim about
oneself needs none. Claims stay on the orchestrator side (the game protocol does not carry them), and only
the evaluation compares them with where everyone really was. That is what makes deception
measurable without parsing free text: an impostor's false claim is a lie, a crewmate's false claim
is a mistake — usually a memory error the ego is allowed to make.

## Spectator dashboard

`http://localhost:9000/among-us` starts a match (seats, baseline seats, seed, impostors, offline) and
follows it live:

- **Ship map** from `GET /among-us/api/matches/{id}/spectator` — where everyone really is, bodies,
  and roles on demand. This is the spectator's ground truth (simulation only); it never enters an
  agent's context.
- **Decision feed** from the event stream: scheduling mode, reason, action, statement, and why a
  model proposal was refused.
- **Agent cards**: ego, memory mode, status, stored/recalled/forgotten memories with a timeline,
  the current prime suspect, the last statement.
- **Evaluation** once the match ends: leakage, validity, claims and deception, ego vs baseline,
  recall per agent.

Agent and model text is rendered as text only, never as markup.

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

A test against a real Among Us client, ground truth from the real game for scoring, vents and
special roles, and any use of Among Us artwork.
