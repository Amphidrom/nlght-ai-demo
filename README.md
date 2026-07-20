# nlght-ai Demo Gallery

This repository is a local-first showcase for the nlght-ai Runtime Platform. It combines workflows, tools, model providers, persistent sessions, runtime isolation, custom HTTP triggers, and streaming into six outcome-focused examples.

## Showcase cases

1. **Product FAQ** — local manual retrieval with explicit source IDs and honest unknown handling.
2. **Shipping Quote** — validated input passed to one registered custom tool.
3. **Incident Triage** — normalize an alert, classify severity, and route the response.
4. **Policy-Aware Document Intake** — structured extraction, validation, and workflow routing.
5. **Preference Assistant** — persist only an explicit user preference in Hive Mind.
6. **Runtime Data Check** — inspect a small CSV with one fixed command in an isolated process.

Each case includes selectable scenarios in the portal. They are not cosmetic presets: the
variants exercise success, routing, validation, remembered session state, or an honest
unsupported outcome.

### What to try

- **Product FAQ:** compare a supported recovery question, a question combining two documented
  topics, and an unsupported satellite question. Answers expose the local manual source IDs;
  unsupported questions are not invented and explicitly request Human-in-the-Loop review.
- **Policy-Aware Document Intake:** a complete renewal passes `CONTRACT-01`; an incomplete
  renewal shows the missing identifiers; an attributed immediate Linux threat passes `SEC-01`
  and routes to `escalated_review`. The result includes every evaluated policy and its reason.
- **Preference Assistant:** first run **Remember concise style**, then **Apply in a later task**
  with the same `session_key`. The second result shows the loaded preference and a visibly
  concise answer. Change the session key to demonstrate session isolation.

The intake policies are intentionally transparent demo rules, not a production compliance
policy set.

## Start locally

Prerequisites are Python 3.11+, Docker, a valid nlght-ai license, PostgreSQL configured in `.config/platform.yaml`, and a local Ollama model.

```powershell
uv sync
uv run python main.py
```

Open <http://localhost:9000/demo>. The portal checks workflow availability before enabling a case. OpenAPI documentation is available at <http://localhost:9000/docs>.

Load or refresh the workflow definitions with the repository's normal PostgreSQL seed process:

From the parent directory of this repository:

```powershell
psql $env:DATABASE_URL -f demo/seed.sql
```

The default configuration uses a Docker Python image for the runtime case. It mounts a disposable
per-run directory below `.workspaces/` at `/workspace` inside the container. The six cases do not
require GitHub, web search, vector databases, or external knowledge stores.

The repository intentionally contains only the six active showcase packages, their shared portal
and invocation helper, configuration, seeds, tests, and documentation. Retired example modules are
not part of this gallery.

## API quickstart

List cases:

```bash
curl http://localhost:9000/demo/api/cases
```

Run the fully local Document Intake case:

```bash
curl -X POST http://localhost:9000/demo/api/cases/document-intake/runs \
  -H "Content-Type: application/json" \
  -d '{"message":"Today Acme Ltd submitted a renewal for contract AC-1042."}'
```

Use the returned `correlation_id` to inspect the run or subscribe to its events:

```bash
curl http://localhost:9000/demo/api/runs/CORRELATION_ID
curl -N http://localhost:9000/demo/api/runs/CORRELATION_ID/events
```

Run snapshots are stored under `.demo-runs/`. They contain public status and result signals, not credentials or full model-provider traces.

## Adding a case

Add the case metadata to `showcase/catalog.py`, register its custom steps in `main.py`, and add a versioned workflow definition under `seed/`. A publishable case needs a local sample input, a useful failure mode, and tests that assert its routing and result contract.

## Tests

The core showcase suite uses the Python standard library and does not require a running model or database:

```powershell
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Live model, GitHub, and web-search checks should remain separate optional end-to-end suites.
