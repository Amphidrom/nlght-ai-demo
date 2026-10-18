# nlght-ai Demo Gallery

This repository is a local-first showcase for the nlght-ai Runtime Platform. It combines workflows, real model calls, tools, persistent sessions, runtime isolation, custom HTTP triggers, and streaming into six outcome-focused examples.

## Showcase cases

1. **Product FAQ** — local manual retrieval with explicit source IDs and honest unknown handling.
2. **Shipping Quote** — validated input passed to one registered custom tool.
3. **Incident Triage** — normalize an alert, classify severity, and route the response.
4. **Policy-Aware Document Intake** — structured extraction, validation, and workflow routing.
5. **Preference Assistant** — persist only an explicit user preference in Hive Mind.
6. **Runtime Data Check** — inspect a small CSV with one fixed command in an isolated process.
7. **Knowledge Base** *(optional stack)* — ingest a handbook into reviewed claims, then answer from them.

Cases 1–6 run on the base stack. Case 7 needs the optional knowledge stack described below and is
shown as unavailable until it is set up.

Each case includes selectable scenarios in the portal. They are not cosmetic presets: the
variants exercise success, routing, validation, remembered session state, or an honest
unsupported outcome.

Every successful case calls the configured model with the visitor's concrete input and grounded
case evidence. Retrieval records, tool prices, routing rules, persisted preferences, and runtime
reports remain authoritative; the model answers or explains from those values instead of returning
hard-coded assistant prose.

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

- **Knowledge Base:** run **Step 1 · Ingest the handbook** once, then ask a question. The answer
  comes from claims a model extracted in the *earlier* run — the handbook itself is never in the
  question's request. Ask the satellite question to see the case refuse rather than improvise.

The intake policies are intentionally transparent demo rules, not a production compliance
policy set.

## Platform requirement

This gallery targets **nlght-ai 0.1.4 or newer** (`nlght-ai[docker,hive-mind]>=0.1.4`). It does not
run on 0.1.3 or older: every case relies on the typed-message and action-semantics APIs introduced in
0.1.3, and on provider-configured sampling and content origins introduced in 0.1.4.

**Typed messages with explicit authority.** A step no longer hands the model a list of
`{"role": ...}` dictionaries. Instructions are `TrustedInstructionMessage` — the only message type
that reaches a provider's system channel, and one only installed deployment code can construct.
Everything the runtime *established* rather than authored — a retrieved manual passage, a tool
result, a deterministic policy evaluation, a sandboxed report, a persisted preference — becomes a
`ContextRecord` and reaches the model as escaped, inert `untrusted_data`. `compose_prompt()` joins
the two domains and `compose_messages()` places the context immediately before the current request:

```python
records = [evidence(
    record_id="manual:MANUAL-204",
    kind=ContextKind.RETRIEVAL,
    content=article,
    source="product_manual",
    document_id="MANUAL-204",
)]
messages = compose_messages(compose_prompt(_INSTRUCTIONS, records), conversation)
await ctx.llm.call(messages=messages)
```

`grounding/context.py` holds the two helpers every case shares; each step still builds and composes
its own records, because that boundary is the thing worth reading.

**Mandatory action semantics on tools.** A `ToolSignature` without an `action` is *unknown*, never
read-only, and an execution-capable catalog refuses it — `ShippingRateTool` declares its rate table
as a pure request-scoped calculation with no side effect and no egress. Steps invoke tools only
through the step's tool catalog, which validates and binds arguments, resolves those semantics, and
applies policy before any method runs.

**Sampling is provider configuration.** Since 0.1.4 `ctx.llm.call()` takes no `temperature`; passing
one is a `TypeError`. The value lives on the model provider in `platform.yaml` and is sent on every
call through it. The demo configures two providers on the same local model:

| Provider | `temperature` | Used by |
|---|---|---|
| `ollama` | `0.1` | every showcase answer step |
| `ollama-extract` | `0` | `knowledge.extract` in the Knowledge Base case only |

A step selects its provider with `model_provider` in its seeded configuration, so a different
sampling setting is a different provider, never a per-call argument.

**Content carries its origin.** Since 0.1.4 every piece of content knows where it came from, and that
origin reaches each action decision. The Shipping Quote case shows the full path:

```python
conversation = [UserMessage(json.dumps({"shipment": shipment}))]
quote = await ctx.tools.execute_with_origin(
    {"name": "shipping_quote", "input": shipment},
    context_origin=origin_of(conversation),   # the caller's shipment: untrusted
)
records = [evidence(..., content=quote.content, origin=quote.origin)]
```

The decision sees that the shipment came from the caller, and `ShippingRateTool` returns a
`ToolOutput` labelled as trusted `rate_table` content — a bare string would be unannotated, which
counts as untrusted. In the Knowledge Base case the store lists the shipped handbook as
`trusted_sources: ["northstar-handbook"]`, so its claims are trusted content. Trust labels content;
it never turns context into instructions and never authorizes an action by itself.

`docs/` and the tests assert both boundaries: evidence must appear in the untrusted context of a
call and must *not* appear in its trusted instructions.

## Start locally

Prerequisites are Python 3.11+, Docker, a valid nlght-ai license, PostgreSQL configured in
`.config/platform.yaml`, and a local Ollama installation.

### Ollama and required models

Ollama must be installed and its local service must be running before the demo starts. nlght-ai
connects to the local Ollama API at `http://localhost:11434`; having only the CLI installed is not
enough.

Start Ollama if it is not already running:

```powershell
ollama serve
```

In another terminal, verify the server and list locally available models:

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

The model used by all six LLM-backed cases is the `default_model` of the configured `ollama`
provider. The repository currently references these models:

| Configuration | Current default model | When it is needed |
|---|---|---|
| `.config/platform.yaml` | `qwen3:30b-a3b-q4_K_M` | Default local development start |
| `docker/platform.yaml` | `qwen3-coder:30b` | When this checked-in example config is selected |

Pull the model for the configuration you intend to use before starting the application:

```powershell
# Default local configuration
ollama pull qwen3:30b-a3b-q4_K_M

# Only when using docker/platform.yaml
ollama pull qwen3-coder:30b
```

You do not need both models unless you switch between both configurations. To use another local
model, change `integrations.model_providers[].config.default_model` in the active platform config,
run `ollama pull <model>`, and restart the demo. Keep the provider names `ollama` and
`ollama-extract` because the seeded LLM steps reference those providers by name.

When upgrading from the 0.1.3 demo, add the sampling settings to your own `.config/platform.yaml`
as in `docker/platform.yaml`: `temperature: 0.1` on `ollama`, and a second provider
`ollama-extract` with the same model and `temperature: 0`.

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

The default configuration uses a Docker Python image for the runtime case. Its file-free command
runs from the container root and does not require a mounted workspace. The six cases do not require
GitHub, web search, vector databases, or external knowledge stores.

The repository intentionally contains only the six active showcase packages, their shared portal
and invocation helper, configuration, seeds, tests, and documentation. Retired example modules are
not part of this gallery.

## Optional: the Knowledge Base stack

Case 7 is the only one that needs anything beyond the base setup, which is why it ships separately:
its workflow is not part of `seed.sql`, so the portal shows it as unavailable until you opt in.

It runs the **knowledge** pipeline, not the data pipeline. A model reads the handbook and proposes
atomic claims; ten structural refinement steps canonicalize, validate, split, merge, deduplicate,
filter and score them; then they are persisted and indexed. Asking a question queries those claims —
*what is asserted* — rather than the page they came from.

That choice is what keeps the footprint small. Claims are found by identity in PostgreSQL and by
keyword in OpenSearch, so this case needs **no vector store and no embedding provider** — and
therefore no multi-gigabyte torch download in the default install.

```powershell
# 1. Optional Python extra (OpenSearch client only)
uv sync --extra knowledge

# 2. Knowledge database and OpenSearch
docker compose -f docker/docker-compose.knowledge.yml up -d

# 3. The knowledge database has its own migration chain and takes its URL
#    explicitly — it is never named platform-wide.
uv run nlght-ai migrate-knowledge -x url=postgresql+asyncpg://knowledge:knowledge@localhost:5434/knowledge

# 4. Seed the case (from the parent directory of this repository)
psql $env:DATABASE_URL -f demo/seed/07_knowledge_base.sql
```

`platform.yaml` needs only the `ollama-extract` provider described above: both stores are resource
rows carrying their own `pg_url`, because the knowledge database is reached only through the
activations that use it. If your OpenSearch or
knowledge database runs elsewhere, edit the two `INSERT INTO resources` rows at the end of
`seed/07_knowledge_base.sql`.

Then restart the demo and run the case twice — once to ingest, once to ask. The first run reports
what it established, in these fields:

```json
{
  "documents_acquired": 0,
  "units_parsed": 0,
  "claims_extracted": 0,
  "claims_quarantined": 0,
  "claims_persisted": 0,
  "claims_retired": 0
}
```

`documents_acquired` is 3 for the shipped corpus and `claims_quarantined` is 0 under this case's
review policy. The other counts depend on your extraction model and are not fixed — that is the
point of reporting them rather than asserting them.

> **`claims_quarantined: 0` is a demo choice, not a production default.** The shipped default for
> `knowledge.review_flag` is `policy: all` — every claim is quarantined and stays invisible to every
> retrieval path until a person approves it. This case sets `policy: thresholds` with
> `min_confidence: 0.0` so that ingest-then-ask is two steps instead of three. Loosening that policy
> in a real deployment should follow from seeing what your own source actually produces.

Re-running the ingest step costs nothing: `knowledge.extract` records what it read, and a document
whose content and extraction version are unchanged is skipped rather than sent to the model again.
After upgrading from nlght-ai 0.1.3 the extraction version changes once (it no longer hashes the
temperature), so the first ingest re-extracts the corpus; later sampling changes leave it alone.
To hold one run against the one before it:

```powershell
uv run nlght-ai knowledge-report -x url=postgresql+asyncpg://knowledge:knowledge@localhost:5434/knowledge
```

## Experimental: Among Us agents

`among_us/` is a separate experiment, not one of the seven showcase cases. Five nlght-ai agents
with distinct egos — Sherlock, Archive, Bubbles, Maverick and Echo — play Among Us against a
deterministic simulation. Each agent runs in its own Hive Mind session, perceives only what its
own character could see, remembers it according to its profile (attention, encoding, forgetting),
derives suspicions from what it can still recall, and lets the model choose among legal actions
only. A validator refuses illegal actions and claims that cite memories the agent does not have.

```powershell
psql $env:DATABASE_URL -f demo/seed/08_among_us.sql
Invoke-RestMethod -Method Post http://localhost:9000/among-us/api/matches `
  -ContentType application/json -Body '{"seed":7}'
```

Add `"offline": true` to play without a model or database. The C# bridge for the real game
(`among_us/bridge`) is an unbuilt skeleton. Architecture, metrics and the bridge protocol are in
`docs/documentation/among-us.md`.

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

## Versioned releases

Demo releases are created from version tags such as `v0.2.0`. Pushing the tag runs the GitHub
release workflow, derives package version `0.2.0`, builds a wheel and source distribution, installs
the wheel in a clean environment, verifies the packaged portal assets and CLI entry point, and then
attaches both artifacts to a GitHub Release with generated release notes.

Create a release intentionally from a clean commit:

```powershell
git tag -a v0.2.0 -m "nlght-ai Demo v0.2.0"
git push origin v0.2.0
```

Install a released demo wheel with the GitHub CLI:

```powershell
gh release download v0.2.0 --pattern "*.whl" --dir .release-download
python -m pip install (Get-ChildItem .release-download\*.whl)
nlght-ai-demo
```

The leading `v` is used by Git tags and GitHub Releases; Python package metadata omits it. Tags must
therefore contain a PEP 440-compatible version after the prefix. Releases are published to GitHub,
not PyPI. The installed demo still requires the configured PostgreSQL database, nlght-ai license,
Docker runtime, and running Ollama service described above.
