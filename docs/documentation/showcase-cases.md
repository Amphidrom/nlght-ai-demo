# Showcase cases

The gallery contains six closed examples. Open a case and select one of its scenarios to replace
the editable JSON input. Each scenario is designed to expose a different workflow outcome, not
just provide different wording.

Each successful scenario performs a real call to the configured model provider. The model receives
the request-specific input together with trusted case evidence: retrieved manuals, a tool result,
severity classification, policy evaluations, persisted preferences, or an isolated-runtime report.
Deterministic components own factual calculations and routing; the model owns language and
contextual explanation.

Each LLM-backed workflow step selects its provider explicitly through `config.model_provider` in
the seeded workflow definition. The portal supplies the configured default model for the request.

The portal follows the nlght.ai dark teal visual system and intentionally uses no animations,
motion effects, or animated backgrounds. Interaction state is communicated through static color,
border, text, and focus changes.

## Product FAQ

Try the supported recovery question, the combined offline-and-roles question, and the unsupported
satellite question. Supported answers include explicit local manual IDs. The unsupported question
demonstrates that retrieval can return an honest unknown instead of fabricating an answer. Its
structured result has status `human_review_required`, an empty `sources` list, and instructions to
open **Help → Contact Support** in Northstar Desk and send the question to a support specialist.

## Policy-Aware Document Intake

The result includes `policy_evaluations`, `missing_fields`, and `routing_status`:

- `CONTRACT-01` accepts a renewal only when organization and contract ID are present.
- `SEC-01` escalates an attributed, high-urgency threat with an identified affected asset.
- `INTAKE-01` sends a generic intake without an attributable organization to review.

These are deliberately small and visible showcase rules. They are not a production compliance
policy set.

## Preference Assistant

Run **Step 1 · Remember** to store the user-supplied `response_style`. Then select
**Step 2 · Apply** and run it with the same `session_key`. The later result reports the applied
preference and changes the response style. **Different session** demonstrates isolation by using a
new key that cannot access the stored preference.

## Other cases

- Shipping Quote contrasts validated tool input with a rejected parcel.
- Incident Triage contrasts urgent and routine workflow branches.
- Runtime Data Check contrasts CSV data with and without missing values. Its fixed, file-free Python
  command executes from the container root and passes CSV data only through `CSV_INPUT`; it does not
  depend on an optional workspace mount.
