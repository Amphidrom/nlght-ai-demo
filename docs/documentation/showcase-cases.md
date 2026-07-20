# Showcase cases

The gallery contains six closed examples. Open a case and select one of its scenarios to replace
the editable JSON input. Each scenario is designed to expose a different workflow outcome, not
just provide different wording.

The portal follows the nlght.ai dark teal visual system and intentionally uses no animations,
motion effects, or animated backgrounds. Interaction state is communicated through static color,
border, text, and focus changes.

## Product FAQ

Try the supported recovery question, the combined offline-and-roles question, and the unsupported
satellite question. Supported answers include explicit local manual IDs. The unsupported question
demonstrates that retrieval can return an honest unknown instead of fabricating an answer.

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
- Runtime Data Check contrasts CSV data with and without missing values.
