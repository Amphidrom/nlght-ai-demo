-- Policy-Aware Document Intake showcase.
-- Extracts local structured fields, validates required data, then routes the
-- intake to accepted or needs_review without external services.

INSERT INTO workflows (workflow_id, name, description)
VALUES (
    'd6100000-0000-0000-0000-000000000001',
    'document_intake',
    'Demo: structured document intake with validation and workflow routing'
)
ON CONFLICT (workflow_id) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description;

INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status)
VALUES (
    'd6100000-0000-0000-0000-000000000002',
    'd6100000-0000-0000-0000-000000000001',
    1,
    'active'
)
ON CONFLICT DO NOTHING;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES (
    'd6100000-0000-0000-0000-000000000003',
    'd6100000-0000-0000-0000-000000000002',
    1, 'extract_intake', 'intake_extract', '{}',
    '{"extracted": "d6100000-0000-0000-0000-000000000004"}',
    true, false
) ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position,
    name = EXCLUDED.name,
    type = EXCLUDED.type,
    config = EXCLUDED.config,
    transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start,
    is_terminal = EXCLUDED.is_terminal;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES (
    'd6100000-0000-0000-0000-000000000004',
    'd6100000-0000-0000-0000-000000000002',
    2, 'validate_intake', 'intake_validate', '{}',
    '{
      "accepted": "d6100000-0000-0000-0000-000000000005",
      "needs_review": "d6100000-0000-0000-0000-000000000006",
      "escalated_review": "d6100000-0000-0000-0000-000000000007"
    }',
    false, false
) ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position,
    name = EXCLUDED.name,
    type = EXCLUDED.type,
    config = EXCLUDED.config,
    transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start,
    is_terminal = EXCLUDED.is_terminal;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES
(
    'd6100000-0000-0000-0000-000000000005',
    'd6100000-0000-0000-0000-000000000002',
    3, 'accepted', 'intake_respond', '{"status":"accepted","model_provider":"ollama"}', '{}',
    false, true
),
(
    'd6100000-0000-0000-0000-000000000006',
    'd6100000-0000-0000-0000-000000000002',
    4, 'needs_review', 'intake_respond', '{"status":"needs_review","model_provider":"ollama"}', '{}',
    false, true
),
(
    'd6100000-0000-0000-0000-000000000007',
    'd6100000-0000-0000-0000-000000000002',
    5, 'escalated_review', 'intake_respond', '{"status":"escalated_review","model_provider":"ollama"}', '{}',
    false, true
)
ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position,
    name = EXCLUDED.name,
    type = EXCLUDED.type,
    config = EXCLUDED.config,
    transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start,
    is_terminal = EXCLUDED.is_terminal;
