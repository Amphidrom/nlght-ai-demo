-- Small Incident Triage showcase.
-- Normalizes one alert, applies deterministic severity thresholds, and routes
-- to either an urgent response or routine review.

INSERT INTO workflows (workflow_id, name, description)
VALUES (
    'f8100000-0000-0000-0000-000000000001',
    'incident_triage',
    'Demo: normalize an operational alert, classify severity, and recommend next actions'
)
ON CONFLICT DO NOTHING;

INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status)
VALUES (
    'f8100000-0000-0000-0000-000000000002',
    'f8100000-0000-0000-0000-000000000001',
    1,
    'active'
)
ON CONFLICT DO NOTHING;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES (
    'f8100000-0000-0000-0000-000000000003',
    'f8100000-0000-0000-0000-000000000002',
    1, 'normalize_signal', 'incident_normalize', '{}',
    '{
      "normalized": "f8100000-0000-0000-0000-000000000004",
      "invalid": "f8100000-0000-0000-0000-000000000006"
    }',
    true, false
)
ON CONFLICT DO NOTHING;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES (
    'f8100000-0000-0000-0000-000000000004',
    'f8100000-0000-0000-0000-000000000002',
    2, 'classify_severity', 'incident_classify', '{}',
    '{
      "urgent": "f8100000-0000-0000-0000-000000000005",
      "routine": "f8100000-0000-0000-0000-000000000006"
    }',
    false, false
)
ON CONFLICT DO NOTHING;

INSERT INTO workflow_steps (
    workflow_step_id, workflow_version_id, position, name, type,
    config, transitions, is_start, is_terminal
) VALUES
(
    'f8100000-0000-0000-0000-000000000005',
    'f8100000-0000-0000-0000-000000000002',
    3, 'urgent_response', 'incident_respond', '{"route": "urgent_response"}', '{}',
    false, true
),
(
    'f8100000-0000-0000-0000-000000000006',
    'f8100000-0000-0000-0000-000000000002',
    4, 'manual_review', 'incident_respond', '{"route": "manual_review"}', '{}',
    false, true
)
ON CONFLICT DO NOTHING;
