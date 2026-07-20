-- Preference Assistant: persist one explicit user preference in Hive Mind.
INSERT INTO workflows (workflow_id, name, description) VALUES
('d5100000-0000-0000-0000-000000000001', 'preference_assistant', 'Persist a user-supplied preference without storing model output as fact')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('d5100000-0000-0000-0000-000000000002', 'd5100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES
('d5100000-0000-0000-0000-000000000003', 'd5100000-0000-0000-0000-000000000002', 1, 'resolve_preference_action', 'preference_remember', '{}', '{"remembered":"d5100000-0000-0000-0000-000000000004","apply":"d5100000-0000-0000-0000-000000000004","missing":"d5100000-0000-0000-0000-000000000004"}', true, false),
('d5100000-0000-0000-0000-000000000004', 'd5100000-0000-0000-0000-000000000002', 2, 'confirm_preference', 'preference_respond', '{"model_provider":"ollama"}', '{}', false, true)
ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position,
    name = EXCLUDED.name,
    type = EXCLUDED.type,
    config = EXCLUDED.config,
    transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start,
    is_terminal = EXCLUDED.is_terminal;
