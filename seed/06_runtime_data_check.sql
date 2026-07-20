-- Runtime Data Check: validate CSV input and inspect it with one fixed command.
INSERT INTO workflows (workflow_id, name, description) VALUES
('e6100000-0000-0000-0000-000000000001', 'runtime_data_check', 'Check CSV shape and missing values in the configured isolated runtime')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('e6100000-0000-0000-0000-000000000002', 'e6100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES
('e6100000-0000-0000-0000-000000000003', 'e6100000-0000-0000-0000-000000000002', 1, 'validate_csv', 'runtime_csv_validate', '{}', '{"valid":"e6100000-0000-0000-0000-000000000004","invalid":"e6100000-0000-0000-0000-000000000005"}', true, false),
('e6100000-0000-0000-0000-000000000004', 'e6100000-0000-0000-0000-000000000002', 2, 'check_csv', 'runtime_csv_check', '{"model_provider":"ollama"}', '{}', false, true),
('e6100000-0000-0000-0000-000000000005', 'e6100000-0000-0000-0000-000000000002', 3, 'invalid_csv', 'runtime_csv_check', '{"invalid":true}', '{}', false, true)
ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position, name = EXCLUDED.name, type = EXCLUDED.type,
    config = EXCLUDED.config, transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start, is_terminal = EXCLUDED.is_terminal;
