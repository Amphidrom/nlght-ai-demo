-- Shipping Quote: validate input, execute one registered custom tool, return JSON.
INSERT INTO workflows (workflow_id, name, description) VALUES
('b2100000-0000-0000-0000-000000000001', 'shipping_quote', 'Validated parcel quote backed by a deterministic custom tool')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('b2100000-0000-0000-0000-000000000002', 'b2100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES
('b2100000-0000-0000-0000-000000000003', 'b2100000-0000-0000-0000-000000000002', 1, 'validate_shipment', 'shipping_validate', '{}', '{"valid":"b2100000-0000-0000-0000-000000000004","invalid":"b2100000-0000-0000-0000-000000000005"}', true, false),
('b2100000-0000-0000-0000-000000000004', 'b2100000-0000-0000-0000-000000000002', 2, 'calculate_quote', 'shipping_quote', '{"model_provider":"ollama"}', '{}', false, true),
('b2100000-0000-0000-0000-000000000005', 'b2100000-0000-0000-0000-000000000002', 3, 'invalid_shipment', 'shipping_quote', '{"invalid":true}', '{}', false, true)
ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position, name = EXCLUDED.name, type = EXCLUDED.type,
    config = EXCLUDED.config, transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start, is_terminal = EXCLUDED.is_terminal;
INSERT INTO resources (resource_id, name, kind, provider, config) VALUES
('b2100000-0000-0000-0000-000000000010', 'shipping-rates', 'shipping_rate', 'local', '{}')
ON CONFLICT DO NOTHING;
