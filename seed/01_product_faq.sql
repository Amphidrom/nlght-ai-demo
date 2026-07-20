-- Product FAQ: retrieve grounded manual passages and answer with source IDs.
INSERT INTO workflows (workflow_id, name, description) VALUES
('a1100000-0000-0000-0000-000000000001', 'product_faq', 'Grounded product FAQ with explicit source IDs')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('a1100000-0000-0000-0000-000000000002', 'a1100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES
('a1100000-0000-0000-0000-000000000003', 'a1100000-0000-0000-0000-000000000002', 1, 'retrieve_manual', 'product_faq_retrieve', '{}', '{"found":"a1100000-0000-0000-0000-000000000004","not_found":"a1100000-0000-0000-0000-000000000005"}', true, false),
('a1100000-0000-0000-0000-000000000004', 'a1100000-0000-0000-0000-000000000002', 2, 'answer_grounded', 'product_faq_answer', '{}', '{}', false, true),
('a1100000-0000-0000-0000-000000000005', 'a1100000-0000-0000-0000-000000000002', 3, 'answer_unknown', 'product_faq_answer', '{}', '{}', false, true)
ON CONFLICT DO NOTHING;
