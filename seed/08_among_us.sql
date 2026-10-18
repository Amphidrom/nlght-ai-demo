-- Among Us (experimental): one agent turn at one decision point.
--
-- The match orchestrator (among_us/adapter.py) starts this workflow once per
-- agent and decision point, in the session "among-us:{match}:{agent}", so every
-- agent keeps its own memories and beliefs in its own Hive Mind session.
--
-- Not loaded by seed.sql. Load it explicitly:
--     psql $DATABASE_URL -f demo/seed/08_among_us.sql
INSERT INTO workflows (workflow_id, name, description) VALUES
('a8100000-0000-0000-0000-000000000001', 'among_us_agent_turn', 'Perceive, remember, believe, decide and validate one Among Us action')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('a8100000-0000-0000-0000-000000000002', 'a8100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;
INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES
('a8100000-0000-0000-0000-000000000010', 'a8100000-0000-0000-0000-000000000002', 1, 'perceive', 'among_us_perceive',
 '{}', '{"perceived":"a8100000-0000-0000-0000-000000000011"}', true, false),
('a8100000-0000-0000-0000-000000000011', 'a8100000-0000-0000-0000-000000000002', 2, 'remember', 'among_us_remember',
 '{}', '{"remembered":"a8100000-0000-0000-0000-000000000012"}', false, false),
('a8100000-0000-0000-0000-000000000012', 'a8100000-0000-0000-0000-000000000002', 3, 'believe', 'among_us_believe',
 '{}', '{"believed":"a8100000-0000-0000-0000-000000000013"}', false, false),
-- The only step that calls a model, and only when more than one action is legal.
('a8100000-0000-0000-0000-000000000013', 'a8100000-0000-0000-0000-000000000002', 4, 'decide', 'among_us_decide',
 '{"model_provider":"ollama","debug":false}', '{"decided":"a8100000-0000-0000-0000-000000000014"}', false, false),
('a8100000-0000-0000-0000-000000000014', 'a8100000-0000-0000-0000-000000000002', 5, 'validate', 'among_us_validate',
 '{}', '{}', false, true)
ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position, name = EXCLUDED.name, type = EXCLUDED.type,
    config = EXCLUDED.config, transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start, is_terminal = EXCLUDED.is_terminal;
