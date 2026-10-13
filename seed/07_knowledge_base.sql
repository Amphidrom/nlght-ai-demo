-- Knowledge Base: one workflow, two separate entry points.
--
--   {"action":"ingest"}  a model reads the handbook and the pipeline turns it
--                        into reviewed, retrievable claims
--   {"action":"ask", "message":"..."}  a question is answered from those claims
--
-- Requires the optional knowledge stack (knowledge database + OpenSearch); see
-- docker/docker-compose.knowledge.yml and the README. Everything between the
-- route step and the report step is the platform's own pipeline.
INSERT INTO workflows (workflow_id, name, description, max_hops) VALUES
('b7100000-0000-0000-0000-000000000001', 'knowledge_base', 'Knowledge ingestion over a local handbook, and retrieval of the reviewed claims', 400)
ON CONFLICT (workflow_id) DO UPDATE SET
    description = EXCLUDED.description, max_hops = EXCLUDED.max_hops;
INSERT INTO workflow_versions (workflow_version_id, workflow_id, version, status) VALUES
('b7100000-0000-0000-0000-000000000002', 'b7100000-0000-0000-0000-000000000001', 1, 'active')
ON CONFLICT DO NOTHING;

INSERT INTO workflow_steps (workflow_step_id, workflow_version_id, position, name, type, config, transitions, is_start, is_terminal) VALUES

-- Entry: which of the two branches this request wants.
('b7100000-0000-0000-0000-000000000010', 'b7100000-0000-0000-0000-000000000002',  1, 'route', 'knowledge_base_route',
 '{}',
 '{"ingest":"b7100000-0000-0000-0000-000000000020","ask":"b7100000-0000-0000-0000-000000000060"}', true, false),

-- Ingestion branch -----------------------------------------------------------
-- A partial view of the source must never reach a step that retires claims, so
-- "incomplete" goes straight to the report instead of into the pipeline.
('b7100000-0000-0000-0000-000000000020', 'b7100000-0000-0000-0000-000000000002',  2, 'acquire', 'ingestion.source',
 '{"type":"filesystem","source_id":"northstar-handbook","roots":[{"path":"./knowledge_base/corpus","alias":"handbook"}],"include":["**/*.md"],"respect_ignore_files":false}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000021","incomplete":"b7100000-0000-0000-0000-000000000050","empty":"b7100000-0000-0000-0000-000000000050"}', false, false),

-- "more" routed back to itself is what finishes the whole snapshot.
('b7100000-0000-0000-0000-000000000021', 'b7100000-0000-0000-0000-000000000002',  3, 'process', 'ingestion.process',
 '{"chunk_size":4000,"chunk_overlap":200,"document_batch_size":1}',
 '{"more":"b7100000-0000-0000-0000-000000000021","DEFAULT":"b7100000-0000-0000-0000-000000000022","empty":"b7100000-0000-0000-0000-000000000050"}', false, false),

('b7100000-0000-0000-0000-000000000022', 'b7100000-0000-0000-0000-000000000002',  4, 'parse', 'knowledge.parse_auto',
 '{"tags":{"product":"northstar-desk","version":"1"}}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000023"}', false, false),

-- The workflow''s own model reads each unit. "unchanged" means every document
-- was already extracted at this exact extraction version — a re-run costs nothing.
('b7100000-0000-0000-0000-000000000023', 'b7100000-0000-0000-0000-000000000002',  5, 'extract', 'knowledge.extract',
 '{"model_provider":"ollama","unit_batch_size":1,"temperature":0,"writer":"northstar-knowledge-writer"}',
 '{"more":"b7100000-0000-0000-0000-000000000023","DEFAULT":"b7100000-0000-0000-0000-000000000030","unchanged":"b7100000-0000-0000-0000-000000000050","empty":"b7100000-0000-0000-0000-000000000050"}', false, false),

-- Ten structural refinements, in the order the step reference documents.
-- They inspect the shape of a candidate, never its meaning.
('b7100000-0000-0000-0000-000000000030', 'b7100000-0000-0000-0000-000000000002',  6, 'canonicalize', 'knowledge.canonicalize',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000031"}', false, false),
('b7100000-0000-0000-0000-000000000031', 'b7100000-0000-0000-0000-000000000002',  7, 'validate', 'knowledge.validate',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000032"}', false, false),
('b7100000-0000-0000-0000-000000000032', 'b7100000-0000-0000-0000-000000000002',  8, 'atomicity', 'knowledge.atomicity',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000033"}', false, false),
('b7100000-0000-0000-0000-000000000033', 'b7100000-0000-0000-0000-000000000002',  9, 'identity_merge', 'knowledge.identity_merge',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000034"}', false, false),
('b7100000-0000-0000-0000-000000000034', 'b7100000-0000-0000-0000-000000000002', 10, 'dedup', 'knowledge.dedup',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000035"}', false, false),
('b7100000-0000-0000-0000-000000000035', 'b7100000-0000-0000-0000-000000000002', 11, 'noise_filter', 'knowledge.noise_filter',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000036"}', false, false),
('b7100000-0000-0000-0000-000000000036', 'b7100000-0000-0000-0000-000000000002', 12, 'domain_classify', 'knowledge.domain_classify',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000037"}', false, false),
('b7100000-0000-0000-0000-000000000037', 'b7100000-0000-0000-0000-000000000002', 13, 'quality_score', 'knowledge.quality_score',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000038"}', false, false),
('b7100000-0000-0000-0000-000000000038', 'b7100000-0000-0000-0000-000000000002', 14, 'graph_quality', 'knowledge.graph_quality',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000039"}', false, false),
('b7100000-0000-0000-0000-000000000039', 'b7100000-0000-0000-0000-000000000002', 15, 'meta_enrichment', 'knowledge.meta_enrichment',
 '{}', '{"DEFAULT":"b7100000-0000-0000-0000-000000000040"}', false, false),

-- DEMO CHOICE, NOT A PRODUCTION DEFAULT. The shipped default is policy "all":
-- every claim is quarantined until a person approves it. This case releases
-- everything so that "ingest" and then "ask" is two steps rather than three.
('b7100000-0000-0000-0000-000000000040', 'b7100000-0000-0000-0000-000000000002', 16, 'review_flag', 'knowledge.review_flag',
 '{"policy":"thresholds","min_confidence":0.0}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000041"}', false, false),

('b7100000-0000-0000-0000-000000000041', 'b7100000-0000-0000-0000-000000000002', 17, 'persist', 'knowledge.persist',
 '{"writer":"northstar-knowledge-writer"}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000050"}', false, false),

('b7100000-0000-0000-0000-000000000050', 'b7100000-0000-0000-0000-000000000002', 18, 'report', 'knowledge_base_report',
 '{}', '{}', false, true),

-- Retrieval branch -----------------------------------------------------------
-- Only the knowledge capability: reviewed claims live in PostgreSQL and
-- OpenSearch, so this branch needs no vector store and no embedding provider.
('b7100000-0000-0000-0000-000000000060', 'b7100000-0000-0000-0000-000000000002', 19, 'search', 'retrieval.search',
 '{"knowledge_store":"northstar-knowledge","sources":["knowledge"],"required_sources":["knowledge"],"final_k":8,"kinds":["fact","rule","pattern","decision"]}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000061","empty":"b7100000-0000-0000-0000-000000000063"}', false, false),

-- Puts the findings into an untrusted context message and its own instructions
-- into a trusted one. A claim never reaches the system channel.
('b7100000-0000-0000-0000-000000000061', 'b7100000-0000-0000-0000-000000000002', 20, 'compose', 'prompt.compose',
 '{"max_passage_chars":6000,"include_memory":false}',
 '{"DEFAULT":"b7100000-0000-0000-0000-000000000062"}', false, false),

('b7100000-0000-0000-0000-000000000062', 'b7100000-0000-0000-0000-000000000002', 21, 'answer', 'knowledge_base_answer',
 '{"model_provider":"ollama"}', '{}', false, true),

('b7100000-0000-0000-0000-000000000063', 'b7100000-0000-0000-0000-000000000002', 22, 'nothing_found', 'knowledge_base_nothing_found',
 '{}', '{}', false, true)

ON CONFLICT (workflow_step_id) DO UPDATE SET
    position = EXCLUDED.position, name = EXCLUDED.name, type = EXCLUDED.type,
    config = EXCLUDED.config, transitions = EXCLUDED.transitions,
    is_start = EXCLUDED.is_start, is_terminal = EXCLUDED.is_terminal;

-- The two knowledge activations. Both carry their own knowledge-database URL:
-- the knowledge database is never named platform-wide, only by the resources
-- that use it. Adjust the URLs to match docker-compose.knowledge.yml.
INSERT INTO resources (resource_id, name, kind, provider, config) VALUES
('b7100000-0000-0000-0000-000000000070', 'northstar-knowledge', 'knowledge_store', 'postgres+opensearch',
 '{"pg_url":"postgresql+asyncpg://knowledge:knowledge@localhost:5434/knowledge","os_url":"http://localhost:9200","index_prefix":"northstar"}'),
('b7100000-0000-0000-0000-000000000071', 'northstar-knowledge-writer', 'knowledge_index_writer', 'postgres+opensearch',
 '{"pg_url":"postgresql+asyncpg://knowledge:knowledge@localhost:5434/knowledge","os_url":"http://localhost:9200","index_prefix":"northstar"}')
ON CONFLICT (resource_id) DO UPDATE SET
    name = EXCLUDED.name, kind = EXCLUDED.kind,
    provider = EXCLUDED.provider, config = EXCLUDED.config;
