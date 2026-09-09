-- Demo seed data — lädt alle Demo-Workflows in die Datenbank.
--
-- Voraussetzung: Migrationen wurden ausgeführt (alembic upgrade head)
-- Ausführen:     psql $DATABASE_URL -f demo/seed.sql
--
-- Alle Inserts sind idempotent; geänderte Workflow-Schritte werden aktualisiert.
-- UUIDs sind fest — mehrfaches Ausführen ist sicher.

\i demo/seed/01_product_faq.sql
\i demo/seed/02_shipping_quote.sql
\i demo/seed/03_incident_triage.sql
\i demo/seed/04_document_intake.sql
\i demo/seed/05_preference_assistant.sql
\i demo/seed/06_runtime_data_check.sql

-- 07_knowledge_base.sql is deliberately NOT loaded here. It needs the optional
-- knowledge stack (knowledge database + OpenSearch), and an unseeded workflow is
-- what makes the portal show that case as unavailable instead of failing on a
-- click. Load it explicitly once the stack is up:
--     psql $DATABASE_URL -f demo/seed/07_knowledge_base.sql
