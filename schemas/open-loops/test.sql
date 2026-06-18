-- Tests for schemas/open-loops/schema.sql
-- Run AFTER applying schema.sql, against a local Supabase DB:
--   psql "$DB_URL" -v ON_ERROR_STOP=1 -f schemas/open-loops/schema.sql
--   psql "$DB_URL" -v ON_ERROR_STOP=1 -f schemas/open-loops/test.sql
-- All assertions run inside a transaction that is ROLLED BACK, so the DB is
-- left untouched. Any failed ASSERT raises and (with ON_ERROR_STOP=1) exits 1.

\set ON_ERROR_STOP on
SET client_min_messages TO NOTICE;

BEGIN;

-- Build a unit vector along axis k (1536 dims) for deterministic similarity.
CREATE FUNCTION pg_temp.axis_vec(k INT) RETURNS vector AS $$
  SELECT ('[' || array_to_string(
    array(SELECT CASE WHEN i = k THEN 1 ELSE 0 END FROM generate_series(1, 1536) i),
    ',') || ']')::vector
$$ LANGUAGE sql IMMUTABLE;

-- ── Seed ──────────────────────────────────────────────────────────────────
-- Status-classification rows (NULL/legacy status + content markers).
INSERT INTO thoughts (id, content, metadata, status) VALUES
  ('00000000-0000-0000-0000-000000000001',
   'David to follow up on the Greptile fitness check next week.',
   '{"type":"task"}'::jsonb, NULL),                                  -- → open, owner David
  ('00000000-0000-0000-0000-000000000002',
   'Eric Alli to share the chat-table snapshot queries. THEIR open loop.',
   '{"type":"task"}'::jsonb, NULL),                                  -- → waiting
  ('00000000-0000-0000-0000-000000000003',
   'CLOSED loop — ListenLabs fitness doc posted to Slack. Done.',
   '{"type":"task"}'::jsonb, NULL),                                  -- → closed
  ('00000000-0000-0000-0000-000000000004',
   'A leftover kanban task from workflow-status.',
   '{"type":"task"}'::jsonb, 'new'),                                 -- → open (normalized)
  ('00000000-0000-0000-0000-000000000005',
   'Just a passing observation, not a task.',
   '{"type":"observation"}'::jsonb, NULL);                           -- stays NULL

-- Hybrid-search rows (with embeddings).
INSERT INTO thoughts (id, content, metadata, status, embedding) VALUES
  ('00000000-0000-0000-0000-0000000000a1',
   'A narrative about the weather and unrelated daily musings.',
   '{"type":"task"}'::jsonb, 'open', pg_temp.axis_vec(1)),           -- semantic match, no keyword
  ('00000000-0000-0000-0000-0000000000a2',
   'Formulate a campaign for partitioning outreach to large customers.',
   '{"type":"task"}'::jsonb, 'open', pg_temp.axis_vec(2));           -- keyword-only (orthogonal vec)

-- ── Backfill ────────────────────────────────────────────────────────────────
SELECT backfill_open_loop_status();

DO $$
DECLARE
  v_status TEXT;
  v_owner  TEXT;
BEGIN
  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000001' INTO v_status;
  ASSERT v_status = 'open', format('row1 should be open, got %s', v_status);
  SELECT metadata->>'owner' FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000001' INTO v_owner;
  ASSERT v_owner = 'David', format('row1 owner should be David, got %s', v_owner);

  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000002' INTO v_status;
  ASSERT v_status = 'waiting', format('row2 should be waiting, got %s', v_status);

  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000003' INTO v_status;
  ASSERT v_status = 'closed', format('row3 should be closed, got %s', v_status);

  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000004' INTO v_status;
  ASSERT v_status = 'open', format('row4 (new) should normalize to open, got %s', v_status);

  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000005' INTO v_status;
  ASSERT v_status IS NULL, format('row5 (observation) should stay NULL, got %s', v_status);

  RAISE NOTICE 'PASS: backfill classification (open/waiting/closed/normalize/skip-observation + owner)';
END $$;

-- ── list_open_loops ──────────────────────────────────────────────────────────
DO $$
DECLARE
  v_rows BIGINT;
  v_total BIGINT;
  v_page1 UUID[];
  v_page2 UUID[];
BEGIN
  -- Default open+waiting: 4 open (rows 1, 4, a1, a2) + 1 waiting (row 2) = 5.
  SELECT count(*), max(total_count) FROM list_open_loops(ARRAY['open','waiting'], NULL, 50, 0)
    INTO v_rows, v_total;
  ASSERT v_rows = 5, format('open+waiting rows should be 5, got %s', v_rows);
  ASSERT v_total = 5, format('open+waiting total_count should be 5, got %s', v_total);

  -- Owner filter: only row1 is tagged David.
  SELECT count(*) FROM list_open_loops(ARRAY['open','waiting'], 'David', 50, 0) INTO v_rows;
  ASSERT v_rows = 1, format('owner=David rows should be 1, got %s', v_rows);

  -- Closed only: row3.
  SELECT count(*) FROM list_open_loops(ARRAY['closed'], NULL, 50, 0) INTO v_rows;
  ASSERT v_rows = 1, format('closed rows should be 1, got %s', v_rows);

  -- Pagination over the 4 open rows (limit 2): pages disjoint, total_count stable.
  SELECT array_agg(id ORDER BY id), max(total_count)
    FROM list_open_loops(ARRAY['open'], NULL, 2, 0) INTO v_page1, v_total;
  ASSERT array_length(v_page1, 1) = 2, format('open page1 should have 2 rows, got %s', array_length(v_page1,1));
  ASSERT v_total = 4, format('open total_count should be 4, got %s', v_total);

  SELECT array_agg(id ORDER BY id) FROM list_open_loops(ARRAY['open'], NULL, 2, 2) INTO v_page2;
  ASSERT array_length(v_page2, 1) = 2, format('open page2 should have 2 rows, got %s', array_length(v_page2,1));
  ASSERT NOT (v_page1 && v_page2), 'open page1 and page2 must not overlap';

  RAISE NOTICE 'PASS: list_open_loops (filter, owner, closed, pagination, total_count)';
END $$;

-- ── match_thoughts_hybrid ────────────────────────────────────────────────────
DO $$
DECLARE
  v_ids UUID[];
  v_total BIGINT;
  v_pure_count INT;
BEGIN
  -- Query: embedding on axis 1 + text 'partitioning'.
  -- a1 (axis 1) is a semantic hit; a2 (axis 2, orthogonal → sim 0) is a
  -- pure-semantic MISS but a keyword hit. Hybrid must surface BOTH.
  SELECT array_agg(id), max(total_count)
    FROM match_thoughts_hybrid(pg_temp.axis_vec(1), 'partitioning', 10, 0, NULL, 0.6)
    INTO v_ids, v_total;

  ASSERT '00000000-0000-0000-0000-0000000000a1' = ANY(v_ids),
    'hybrid must include the semantic hit a1';
  ASSERT '00000000-0000-0000-0000-0000000000a2' = ANY(v_ids),
    'hybrid must rescue the keyword-only/semantic-miss row a2';
  ASSERT v_total >= 2, format('hybrid total_count should be >= 2, got %s', v_total);

  -- Confirm a2 IS a pure-semantic miss at the standard 0.5 threshold, i.e. the
  -- hybrid result is doing real work that plain match_thoughts would not.
  SELECT count(*) FROM match_thoughts(pg_temp.axis_vec(1), 0.5, 10, '{}'::jsonb)
    WHERE id = '00000000-0000-0000-0000-0000000000a2' INTO v_pure_count;
  ASSERT v_pure_count = 0, 'a2 should NOT appear in pure semantic match_thoughts (proves hybrid value)';

  -- Status filter excludes everything when restricted to waiting.
  SELECT count(*) FROM match_thoughts_hybrid(pg_temp.axis_vec(1), 'partitioning', 10, 0, ARRAY['waiting'], 0.6)
    WHERE id IN ('00000000-0000-0000-0000-0000000000a1','00000000-0000-0000-0000-0000000000a2')
    INTO v_pure_count;
  ASSERT v_pure_count = 0, 'status=waiting filter should exclude the open hybrid rows';

  RAISE NOTICE 'PASS: match_thoughts_hybrid (semantic + keyword rescue, total_count, status filter)';
END $$;

-- ── set_thought_status ───────────────────────────────────────────────────────
DO $$
DECLARE
  v_res JSONB;
  v_status TEXT;
  v_raised BOOLEAN := false;
BEGIN
  v_res := set_thought_status('00000000-0000-0000-0000-000000000001', 'closed');
  ASSERT (v_res->>'updated')::boolean, 'set_thought_status should report updated=true';
  SELECT status FROM thoughts WHERE id = '00000000-0000-0000-0000-000000000001' INTO v_status;
  ASSERT v_status = 'closed', format('row1 should now be closed, got %s', v_status);

  -- Unknown id → updated=false, no error.
  v_res := set_thought_status('00000000-0000-0000-0000-0000deadbeef', 'open');
  ASSERT NOT (v_res->>'updated')::boolean, 'unknown id should report updated=false';

  -- Invalid status → raises.
  BEGIN
    PERFORM set_thought_status('00000000-0000-0000-0000-000000000001', 'bogus');
  EXCEPTION WHEN others THEN
    v_raised := true;
  END;
  ASSERT v_raised, 'invalid status should raise';

  RAISE NOTICE 'PASS: set_thought_status (transition, unknown id, invalid status)';
END $$;

ROLLBACK;

\echo '──────────────────────────────────────────────'
\echo 'ALL OPEN-LOOPS SCHEMA TESTS PASSED'
