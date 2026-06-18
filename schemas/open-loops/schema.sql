-- Open Loops: status tracking, exhaustive listing, and hybrid recall
-- ---------------------------------------------------------------------------
-- Adds an open/waiting/closed lifecycle to thoughts plus two RPCs that fix
-- the "I can't enumerate my open loops" problem:
--   * list_open_loops()       — paginated, status/owner-filtered listing
--   * match_thoughts_hybrid()  — semantic + keyword search (less phrasing-brittle)
-- and a one-time heuristic backfill that classifies existing task thoughts.
--
-- Fully idempotent (ADD COLUMN / CREATE INDEX IF NOT EXISTS, CREATE OR REPLACE).
-- Safe to run after schemas/workflow-status (this standardizes its vocabulary
-- from 'new' to the open/waiting/closed set) and schemas/enhanced-thoughts
-- (shares the same column names + tsvector index).
--
-- GUARD RAILS: only ADDS columns/indexes/functions; never alters or drops a
-- core thoughts column; the backfill uses qualified UPDATEs only.

-- ============================================================
-- 1. STATUS COLUMNS  (compatible with workflow-status / enhanced-thoughts)
-- ============================================================

ALTER TABLE thoughts ADD COLUMN IF NOT EXISTS status TEXT DEFAULT NULL;
ALTER TABLE thoughts ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMPTZ DEFAULT now();

-- ============================================================
-- 2. INDEXES
-- ============================================================

-- Partial index for status filtering (matches workflow-status / enhanced-thoughts).
CREATE INDEX IF NOT EXISTS idx_thoughts_status
  ON thoughts (status) WHERE status IS NOT NULL;

-- Composite index for the default "list open loops, newest activity first" path.
CREATE INDEX IF NOT EXISTS idx_thoughts_status_activity
  ON thoughts (status, status_updated_at DESC) WHERE status IS NOT NULL;

-- Owner lookups go through metadata->>'owner'; the existing GIN index on
-- metadata already covers containment, and the expression index below makes
-- the equality filter in list_open_loops fast.
CREATE INDEX IF NOT EXISTS idx_thoughts_owner
  ON thoughts ((metadata->>'owner')) WHERE metadata->>'owner' IS NOT NULL;

-- Full-text index for the keyword half of hybrid search. Same name + definition
-- as schemas/enhanced-thoughts so applying both is a no-op, not a duplicate.
CREATE INDEX IF NOT EXISTS idx_thoughts_content_tsvector
  ON thoughts USING gin (to_tsvector('simple', coalesce(content, '')));

-- ============================================================
-- 3. list_open_loops()
--    The headline "list all open loops" query: status + owner filters,
--    real pagination (offset/limit, capped at 200), and a window total_count.
-- ============================================================

CREATE OR REPLACE FUNCTION list_open_loops(
  p_status TEXT[] DEFAULT ARRAY['open', 'waiting'],
  p_owner TEXT DEFAULT NULL,
  p_limit INT DEFAULT 50,
  p_offset INT DEFAULT 0
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  status TEXT,
  owner TEXT,
  status_updated_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ,
  total_count BIGINT
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  WITH filtered AS (
    SELECT
      t.id,
      t.content,
      t.metadata,
      t.status,
      t.metadata->>'owner' AS owner,
      t.status_updated_at,
      t.created_at
    FROM public.thoughts t
    WHERE t.status = ANY(p_status)
      AND (p_owner IS NULL OR t.metadata->>'owner' = p_owner)
  ),
  counted AS (
    SELECT count(*) AS cnt FROM filtered
  )
  SELECT
    f.id, f.content, f.metadata, f.status, f.owner,
    f.status_updated_at, f.created_at,
    c.cnt AS total_count
  FROM filtered f CROSS JOIN counted c
  ORDER BY f.status_updated_at DESC NULLS LAST, f.created_at DESC, f.id
  OFFSET greatest(0, coalesce(p_offset, 0))
  LIMIT greatest(1, least(coalesce(p_limit, 50), 200));
END;
$$;

-- SECURITY DEFINER bypasses RLS on `thoughts`. Do NOT grant to `anon` — that
-- would expose the whole brain to anyone with the project URL. See README.
GRANT EXECUTE ON FUNCTION list_open_loops(TEXT[], TEXT, INT, INT)
  TO authenticated, service_role;

-- ============================================================
-- 4. match_thoughts_hybrid()
--    Blends semantic similarity (pgvector cosine) with keyword rank
--    (tsvector + ILIKE fallback) so recall stops depending on echoing a
--    capture's exact vocabulary. Paginated, optional status filter,
--    window total_count. Self-contained (no enhanced-thoughts columns).
-- ============================================================

CREATE OR REPLACE FUNCTION match_thoughts_hybrid(
  query_embedding vector(1536),
  p_query_text TEXT DEFAULT '',
  match_count INT DEFAULT 10,
  match_offset INT DEFAULT 0,
  p_status TEXT[] DEFAULT NULL,
  semantic_weight FLOAT DEFAULT 0.6
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  status TEXT,
  created_at TIMESTAMPTZ,
  similarity FLOAT,
  text_rank FLOAT,
  score FLOAT,
  total_count BIGINT
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
SET statement_timeout = '25s'
AS $$
DECLARE
  v_weight FLOAT := least(1.0, greatest(0.0, coalesce(semantic_weight, 0.6)));
BEGIN
  RETURN QUERY
  WITH q AS (
    SELECT
      trim(coalesce(p_query_text, '')) AS raw_q,
      websearch_to_tsquery('simple', trim(coalesce(p_query_text, ''))) AS tsq
  ),
  scored AS (
    SELECT
      t.id, t.content, t.metadata, t.status, t.created_at,
      CASE
        WHEN t.embedding IS NOT NULL
          THEN (1 - (t.embedding <=> query_embedding))::float
        ELSE 0
      END AS sim,
      greatest(
        CASE
          WHEN q.raw_q <> ''
            THEN ts_rank_cd(to_tsvector('simple', coalesce(t.content, '')), q.tsq)
          ELSE 0
        END,
        CASE
          WHEN q.raw_q <> '' AND t.content ILIKE '%' || q.raw_q || '%'
            THEN 0.35
          ELSE 0
        END
      )::float AS txt
    FROM public.thoughts t CROSS JOIN q
    WHERE (p_status IS NULL OR t.status = ANY(p_status))
      AND (
        (t.embedding IS NOT NULL AND (1 - (t.embedding <=> query_embedding)) > 0.15)
        OR (q.raw_q <> '' AND to_tsvector('simple', coalesce(t.content, '')) @@ q.tsq)
        OR (q.raw_q <> '' AND t.content ILIKE '%' || q.raw_q || '%')
      )
  ),
  combined AS (
    SELECT
      s.*,
      (v_weight * greatest(s.sim, 0) + (1 - v_weight) * least(s.txt, 1.0))::float AS score
    FROM scored s
  ),
  counted AS (
    SELECT count(*) AS cnt FROM combined
  )
  SELECT
    c.id, c.content, c.metadata, c.status, c.created_at,
    c.sim AS similarity, c.txt AS text_rank, c.score,
    cc.cnt AS total_count
  FROM combined c CROSS JOIN counted cc
  ORDER BY c.score DESC, c.created_at DESC, c.id
  OFFSET greatest(0, coalesce(match_offset, 0))
  LIMIT greatest(1, least(coalesce(match_count, 10), 200));
END;
$$;

-- SECURITY DEFINER bypasses RLS on `thoughts`. Do NOT grant to `anon`. See README.
GRANT EXECUTE ON FUNCTION
  match_thoughts_hybrid(vector, TEXT, INT, INT, TEXT[], FLOAT)
  TO authenticated, service_role;

-- ============================================================
-- 5. set_thought_status()
--    Transition a single loop's status and stamp status_updated_at.
--    Used by the MCP set_thought_status tool so closed loops stop resurfacing.
-- ============================================================

CREATE OR REPLACE FUNCTION set_thought_status(
  p_id UUID,
  p_status TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_row public.thoughts%ROWTYPE;
BEGIN
  IF p_status IS NULL OR p_status NOT IN ('open', 'waiting', 'closed') THEN
    RAISE EXCEPTION 'invalid status %, expected one of open|waiting|closed', p_status;
  END IF;

  UPDATE public.thoughts
  SET status = p_status,
      status_updated_at = now()
  WHERE id = p_id
  RETURNING * INTO v_row;

  IF NOT FOUND THEN
    RETURN jsonb_build_object('updated', false, 'id', p_id);
  END IF;

  RETURN jsonb_build_object(
    'updated', true,
    'id', v_row.id,
    'status', v_row.status,
    'status_updated_at', v_row.status_updated_at
  );
END;
$$;

GRANT EXECUTE ON FUNCTION set_thought_status(UUID, TEXT) TO service_role;

-- ============================================================
-- 6. BACKFILL EXISTING DATA  (qualified UPDATEs only — re-runnable)
--    Classifies existing task thoughts so today's loops are queryable now.
--    Wrapped in a function so it is idempotent, testable, and re-runnable.
-- ============================================================

CREATE OR REPLACE FUNCTION backfill_open_loop_status()
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SET search_path = public
AS $$
DECLARE
  v_norm   BIGINT;
  v_closed BIGINT;
  v_wait   BIGINT;
  v_open   BIGINT;
  v_owner  BIGINT;
BEGIN
  -- Normalize the legacy kanban 'new' status (workflow-status /
  -- enhanced-thoughts) to the open/waiting/closed vocabulary.
  UPDATE public.thoughts
  SET status = 'open', status_updated_at = now()
  WHERE status = 'new';
  GET DIAGNOSTICS v_norm = ROW_COUNT;

  -- Closed: resolution markers in the content.
  UPDATE public.thoughts
  SET status = 'closed', status_updated_at = now()
  WHERE metadata->>'type' = 'task'
    AND status IS DISTINCT FROM 'closed'
    AND (
      content ILIKE '%CLOSED loop%'
      OR content ILIKE '%RESOLVED / DONE%'
      OR content ILIKE '%RESOLVED/DONE%'
      OR content ILIKE '%— DONE%'
      OR content ILIKE '% is now DONE%'
      OR content ILIKE '%this loop is CLOSED%'
    );
  GET DIAGNOSTICS v_closed = ROW_COUNT;

  -- Waiting: blocked-on-someone-else markers.
  UPDATE public.thoughts
  SET status = 'waiting', status_updated_at = now()
  WHERE metadata->>'type' = 'task'
    AND status IS DISTINCT FROM 'closed'
    AND status IS DISTINCT FROM 'waiting'
    AND (
      content ILIKE '%WAITING ON OTHERS%'
      OR content ILIKE '%THEIR open loop%'
      OR content ILIKE '%ball in %court%'
      OR (content ILIKE '%gates %' AND content ILIKE '%THEIR%')
    );
  GET DIAGNOSTICS v_wait = ROW_COUNT;

  -- Everything else that is a task and still unclassified defaults to open.
  UPDATE public.thoughts
  SET status = 'open', status_updated_at = now()
  WHERE metadata->>'type' = 'task'
    AND status IS NULL;
  GET DIAGNOSTICS v_open = ROW_COUNT;

  -- Best-effort owner tagging for David's own loops (heuristic — verify before
  -- trusting the owner filter). Only sets owner where none is present.
  UPDATE public.thoughts
  SET metadata = jsonb_set(metadata, '{owner}', '"David"', true)
  WHERE metadata->>'type' = 'task'
    AND (metadata->>'owner') IS NULL
    AND (
      content ILIKE '%OUR open loop (David)%'
      OR content ILIKE '%David to %'
      OR content ILIKE '%David Ventimiglia to %'
      OR content ILIKE '%David owes%'
      OR content ILIKE '%David''s to-do%'
      OR content ILIKE '%OUR open loop (joint David%'
    );
  GET DIAGNOSTICS v_owner = ROW_COUNT;

  RETURN jsonb_build_object(
    'normalized_new', v_norm,
    'closed', v_closed,
    'waiting', v_wait,
    'open', v_open,
    'owner_tagged', v_owner
  );
END;
$$;

GRANT EXECUTE ON FUNCTION backfill_open_loop_status() TO service_role;

-- Run the backfill once as part of applying this schema (paste-and-run flow).
SELECT backfill_open_loop_status();

-- ============================================================
-- 7. Reload PostgREST schema cache so the new RPCs are callable immediately.
-- ============================================================

NOTIFY pgrst, 'reload schema';
