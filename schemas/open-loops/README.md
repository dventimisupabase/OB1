# Open Loops

> Status tracking, exhaustive paginated listing, and phrasing-robust hybrid recall for your open loops.

## What It Does

Open Brain is great at *storing* commitments but, out of the box, hard to *exhaustively retrieve* — there is no status, the `list`/`search` tools cap at 10 results with no pagination, and pure semantic search is brittle to phrasing. This schema fixes the data layer for that:

- Adds a **status lifecycle** (`open` → `waiting` → `closed`) on the `thoughts` table (`status`, `status_updated_at`).
- `list_open_loops(p_status, p_owner, p_limit, p_offset)` — the headline "list all open loops" query, with **real pagination** (offset/limit, capped at 200) and a window `total_count`.
- `match_thoughts_hybrid(query_embedding, p_query_text, ...)` — blends **semantic similarity** (pgvector cosine) with **keyword rank** (tsvector + ILIKE fallback) so recall no longer depends on echoing a capture's exact vocabulary.
- `set_thought_status(p_id, p_status)` — transition a single loop so closed loops stop resurfacing.
- `backfill_open_loop_status()` — a re-runnable, **qualified** UPDATE that classifies existing task thoughts from their text markers (`THEIR open loop` → waiting, `RESOLVED / DONE` / `CLOSED loop` → closed, else → open) and best-effort tags David-owned loops.

It standardizes the vocabulary of [`schemas/workflow-status`](../workflow-status/) (which seeds `'new'`) to `open`/`waiting`/`closed`, and shares the same `status` columns and `idx_thoughts_content_tsvector` index as [`schemas/enhanced-thoughts`](../enhanced-thoughts/) — all column/index adds use `IF NOT EXISTS`, so applying any combination is safe.

The companion MCP tools (`list_open_loops`, `set_thought_status`, plus status-aware `list_thoughts`/`search_thoughts`) live in the core `open-brain-mcp` server.

## Prerequisites

- Working Open Brain setup ([guide](../../docs/01-getting-started.md)) — the core `thoughts` table and `match_thoughts` RPC.
- `pgvector` enabled (already required by core Open Brain).

## Credential Tracker

Copy this block into a text editor and fill it in as you go.

```text
OPEN LOOPS -- CREDENTIAL TRACKER
--------------------------------------

SUPABASE (from your Open Brain setup)
  Project URL:           ____________
  Secret key:            ____________

--------------------------------------
```

## Steps

1. Open your Supabase SQL Editor (or use the Supabase CLI).
2. Run the migration:

   ```sql
   -- Paste the contents of schema.sql, or via CLI:
   -- psql "$DB_URL" -v ON_ERROR_STOP=1 -f schemas/open-loops/schema.sql
   ```

3. (Optional) Re-run the backfill at any time:

   ```sql
   SELECT backfill_open_loop_status();
   ```

4. Verify (see Expected Outcome). A test script is provided:

   ```sh
   psql "$DB_URL" -v ON_ERROR_STOP=1 -f schemas/open-loops/test.sql
   ```

## Expected Outcome

After running, the `thoughts` table has `status` + `status_updated_at` columns, and these functions exist and are callable by `service_role`:

- `list_open_loops(text[], text, int, int)` → returns rows + `total_count`
- `match_thoughts_hybrid(vector, text, int, int, text[], float)` → ranked rows + `total_count`
- `set_thought_status(uuid, text)` → `{updated, id, status, status_updated_at}`
- `backfill_open_loop_status()` → JSON counts of rows classified

Quick check:

```sql
SELECT status, count(*) FROM thoughts GROUP BY status ORDER BY 1;
SELECT id, status, owner, left(content, 60) FROM list_open_loops(ARRAY['open','waiting'], 'David', 50, 0);
```

`test.sql` should print `ALL OPEN-LOOPS SCHEMA TESTS PASSED`.

## Security

`list_open_loops`, `match_thoughts_hybrid`, and `set_thought_status` are `SECURITY DEFINER` and bypass RLS on `thoughts`. They are granted to `authenticated` and `service_role` only — **never** grant them to `anon`, which would expose the entire brain to anyone with the project URL. The default Open Brain RLS keeps `thoughts` behind `service_role`.

## Troubleshooting

**Issue: `function match_thoughts_hybrid(...) does not exist` from the MCP server.**
Solution: Run `NOTIFY pgrst, 'reload schema';` (included at the end of `schema.sql`) so PostgREST/RPC picks up the new functions, or restart the API.

**Issue: backfill left some loops with the wrong status.**
Solution: The backfill is a best-effort heuristic on content markers. Correct individual rows with `SELECT set_thought_status('<uuid>', 'open|waiting|closed');`. Re-running `backfill_open_loop_status()` will not overwrite a status you set unless the content matches a stronger marker (closed wins over waiting).

**Issue: `owner` filter returns fewer rows than expected.**
Solution: Owner tagging is heuristic (it looks for "David to …" style phrasing). Set it explicitly with `UPDATE thoughts SET metadata = jsonb_set(metadata, '{owner}', '"David"') WHERE id = '<uuid>';`.
