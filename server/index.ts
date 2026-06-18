import "jsr:@supabase/functions-js/edge-runtime.d.ts";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPTransport } from "@hono/mcp";
import { Hono } from "hono";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OPENROUTER_API_KEY = Deno.env.get("OPENROUTER_API_KEY")!;
const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY")!;

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

type ThoughtMatch = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
  created_at: string;
};

type ThoughtRecord = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at?: string | null;
};

// Row shape returned by match_thoughts_hybrid (and, minus the extra fields, the
// match_thoughts fallback).
type HybridMatch = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  status?: string | null;
  created_at: string;
  similarity: number;
  text_rank?: number;
  score?: number;
  total_count?: number;
};

const CITATION_BASE_URL =
  Deno.env.get("OPEN_BRAIN_CITATION_BASE_URL") || "https://openbrain.local/thoughts";

function thoughtTitle(content: string, createdAt?: string): string {
  const firstLine = content.replace(/\s+/g, " ").trim().slice(0, 80);
  const datePrefix = createdAt ? new Date(createdAt).toLocaleDateString() : "Open Brain";
  return firstLine ? `${datePrefix} - ${firstLine}` : `${datePrefix} thought`;
}

function thoughtUrl(id: string): string {
  return `${CITATION_BASE_URL.replace(/\/$/, "")}/${id}`;
}

// Open-loop status vocabulary (see schemas/open-loops).
const OPEN_LOOP_STATUSES = ["open", "waiting", "closed"] as const;
type OpenLoopStatus = (typeof OPEN_LOOP_STATUSES)[number];

// Clamp a requested page size into [1, max]; fall back to def when unset.
function clampLimit(n: number | undefined, def: number, max: number): number {
  const v = typeof n === "number" && Number.isFinite(n) ? Math.floor(n) : def;
  return Math.max(1, Math.min(v, max));
}

// Non-negative offset.
function safeOffset(n: number | undefined): number {
  const v = typeof n === "number" && Number.isFinite(n) ? Math.floor(n) : 0;
  return Math.max(0, v);
}

// "Showing X–Y of N" footer for paginated tool output. Returns "" when there
// is nothing more to page through.
function pageFooter(offset: number, returned: number, total: number | null): string {
  if (total == null) return "";
  const first = total === 0 ? 0 : offset + 1;
  const last = offset + returned;
  let footer = `\nShowing ${first}–${last} of ${total}.`;
  if (last < total) footer += ` Pass offset=${last} for the next page.`;
  return footer;
}

async function getEmbedding(text: string): Promise<number[]> {
  const r = await fetch(`${OPENROUTER_BASE}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/text-embedding-3-small",
      input: text,
    }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    throw new Error(`OpenRouter embeddings failed: ${r.status} ${msg}`);
  }
  const d = await r.json();
  return d.data[0].embedding;
}

async function extractMetadata(text: string): Promise<Record<string, unknown>> {
  const r = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: `You classify a captured thought for a personal "second brain". Return STRICT JSON.

Keys for every thought:
- "type": one of "observation" | "task" | "idea" | "reference" | "person_note".
  Choose by INTENT, not keywords:
    • task = something someone still needs to DO or follow up on (an open loop, a commitment, an owed action).
    • observation = a fact, decision, resolution, or outcome — what happened or is true. A DECIDED or FINISHED thing is an observation, NOT a task, even if it contains action verbs ("decided to", "resolved", "shipped", "done").
    • idea = a proposal or possibility not yet committed to.
    • reference = durable reference info (links, specs, how-tos).
    • person_note = a note primarily about a person.
- "people": array of people mentioned (empty if none)
- "action_items": array of implied to-dos (empty if none)
- "dates_mentioned": array of dates YYYY-MM-DD (empty if none)
- "topics": array of 1-3 short topic tags (always at least one)

ONLY for "type":"task", also return:
- "status": "open" | "waiting" | "closed".
    • open = the author still owes the action / it is theirs to do next.
    • waiting = blocked on someone else; the next move is another party's ("THEIR open loop", "waiting on X").
    • closed = already done / resolved.
- "owner": who owes the action — a name, or "me" when it is the author's own. Empty string if unclear.

Only extract what is explicitly there. Examples:
{"input":"David to run the Greptile fitness check next week.","output":{"type":"task","status":"open","owner":"David","topics":["greptile"],"people":["David"],"action_items":["run fitness check"],"dates_mentioned":[]}}
{"input":"Eric to send the snapshot queries — THEIR open loop.","output":{"type":"task","status":"waiting","owner":"Eric","topics":["queries"],"people":["Eric"],"action_items":["send snapshot queries"],"dates_mentioned":[]}}
{"input":"Decided to upgrade to Postgres 17 after testing in a shadow project.","output":{"type":"observation","topics":["postgres","upgrade"],"people":[],"action_items":[],"dates_mentioned":[]}}
{"input":"CLOSED loop — posted the database fitness doc to the Slack channel.","output":{"type":"task","status":"closed","owner":"me","topics":["fitness-doc"],"people":[],"action_items":[],"dates_mentioned":[]}}`,
        },
        { role: "user", content: text },
      ],
    }),
  });
  const d = await r.json();
  try {
    return JSON.parse(d.choices[0].message.content);
  } catch {
    return { topics: ["uncategorized"], type: "observation" };
  }
}

// --- MCP Server Setup ---

function buildServer(): McpServer {
  const server = new McpServer({
    name: "open-brain",
    version: "1.0.0",
  });

  // ChatGPT compatibility: restricted connector surfaces, company knowledge, and deep
  // research look for exact read-only `search` and `fetch` tool shapes.
  server.registerTool(
    "search",
    {
      title: "Search Open Brain",
      description:
        "Search Open Brain memories by meaning. Use this read-only compatibility tool when ChatGPT needs search/fetch-style access to stored thoughts.",
      annotations: {
        readOnlyHint: true,
      },
      inputSchema: {
        query: z.string().describe("The search query to run against Open Brain thoughts"),
      },
    },
    async ({ query }) => {
      try {
        const qEmb = await getEmbedding(query);
        const { data, error } = await supabase.rpc("match_thoughts", {
          query_embedding: qEmb,
          match_threshold: 0.5,
          match_count: 10,
          filter: {},
        });

        if (error) {
          return {
            content: [{ type: "text" as const, text: `Search error: ${error.message}` }],
            isError: true,
          };
        }

        const results = ((data || []) as ThoughtMatch[]).map((t) => ({
          id: t.id,
          title: thoughtTitle(t.content, t.created_at),
          url: thoughtUrl(t.id),
        }));

        return {
          content: [{ type: "text" as const, text: JSON.stringify({ results }) }],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  server.registerTool(
    "fetch",
    {
      title: "Fetch Open Brain Thought",
      description:
        "Fetch one Open Brain thought by ID after using search. Use this read-only compatibility tool to retrieve the full text and metadata for citation.",
      annotations: {
        readOnlyHint: true,
      },
      inputSchema: {
        id: z.string().describe("The Open Brain thought ID returned by the search tool"),
      },
    },
    async ({ id }) => {
      try {
        const { data, error } = await supabase
          .from("thoughts")
          .select("id, content, metadata, created_at, updated_at")
          .eq("id", id)
          .single();

        if (error) {
          return {
            content: [{ type: "text" as const, text: `Fetch error: ${error.message}` }],
            isError: true,
          };
        }

        const thought = data as ThoughtRecord;
        const document = {
          id: thought.id,
          title: thoughtTitle(thought.content, thought.created_at),
          text: thought.content,
          url: thoughtUrl(thought.id),
          metadata: {
            ...thought.metadata,
            created_at: thought.created_at,
            updated_at: thought.updated_at,
          },
        };

        return {
          content: [{ type: "text" as const, text: JSON.stringify(document) }],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 1: Semantic Search
  server.registerTool(
    "search_thoughts",
    {
      title: "Search Thoughts",
      description:
        "Search captured thoughts by meaning. Use this when the user asks about a topic, person, or idea they've previously captured.",
      annotations: {
        readOnlyHint: true,
      },
      inputSchema: {
        query: z.string().describe("What to search for"),
        limit: z.number().optional().default(10).describe("Page size (default 10, max 200)"),
        offset: z.number().optional().describe("Skip this many ranked results for pagination (default 0)"),
        threshold: z.number().optional().default(0.5).describe("Min semantic similarity for the pure-semantic fallback"),
        status: z.array(z.string()).optional().describe("Restrict to these open-loop statuses, e.g. [\"open\",\"waiting\"]"),
      },
    },
    async ({ query, limit, offset, threshold, status }) => {
      try {
        const lim = clampLimit(limit, 10, 200);
        const off = safeOffset(offset);
        const statusFilter = status && status.length ? status : null;
        const qEmb = await getEmbedding(query);

        // Prefer hybrid (semantic + keyword) ranking; fall back to pure semantic
        // if match_thoughts_hybrid isn't installed (schemas/open-loops not applied).
        let rows: HybridMatch[] = [];
        let total: number | null = null;
        let hybrid = true;

        const h = await supabase.rpc("match_thoughts_hybrid", {
          query_embedding: qEmb,
          p_query_text: query,
          match_count: lim,
          match_offset: off,
          p_status: statusFilter,
          semantic_weight: 0.6,
        });

        if (h.error) {
          hybrid = false;
          const s = await supabase.rpc("match_thoughts", {
            query_embedding: qEmb,
            match_threshold: threshold,
            match_count: lim,
            filter: {},
          });
          if (s.error) {
            return {
              content: [{ type: "text" as const, text: `Search error: ${s.error.message}` }],
              isError: true,
            };
          }
          rows = (s.data || []) as HybridMatch[];
        } else {
          rows = (h.data || []) as HybridMatch[];
          total = rows.length ? Number(rows[0].total_count ?? rows.length) : 0;
        }

        if (!rows.length) {
          return {
            content: [{ type: "text" as const, text: `No thoughts found matching "${query}".` }],
          };
        }

        const results = rows.map((t: HybridMatch, i: number) => {
          const m = t.metadata || {};
          const pct = typeof t.score === "number" ? t.score : t.similarity;
          const parts = [
            `--- Result ${off + i + 1} (${(pct * 100).toFixed(1)}% ${hybrid ? "hybrid" : "match"}) ---`,
            `Captured: ${new Date(t.created_at).toLocaleDateString()}`,
            `Type: ${m.type || "unknown"}${t.status ? ` {${t.status}}` : ""}`,
          ];
          if (Array.isArray(m.topics) && m.topics.length)
            parts.push(`Topics: ${(m.topics as string[]).join(", ")}`);
          if (Array.isArray(m.people) && m.people.length)
            parts.push(`People: ${(m.people as string[]).join(", ")}`);
          if (Array.isArray(m.action_items) && m.action_items.length)
            parts.push(`Actions: ${(m.action_items as string[]).join("; ")}`);
          parts.push(`\n${t.content}`);
          return parts.join("\n");
        });

        return {
          content: [
            {
              type: "text" as const,
              text: `Found thought(s):\n\n${results.join("\n\n")}${pageFooter(off, rows.length, total)}`,
            },
          ],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 2: List Recent
  server.registerTool(
    "list_thoughts",
    {
      title: "List Recent Thoughts",
      description:
        "List recently captured thoughts with optional filters by type, topic, person, or time range.",
      annotations: {
        readOnlyHint: true,
      },
      inputSchema: {
        limit: z.number().optional().default(25).describe("Page size (default 25, max 200)"),
        offset: z.number().optional().describe("Skip this many rows for pagination (default 0)"),
        type: z.string().optional().describe("Filter by type: observation, task, idea, reference, person_note"),
        topic: z.string().optional().describe("Filter by topic tag"),
        person: z.string().optional().describe("Filter by person mentioned"),
        status: z.string().optional().describe("Filter by open-loop status: open, waiting, closed"),
        days: z.number().optional().describe("Only thoughts from the last N days"),
      },
    },
    async ({ limit, offset, type, topic, person, status, days }) => {
      try {
        const lim = clampLimit(limit, 25, 200);
        const off = safeOffset(offset);

        let q = supabase
          .from("thoughts")
          .select("content, metadata, created_at, status", { count: "exact" })
          .order("created_at", { ascending: false })
          .range(off, off + lim - 1);

        if (type) q = q.contains("metadata", { type });
        if (topic) q = q.contains("metadata", { topics: [topic] });
        if (person) q = q.contains("metadata", { people: [person] });
        if (status) q = q.eq("status", status);
        if (days) {
          const since = new Date();
          since.setDate(since.getDate() - days);
          q = q.gte("created_at", since.toISOString());
        }

        const { data, error, count } = await q;

        if (error) {
          return {
            content: [{ type: "text" as const, text: `Error: ${error.message}` }],
            isError: true,
          };
        }

        if (!data || !data.length) {
          return { content: [{ type: "text" as const, text: "No thoughts found." }] };
        }

        const results = data.map(
          (
            t: { content: string; metadata: Record<string, unknown>; created_at: string; status?: string | null },
            i: number
          ) => {
            const m = t.metadata || {};
            const tags = Array.isArray(m.topics) ? (m.topics as string[]).join(", ") : "";
            const st = t.status ? ` {${t.status}}` : "";
            return `${off + i + 1}. [${new Date(t.created_at).toLocaleDateString()}] (${m.type || "??"}${tags ? " - " + tags : ""})${st}\n   ${t.content}`;
          }
        );

        return {
          content: [
            {
              type: "text" as const,
              text: `Recent thought(s):\n\n${results.join("\n\n")}${pageFooter(off, data.length, count ?? null)}`,
            },
          ],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 3: Stats
  server.registerTool(
    "thought_stats",
    {
      title: "Thought Statistics",
      description: "Get a summary of all captured thoughts: totals, types, top topics, and people.",
      annotations: {
        readOnlyHint: true,
      },
      inputSchema: {},
    },
    async () => {
      try {
        const { count } = await supabase
          .from("thoughts")
          .select("*", { count: "exact", head: true });

        const { data } = await supabase
          .from("thoughts")
          .select("metadata, created_at")
          .order("created_at", { ascending: false });

        const types: Record<string, number> = {};
        const topics: Record<string, number> = {};
        const people: Record<string, number> = {};

        for (const r of data || []) {
          const m = (r.metadata || {}) as Record<string, unknown>;
          if (m.type) types[m.type as string] = (types[m.type as string] || 0) + 1;
          if (Array.isArray(m.topics))
            for (const t of m.topics) topics[t as string] = (topics[t as string] || 0) + 1;
          if (Array.isArray(m.people))
            for (const p of m.people) people[p as string] = (people[p as string] || 0) + 1;
        }

        const sort = (o: Record<string, number>): [string, number][] =>
          Object.entries(o)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        const lines: string[] = [
          `Total thoughts: ${count}`,
          `Date range: ${
            data?.length
              ? new Date(data[data.length - 1].created_at).toLocaleDateString() +
                " → " +
                new Date(data[0].created_at).toLocaleDateString()
              : "N/A"
          }`,
          "",
          "Types:",
          ...sort(types).map(([k, v]) => `  ${k}: ${v}`),
        ];

        if (Object.keys(topics).length) {
          lines.push("", "Top topics:");
          for (const [k, v] of sort(topics)) lines.push(`  ${k}: ${v}`);
        }

        if (Object.keys(people).length) {
          lines.push("", "People mentioned:");
          for (const [k, v] of sort(people)) lines.push(`  ${k}: ${v}`);
        }

        // Open-loop status breakdown (best-effort: the `status` column only
        // exists once schemas/open-loops has been applied — a missing column
        // returns an error we simply skip rather than failing the whole stat).
        const statusRes = await supabase.from("thoughts").select("status");
        if (!statusRes.error && statusRes.data) {
          const statuses: Record<string, number> = {};
          for (const r of statusRes.data as { status: string | null }[]) {
            if (r.status) statuses[r.status] = (statuses[r.status] || 0) + 1;
          }
          if (Object.keys(statuses).length) {
            lines.push("", "Open loops:");
            for (const k of ["open", "waiting", "closed"]) {
              if (statuses[k]) lines.push(`  ${k}: ${statuses[k]}`);
            }
          }
        }

        return { content: [{ type: "text" as const, text: lines.join("\n") }] };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 4: List Open Loops
  server.registerTool(
    "list_open_loops",
    {
      title: "List Open Loops",
      description:
        "Enumerate your open loops — tasks/commitments — by status, with pagination. This is the way to get the COMPLETE outstanding-commitment list; unlike a recency list it is not capped at 10. Status is any of: open (you owe it), waiting (blocked on someone else), closed (done). Optionally filter by owner (e.g. 'David').",
      annotations: { readOnlyHint: true },
      inputSchema: {
        status: z
          .array(z.enum(OPEN_LOOP_STATUSES))
          .optional()
          .describe("Statuses to include (default: open + waiting)"),
        owner: z.string().optional().describe("Only loops owned by this person (e.g. 'David')"),
        limit: z.number().optional().default(50).describe("Page size (default 50, max 200)"),
        offset: z.number().optional().describe("Skip this many rows for pagination (default 0)"),
      },
    },
    async ({ status, owner, limit, offset }) => {
      try {
        const statuses = status && status.length ? status : ["open", "waiting"];
        const lim = clampLimit(limit, 50, 200);
        const off = safeOffset(offset);

        const { data, error } = await supabase.rpc("list_open_loops", {
          p_status: statuses,
          p_owner: owner ?? null,
          p_limit: lim,
          p_offset: off,
        });

        if (error) {
          return {
            content: [{ type: "text" as const, text: `Error: ${error.message}` }],
            isError: true,
          };
        }

        const rows = (data || []) as Array<{
          id: string;
          content: string;
          metadata: Record<string, unknown>;
          status: string;
          owner: string | null;
          created_at: string;
          total_count: number;
        }>;

        if (!rows.length) {
          return {
            content: [
              {
                type: "text" as const,
                text: `No open loops found for status ${statuses.join("/")}${owner ? ` owned by ${owner}` : ""}.`,
              },
            ],
          };
        }

        const total = Number(rows[0].total_count ?? rows.length);
        const icon: Record<string, string> = { open: "🔴", waiting: "🟡", closed: "✅" };
        const lines = rows.map((t, i) => {
          const o = t.owner ? ` — ${t.owner}` : "";
          return `${off + i + 1}. ${icon[t.status] || ""} [${t.status}${o}] (${new Date(t.created_at).toLocaleDateString()})\n   id: ${t.id}\n   ${t.content}`;
        });

        return {
          content: [
            {
              type: "text" as const,
              text: `Open loops (${statuses.join(", ")}${owner ? `, owner=${owner}` : ""}):\n\n${lines.join("\n\n")}${pageFooter(off, rows.length, total)}`,
            },
          ],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 5: Set Thought Status
  server.registerTool(
    "set_thought_status",
    {
      title: "Set Thought Status",
      description:
        "Set the open-loop status of a thought (open, waiting, or closed) so resolved loops stop resurfacing. Pass the thought id shown by list_open_loops or fetch.",
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
      inputSchema: {
        id: z.string().describe("The thought ID (UUID)"),
        status: z.enum(OPEN_LOOP_STATUSES).describe("New status: open, waiting, or closed"),
      },
    },
    async ({ id, status }) => {
      try {
        const { data, error } = await supabase.rpc("set_thought_status", {
          p_id: id,
          p_status: status,
        });

        if (error) {
          return {
            content: [{ type: "text" as const, text: `Error: ${error.message}` }],
            isError: true,
          };
        }

        const res = (data || {}) as { updated?: boolean; status?: string };
        if (!res.updated) {
          return { content: [{ type: "text" as const, text: `No thought found with id ${id}.` }] };
        }

        return {
          content: [{ type: "text" as const, text: `Updated ${id} → ${res.status}.` }],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  // Tool 6: Capture Thought
  server.registerTool(
    "capture_thought",
    {
      title: "Capture Thought",
      description:
        "Save a new thought to the Open Brain. Generates an embedding and extracts metadata automatically. Use this when the user wants to save something to their brain directly from any AI client — notes, insights, decisions, or migrated content from other systems.",
      annotations: {
        readOnlyHint: false,
        openWorldHint: false,
        destructiveHint: false,
        idempotentHint: false,
      },
      inputSchema: {
        content: z.string().describe("The thought to capture — a clear, standalone statement that will make sense when retrieved later by any AI"),
      },
    },
    async ({ content }) => {
      try {
        const [embedding, metadata] = await Promise.all([
          getEmbedding(content),
          extractMetadata(content),
        ]);

        const { data: upsertResult, error: upsertError } = await supabase.rpc("upsert_thought", {
          p_content: content,
          p_payload: { metadata: { ...metadata, source: "mcp" } },
        });

        if (upsertError) {
          return {
            content: [{ type: "text" as const, text: `Failed to capture: ${upsertError.message}` }],
            isError: true,
          };
        }

        const thoughtId = upsertResult?.id;
        const { error: embError } = await supabase
          .from("thoughts")
          .update({ embedding })
          .eq("id", thoughtId);

        if (embError) {
          return {
            content: [{ type: "text" as const, text: `Failed to save embedding: ${embError.message}` }],
            isError: true,
          };
        }

        const meta = metadata as Record<string, unknown>;

        // For task-type thoughts, give them an open-loop status. The value also
        // rides along in metadata (above); writing the dedicated `status` column
        // is best-effort — it only exists once schemas/open-loops (or
        // workflow-status / enhanced-thoughts) has been applied, so a missing
        // column must not fail the capture.
        let status: OpenLoopStatus | null = null;
        if (meta.type === "task") {
          const raw = typeof meta.status === "string" ? meta.status.toLowerCase() : "";
          status = (OPEN_LOOP_STATUSES as readonly string[]).includes(raw)
            ? (raw as OpenLoopStatus)
            : "open";
          await supabase
            .from("thoughts")
            .update({ status, status_updated_at: new Date().toISOString() })
            .eq("id", thoughtId);
        }

        let confirmation = `Captured as ${meta.type || "thought"}`;
        if (status) confirmation += ` [${status}]`;
        if (Array.isArray(meta.topics) && meta.topics.length)
          confirmation += ` — ${(meta.topics as string[]).join(", ")}`;
        if (typeof meta.owner === "string" && meta.owner)
          confirmation += ` | Owner: ${meta.owner}`;
        if (Array.isArray(meta.people) && meta.people.length)
          confirmation += ` | People: ${(meta.people as string[]).join(", ")}`;
        if (Array.isArray(meta.action_items) && meta.action_items.length)
          confirmation += ` | Actions: ${(meta.action_items as string[]).join("; ")}`;

        return {
          content: [{ type: "text" as const, text: confirmation }],
        };
      } catch (err: unknown) {
        return {
          content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
          isError: true,
        };
      }
    }
  );

  return server;
}

// --- Hono App with Auth + CORS ---

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-brain-key, accept, mcp-session-id, mcp-protocol-version, last-event-id",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS, DELETE",
};

// JSON-RPC error code for unauthorized requests.
// Per the JSON-RPC 2.0 spec, the range -32099 to -32000 is reserved for
// implementation-defined server errors. -32001 is the conventional
// "Unauthorized" code used by MCP clients/servers in the wild.
//
// Why a JSON-RPC envelope (HTTP 200) instead of a bare HTTP 401?
// Strict MCP hosts (Codex CLI, Claude Code) treat bare HTTP 4xx responses
// as transport-level failures and tear the connection down rather than
// surfacing the failure to the application layer. Wrapping the auth
// rejection in a JSON-RPC error keeps the connection alive and lets
// clients recover (e.g. prompt the user for a new key, refetch a stale
// cache) instead of dying.
const JSON_RPC_UNAUTHORIZED_CODE = -32001;
const UNAUTHORIZED_MESSAGE = "Unauthorized: missing or invalid authentication.";

/**
 * Read the request body as text without consuming the original request's
 * body stream for downstream handlers. Returns null on bodyless methods
 * or read failure.
 */
async function readBodyText(req: Request): Promise<string | null> {
  if (req.method === "GET" || req.method === "HEAD" || req.method === "DELETE") {
    return null;
  }
  try {
    return await req.text();
  } catch {
    return null;
  }
}

/**
 * Best-effort extraction of the JSON-RPC `id` from a raw request body.
 * Returns null when the body is missing, not JSON, or not a JSON-RPC
 * shape with an id. Per the JSON-RPC 2.0 spec, id may be a string,
 * number, or null — we preserve any of those; anything else becomes null.
 */
function extractJsonRpcId(bodyText: string | null): string | number | null {
  if (!bodyText) return null;
  try {
    const parsed = JSON.parse(bodyText);
    if (parsed && typeof parsed === "object" && "id" in parsed) {
      const id = (parsed as { id: unknown }).id;
      if (typeof id === "string" || typeof id === "number" || id === null) {
        return id;
      }
    }
  } catch {
    // fall through — malformed body
  }
  return null;
}

/**
 * Build a JSON-RPC 2.0 error envelope response for auth failures.
 * Returns HTTP 200 — the JSON-RPC layer expresses the error so that
 * strict MCP clients keep the connection alive instead of treating
 * the failure as a transport-level fault.
 */
function unauthorizedResponse(id: string | number | null): Response {
  const body = {
    jsonrpc: "2.0",
    error: {
      code: JSON_RPC_UNAUTHORIZED_CODE,
      message: UNAUTHORIZED_MESSAGE,
    },
    id,
  };
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders,
    },
  });
}

const app = new Hono();

// CORS preflight — required for browser/Electron-based clients (Claude Desktop, claude.ai)
app.options("*", (c) => {
  return c.text("ok", 200, corsHeaders);
});

app.all("*", async (c) => {
  // Accept access key via header OR URL query parameter
  const provided = c.req.header("x-brain-key") || new URL(c.req.url).searchParams.get("key");
  if (!provided || provided !== MCP_ACCESS_KEY) {
    // Return a JSON-RPC 2.0 error envelope (HTTP 200) instead of a bare
    // HTTP 401 so strict MCP hosts treat this as an application-level
    // error rather than a transport fault and keep the connection alive.
    // Best-effort echo of the inbound request id keeps the response
    // correlated; malformed/missing bodies fall back to id: null.
    const bodyText = await readBodyText(c.req.raw);
    const id = extractJsonRpcId(bodyText);
    return unauthorizedResponse(id);
  }

  // Fix: Claude Desktop connectors don't send the Accept header that
  // StreamableHTTPTransport requires. Build a patched request if missing.
  // See: https://github.com/NateBJones-Projects/OB1/issues/33
  if (!c.req.header("accept")?.includes("text/event-stream")) {
    const headers = new Headers(c.req.raw.headers);
    headers.set("Accept", "application/json, text/event-stream");
    const patched = new Request(c.req.raw.url, {
      method: c.req.raw.method,
      headers,
      body: c.req.raw.body,
      // @ts-ignore -- duplex required for streaming body in Deno
      duplex: "half",
    });
    Object.defineProperty(c.req, "raw", { value: patched, writable: true });
  }

  const server = buildServer();
  const transport = new StreamableHTTPTransport();
  await server.connect(transport);
  const response = await transport.handleRequest(c);
  if (!response) return c.json({ error: "No response from MCP transport" }, 500, corsHeaders);
  response.headers.delete("mcp-session-id");
  for (const [k, v] of Object.entries(corsHeaders)) response.headers.set(k, v);
  return response;
});

Deno.serve(app.fetch);
