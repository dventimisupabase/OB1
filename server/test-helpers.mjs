/**
 * test-helpers.mjs
 *
 * Unit tests for the pure pagination/clamp/format helpers used by the MCP
 * tools. These run with no infra (no Supabase, no Deno).
 *
 * NOTE: the three helpers below are a verbatim mirror of the ones in
 * server/index.ts (Deno can't be imported into Node because index.ts opens a
 * server at import time). Keep them in sync — this mirrors the existing
 * re-implementation pattern in test-stateless.mjs.
 *
 *   node test-helpers.mjs   # or: npm test
 */

import assert from "node:assert/strict";

// ── Mirror of server/index.ts helpers ─────────────────────────────────────────

function clampLimit(n, def, max) {
  const v = typeof n === "number" && Number.isFinite(n) ? Math.floor(n) : def;
  return Math.max(1, Math.min(v, max));
}

function safeOffset(n) {
  const v = typeof n === "number" && Number.isFinite(n) ? Math.floor(n) : 0;
  return Math.max(0, v);
}

function pageFooter(offset, returned, total) {
  if (total == null) return "";
  const first = total === 0 ? 0 : offset + 1;
  const last = offset + returned;
  let footer = `\nShowing ${first}–${last} of ${total}.`;
  if (last < total) footer += ` Pass offset=${last} for the next page.`;
  return footer;
}

// ── Tests ──────────────────────────────────────────────────────────────────────

let passed = 0;
function check(label, fn) {
  fn();
  console.log(`  ✓  ${label}`);
  passed++;
}

console.log("\n[clampLimit]");
check("undefined → default", () => assert.equal(clampLimit(undefined, 25, 200), 25));
check("within range passes through", () => assert.equal(clampLimit(5, 25, 200), 5));
check("over max clamps to max", () => assert.equal(clampLimit(500, 25, 200), 200));
check("zero clamps up to 1", () => assert.equal(clampLimit(0, 25, 200), 1));
check("negative clamps up to 1", () => assert.equal(clampLimit(-3, 25, 200), 1));
check("floors fractional", () => assert.equal(clampLimit(3.9, 25, 200), 3));

console.log("\n[safeOffset]");
check("undefined → 0", () => assert.equal(safeOffset(undefined), 0));
check("negative → 0", () => assert.equal(safeOffset(-5), 0));
check("floors fractional", () => assert.equal(safeOffset(7.8), 7));

console.log("\n[pageFooter]");
check("first page has next-page hint", () =>
  assert.equal(pageFooter(0, 10, 25), "\nShowing 1–10 of 25. Pass offset=10 for the next page."));
check("middle page advances range + offset", () =>
  assert.equal(pageFooter(10, 10, 25), "\nShowing 11–20 of 25. Pass offset=20 for the next page."));
check("last page has no next-page hint", () =>
  assert.equal(pageFooter(20, 5, 25), "\nShowing 21–25 of 25."));
check("empty result set", () =>
  assert.equal(pageFooter(0, 0, 0), "\nShowing 0–0 of 0."));
check("null total (fallback, no count) → empty footer", () =>
  assert.equal(pageFooter(0, 5, null), ""));

console.log(`\n${"─".repeat(50)}`);
console.log(`${passed} assertions passed`);
console.log("PASS\n");
