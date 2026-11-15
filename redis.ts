const JOB_HASHES_KEY = "job_hashes";

const member = (s: string) =>
  '"' + s.replace(/[\\"'\s]/g, (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase().padStart(2, "0")) + '"';

async function command(...parts: string[]): Promise<any[]> {
  const url = Deno.env.get("UPSTASH_REDIS_REST_URL")!;
  const token = Deno.env.get("UPSTASH_REDIS_REST_TOKEN")!;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: parts.join(" ") }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!res.ok) throw new Error(`Redis ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = await res.json() as { rows?: any[] };
  if (!Array.isArray(body.rows)) throw new Error("Redis: no rows in reply");
  return body.rows;
}

/** Subset of hashes NOT already in the set. Fail-open on error (cleaner should not block writes). */
export async function filterUnseenHashes(hashes: string[]): Promise<string[]> {
  if (hashes.length === 0) return [];
  try {
    const rows = await command("SMISMEMBER", JOB_HASHES_KEY, ...hashes.map(member));
    const flags: number[] = new Array(hashes.length).fill(-1);
    for (const r of rows) flags[Number(r.index)] = Number(r.value);
    if (flags.includes(-1)) throw new Error("Redis smismember returned an invalid result");
    return hashes.filter((_, i) => flags[i] === 0);
  } catch (err) {
    console.error("[redis] smismember failed open:", err);
    return hashes;
  }
}

/** Mark hashes seen after jobs_master write. Log-only on failure — job is already saved. */
export async function markHashesSeen(hashes: string[]): Promise<void> {
  if (hashes.length === 0) return;
  try {
    await command("SADD", JOB_HASHES_KEY, ...hashes.map(member));
  } catch (err) {
    console.error("[redis] sadd error:", err);
  }
}