import { Redis } from "https://esm.sh/@upstash/redis@1.31.6";

export const redis = new Redis({
  url: Deno.env.get("UPSTASH_REDIS_REST_URL")!,
  token: Deno.env.get("UPSTASH_REDIS_REST_TOKEN")!,
});

export const JOB_HASHES_KEY = "job_hashes";

/** Returns the subset of `hashes` NOT already present in the Redis set. */
export async function filterUnseenHashes(hashes: string[]): Promise<string[]> {
  if (hashes.length === 0) return [];
  const BATCH = 500;
  const unseen: string[] = [];
  for (let i = 0; i < hashes.length; i += BATCH) {
    const batch = hashes.slice(i, i + BATCH);
    try {
      const results = await redis.smismember(JOB_HASHES_KEY, ...batch);
      batch.forEach((h, idx) => {
        if (results[idx] === 0) unseen.push(h);
      });
    } catch (err) {
      // Fail-open: on a Redis error, treat the batch as unseen rather than
      // silently dropping real jobs. Worst case is a harmless duplicate
      // upsert (dedup_hash conflict handles that at the DB level anyway).
      console.error(`[redis] smismember batch error, failing open: ${err}`);
      unseen.push(...batch);
    }
  }
  return unseen;
}

/** Marks hashes as seen. Call ONLY once a job is actually saved to jobs_master. */
export async function markHashesSeen(hashes: string[]): Promise<void> {
  if (hashes.length === 0) return;
  try {
    await redis.sadd(JOB_HASHES_KEY, ...hashes);
  } catch (err) {
    console.error(`[redis] sadd error: ${err}`);
  }
}
