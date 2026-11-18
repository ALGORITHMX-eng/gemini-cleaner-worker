// redis.ts — Aiven Valkey over RESP/TLS (replaces the Layerbase HTTP query gateway).
//
// Same two exports as before, same signatures, so worker.ts needs no call-site changes:
//   markHashesSeen(hashes)        -> SET jh:<hash> 1 EX 90d
//   filterUnseenHashes(hashes)    -> MGET jh:<hash>...
//
// Two differences from the old file that matter:
//   1. Members are stored RAW. Layerbase needed percent-escaping inside a text command;
//      RESP is length-prefixed and binary safe, so escaping would store values that no
//      longer match what the scout queries.
//   2. The connection is persistent. This service marks one hash per finished job, so a
//      fresh TLS handshake each time would add hundreds of milliseconds per job. One
//      socket is opened on first use, reconnected after any error, and shared by all lanes.
//
// Aiven Valkey serves a publicly-trusted certificate chain, so no CA file is needed here
// (unlike Postgres, which uses Aiven's self-signed project CA).

const PREFIX = "jh:";
const TTL_SECONDS = 90 * 86_400;
const TIMEOUT_MS = 10_000;

type Reply = string | number | null | Reply[];

const VALKEY_URL = Deno.env.get("VALKEY_URL") ?? "";

// ---------------------------------------------------------------- wire format
function encode(args: (string | number)[]): Uint8Array {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [enc.encode(`*${args.length}\r\n`)];
  for (const a of args) {
    const b = enc.encode(String(a));
    parts.push(enc.encode(`$${b.length}\r\n`), b, enc.encode("\r\n"));
  }
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

function parseOne(b: Uint8Array): { value: Reply; offset: number } | null {
  if (b.length < 3) return null;
  let nl = -1;
  for (let i = 0; i + 1 < b.length; i++) if (b[i] === 13 && b[i + 1] === 10) { nl = i; break; }
  if (nl === -1) return null;
  const type = b[0];
  const line = new TextDecoder().decode(b.subarray(1, nl));

  if (type === 43) return { value: line, offset: nl + 2 };                                  // +
  if (type === 45) return { value: { error: line } as unknown as Reply, offset: nl + 2 };   // -
  if (type === 58) return { value: Number(line), offset: nl + 2 };                          // :
  if (type === 36) {                                                                         // $
    const n = Number(line);
    if (n === -1) return { value: null, offset: nl + 2 };
    const end = nl + 2 + n + 2;
    if (b.length < end) return null;
    return { value: new TextDecoder().decode(b.subarray(nl + 2, nl + 2 + n)), offset: end };
  }
  if (type === 42) {                                                                         // *
    const n = Number(line);
    if (n === -1) return { value: null, offset: nl + 2 };
    let off = nl + 2;
    const arr: Reply[] = [];
    for (let i = 0; i < n; i++) {
      const r = parseOne(b.subarray(off));
      if (!r) return null;
      arr.push(r.value);
      off += r.offset;
    }
    return { value: arr, offset: off };
  }
  throw new Error(`RESP: unexpected type byte ${type}`);
}

// ---------------------------------------------------------------- connection
let sock: Deno.TlsConn | null = null;
let buf = new Uint8Array(0);
let waiters: { resolve: (v: Reply) => void; reject: (e: Error) => void }[] = [];
let connecting: Promise<void> | null = null;

function fail(err: Error): void {
  const s = sock;
  sock = null;
  buf = new Uint8Array(0);
  const pending = waiters;
  waiters = [];
  if (s) { try { s.close(); } catch { /* already gone */ } }
  for (const w of pending) w.reject(err);
}

function drain(): void {
  while (waiters.length) {
    let r: { value: Reply; offset: number } | null;
    try { r = parseOne(buf); } catch (e) { fail(e as Error); return; }
    if (!r) return;
    buf = buf.subarray(r.offset);
    const w = waiters.shift()!;
    const v = r.value as any;
    if (v && typeof v === "object" && !Array.isArray(v) && "error" in v) w.reject(new Error(`valkey: ${v.error}`));
    else w.resolve(r.value);
  }
}

async function readLoop(conn: Deno.TlsConn): Promise<void> {
  const chunk = new Uint8Array(32 * 1024);
  try {
    while (true) {
      const n = await conn.read(chunk);
      if (n === null) break;
      const next = new Uint8Array(buf.length + n);
      next.set(buf);
      next.set(chunk.subarray(0, n), buf.length);
      buf = next;
      drain();
    }
    fail(new Error("valkey: connection closed"));
  } catch (e) {
    fail(e instanceof Error ? e : new Error(String(e)));
  }
}

async function connectOnce(): Promise<void> {
  if (!VALKEY_URL) throw new Error("VALKEY_URL is not set");
  const u = new URL(VALKEY_URL);
  if (u.protocol !== "rediss:") throw new Error("VALKEY_URL must start with rediss://");
  const conn = await Deno.connectTls({ hostname: u.hostname, port: Number(u.port || 6379) });
  sock = conn;
  buf = new Uint8Array(0);
  waiters = [];
  void readLoop(conn);
  const auth = await cmd("AUTH", decodeURIComponent(u.username || "default"), decodeURIComponent(u.password));
  if (auth !== "OK") throw new Error(`valkey: AUTH answered ${String(auth).slice(0, 40)}`);
}

async function ensure(): Promise<void> {
  if (sock) return;
  if (!connecting) connecting = connectOnce().finally(() => { connecting = null; });
  await connecting;
}

function cmd(...args: (string | number)[]): Promise<Reply> {
  return ensure().then(() => new Promise<Reply>((resolve, reject) => {
    let settled = false;
    const w = {
      resolve: (v: Reply) => { if (!settled) { settled = true; clearTimeout(timer); resolve(v); } },
      reject: (e: Error) => { if (!settled) { settled = true; clearTimeout(timer); reject(e); } },
    };
    const timer = setTimeout(() => {
      // A silent socket would wedge this lane forever, and a late reply would arrive out of
      // order. Drop the connection: the next command reconnects.
      w.reject(new Error(`valkey: timed out after ${TIMEOUT_MS}ms`));
      fail(new Error("valkey: timeout"));
    }, TIMEOUT_MS);
    waiters.push(w);
    const s = sock;
    if (!s) { w.reject(new Error("valkey: no connection")); return; }
    s.write(encode(args)).catch((e) => {
      w.reject(e instanceof Error ? e : new Error(String(e)));
      fail(e instanceof Error ? e : new Error(String(e)));
    });
  }));
}

// ---------------------------------------------------------------- public API
export async function filterUnseenHashes(hashes: string[]): Promise<string[]> {
  if (!hashes.length) return [];
  try {
    const uniq = [...new Set(hashes)];
    const seen = new Set<string>();
    const CHUNK = 500;
    for (let i = 0; i < uniq.length; i += CHUNK) {
      const chunk = uniq.slice(i, i + CHUNK);
      const reply = await cmd("MGET", ...chunk.map((h) => PREFIX + h));
      if (!Array.isArray(reply)) throw new Error("MGET returned a non-array");
      for (let k = 0; k < chunk.length; k++) if (reply[k] !== null) seen.add(chunk[k]);
    }
    return hashes.filter((h) => !seen.has(h));
  } catch (err) {
    // same contract as before: fail OPEN here, the caller decides about duplicates
    console.error("[redis] mget failed open:", String(err).slice(0, 200));
    return hashes;
  }
}

export async function markHashesSeen(hashes: string[]): Promise<void> {
  if (!hashes.length) return;
  try {
    const uniq = [...new Set(hashes)];
    for (let i = 0; i < uniq.length; i += 500) {
      const chunk = uniq.slice(i, i + 500);
      await Promise.all(chunk.map((h) => cmd("SET", PREFIX + h, "1", "EX", TTL_SECONDS)));
    }
    console.log(`[redis] sadd ok n=${uniq.length}`);
  } catch (err) {
    console.error("[redis] sadd error:", String(err).slice(0, 200));
  }
}
