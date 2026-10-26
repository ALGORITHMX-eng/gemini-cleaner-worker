// worker.ts - runs on Render (Docker + Deno). No Inngest.
// v3 (LLM answer is final):
// - jobs_pending has NO location / workplace type / employment type. The model reads them from the posting
//   text (including the "Label: value" lines at the top) and whatever it returns is what gets stored.
// - No regex, word list, evidence check or default in code overrides or contradicts the model.
// - The ONLY reason a job is thrown away is no location (or a stale date). A missing workplace type or
//   employment type is simply stored empty (null). Nothing goes to jobs_needs_review any more.
// - temperature 0 and thinking off on every provider.
// - LANES_PER_KEY lanes per key, each lane = claim -> model -> write -> repeat
// - freshness check every 10 min + daily proof
// - every step is logged so you can SEE it working in the Render logs
// Copy ../_shared/redis.ts next to this file (same folder) for markHashesSeen.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { markHashesSeen } from "./redis.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

const USE = (Deno.env.get("LLM") ?? "gemini").toLowerCase(); // "gemini" | "openrouter" | "nvidia"
const MODEL = USE === "nvidia"
  ? (Deno.env.get("NVIDIA_MODEL") ?? "nvidia/nemotron-3.5-lightning-30b-a3b")
  : USE === "openrouter"
  ? (Deno.env.get("OPENROUTER_MODEL") ?? "nvidia/nemotron-3-super-120b-a12b:free")
  : (Deno.env.get("GEMINI_CLEAN_MODEL") ?? "gemini-3.8-flash");
const PROVIDER = `${USE}-clean`; // gemini-clean | openrouter-clean | nvidia-clean
const GEMINI_URL = USE === "nvidia"
  ? "https://integrate.api.nvidia.com/v1/chat/completions"
  : USE === "openrouter"
  ? "https://openrouter.ai/api/v1/chat/completions"
  : "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const KEY_PREFIX = USE === "nvidia" ? "NVIDIA_API_KEY" : USE === "openrouter" ? "OPENROUTER_API_KEY" : "GEMINI_API_KEY";
const BATCH = Number(Deno.env.get("BATCH") ?? 60);
const MIN_PENDING = Number(Deno.env.get("MIN_PENDING") ?? 20); // don't waste a call on tiny batches
const DAILY_CAP = Number(Deno.env.get("DAILY_CAP") ?? (USE === "nvidia" ? 5000 : 1500));   // requests per key per day
const LANES_PER_KEY = Number(Deno.env.get("LANES_PER_KEY") ?? 10);
const IDLE_MS = 15_000;
const GEMINI_TIMEOUT_MS = Number(Deno.env.get("LLM_TIMEOUT_MS") ?? 120_000);
const MAX_CHARS = 17000; // content characters sent per job (the label lines are at the top, so they always fit)
const MAX_BATCH_CHARS = Number(Deno.env.get("MAX_BATCH_CHARS") ?? 200_000); // a call never carries more content than this; extra jobs go back to pending
const EXPIRES_DAYS = 90;
const MAX_AGE_DAYS = Number(Deno.env.get("MAX_AGE_DAYS") ?? 30); // a posting the model dates older than this is dropped as stale. 0 = keep everything
const STORE_RAW_TEXT = false;
const GLOBAL_LABEL = "Worldwide"; // the word the model is told to use when a posting is open to anywhere
const FRESHNESS_TABLE = "job_freshness";
const INFLIGHT = "openrouter_inflight";
const DAY_MS = 86_400_000;

const FRESHNESS_SECRET = Deno.env.get("FRESHNESS_SECRET") || "";
const CF_WORKER_URL = Deno.env.get("CF_WORKER_URL") || "";
const FRESHNESS_EVERY_MS = 10 * 60 * 1000;
const FRESHNESS_BATCHES = 3;
const FRESHNESS_LIMIT = 200;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- keys
function loadKeys(): { idx: number; key: string }[] {
  const seen = new Set<string>();
  const out: { idx: number; key: string }[] = [];
  for (let i = 1; i <= 100; i++) {
    const k = (Deno.env.get(`${KEY_PREFIX}_${i}`) ?? "").trim();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push({ idx: i, key: k });
  }
  // also accept a single plain NVIDIA_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY
  const plain = (Deno.env.get(KEY_PREFIX) ?? "").trim();
  if (plain && !seen.has(plain)) out.unshift({ idx: 1, key: plain });
  console.log(`[init] provider=${USE} model=${MODEL || "(NOT SET)"} url=${GEMINI_URL} keys=${out.length} lanes_per_key=${LANES_PER_KEY} batch=${BATCH} timeout=${GEMINI_TIMEOUT_MS / 1000}s max_age_days=${MAX_AGE_DAYS}`);
  return out;
}
const KEYS = loadKeys();

function nextUtcMidnight(now = Date.now()): number {
  const d = new Date(now);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1, 0, 5, 0);
}

// in-memory pause per key (429 / daily cap)
const pausedUntil = new Map<number, number>();
const missCount = new Map<string, number>(); // job id -> times a good reply came back without it
const MAX_MISSES = Number(Deno.env.get("MAX_MISSES") ?? 3);

// live counters, shown in logs + /health
const state = {
  started_at: new Date().toISOString(),
  lanes_total: 0,
  inflight: 0,
  batches_ok: 0,
  batches_failed: 0,
  jobs_written: 0,
  jobs_dropped: 0,
  jobs_stale: 0,
  last_gemini_ms: 0,
  freshness_last_run: null as string | null,
};

// ---------------------------------------------------------------- prompt
// Built per call so the model always knows today's date (it needs it to read "posted 3 days ago").
function buildInstructions(): string {
  const today = new Date().toISOString().slice(0, 10);
  return `You receive several FULL job postings (already cleaned plain text, no HTML). Classify EACH job separately.
Each posting starts with its Title and URL. The text after that usually begins with a few "Label: value" lines copied from the company's own job board (for example "Location:", "All locations:", "Office:", "Country:", "Workplace type:", "Remote status:", "Employment type:", "Posted:", "Compensation:"), followed by the description. Nobody tells you the location, workplace type, employment type or posting date: you read them from the title, the URL, those label lines and the description. Label lines are explicit statements and you may quote them.
For every job, classify every relevant sentence into exactly one of these sections:
- role_description: what the job IS, day-to-day duties, what the person will do
- requirements: concrete skills, tools, and soft requirements the person MUST have
- nice_to_have: skills, experience, or qualifications explicitly framed as preferred, a bonus, or "nice to have" but not required
- qualifications: years-of-experience thresholds, formal education/certification requirements, and work authorization or sponsorship statements
- compensation: salary figures, pay ranges, hourly rates, equity/stock grants, bonuses, commission structure
- benefits: perks and time off with NO dollar figure attached
- work_setup: ONLY remote/hybrid/onsite arrangement, physical location, work schedule/hours. Begin with "Arrangement: REMOTE" / "Arrangement: HYBRID" / "Arrangement: ONSITE" only when explicitly stated, else "Arrangement: UNKNOWN"
- about_company: factual description of what the company does, its product/industry, founding story, size, funding stage
- company_culture: mission, values, DEI/EEO legal boilerplate
When a bullet mixes a skill with an experience threshold, split it across sections. Discard HTML/template junk. Never invent content not present in source. Preserve original language.
Additionally extract every distinct technical skill, tool, language, framework, or platform mentioned anywhere in the posting into a flat, deduplicated array called "extracted_skills". Normalize casing to how the skill is conventionally written (e.g. "JavaScript" not "javascript", "AWS" not "aws"). Do not include soft skills, years-of-experience phrases, or degree requirements.
Also read the fields below from the WHOLE posting. Your answer is FINAL: it is stored exactly as you give it and nothing in code checks or corrects it. So answer only from what the posting explicitly says, never guess or infer from context. If the posting does not state a field, return null. A null workplace_type or employment_type is simply left empty, and a null location (no place stated at all) means the job is thrown away, so never invent one to save a job. A guessed value is a failure. The work arrangement or employment type is often one sentence far from the top. Today's date is ${today}.
- "workplace_evidence": FIRST find the exact short phrase (under 25 words, copied word-for-word from the posting) that states whether the role is remote, hybrid or onsite. A label line such as "Workplace type: OnSite" or "Remote status: hybrid" is valid evidence; a label whose value is blank, unspecified or not specified is NOT. null if the posting has no such phrase. Fill this BEFORE workplace_type.
- "workplace_type": one of "REMOTE", "HYBRID", "ONSITE", or null. Set it ONLY from workplace_evidence. If workplace_evidence is null this MUST be null. A city/office address alone is NOT a statement of arrangement. If the phrase says "hybrid" (even next to "onsite" or "in office"), the answer is HYBRID. Ignore "remote monitoring", "remote sensing", "hybrid cloud" or "hybrid vehicles". A company-level blurb such as "we are a remote-first company" is NOT evidence about this role unless the posting says this role itself is remote.
- "employment_evidence": the exact short phrase (under 25 words, copied word-for-word) that states the type of employment. A label line such as "Employment type: FullTime" is valid evidence. null if there is none. Fill this BEFORE employment_type.
- "employment_type": one of "FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY", "INTERNSHIP", "VOLUNTEER", or null. Set it ONLY from employment_evidence. Map label values the obvious way: FullTime or Full-time = FULL_TIME, PartTime or Part-time = PART_TIME, Contract, Contractor or Freelance = CONTRACT, Temp or Temporary = TEMPORARY, Intern or Internship = INTERNSHIP. "contract assignment" or "fixed-term contract position" means CONTRACT. "employment contract" or "subject to contract" is NOT evidence. "40 hours per week" about THIS role counts as FULL_TIME. A title containing the whole word "Intern" or "Internship" counts as INTERNSHIP. Do not infer from salary or seniority.
- "location_evidence": the exact short phrase(s) (copied word-for-word) from the "Location", "All locations", "Office" or "Country" label lines, the title or the description that state where the candidate must be based, or that the job is open worldwide. null if nothing states it. Fill this BEFORE location.
- "location": where the candidate must be based, or null. Write the place(s) EXACTLY as the posting states them: a city, a city and state, a state or province, a country, a city and country, a region, or several of these. Whatever the posting states is the answer, even if it is only a city or only a country. NEVER invent, complete or work out any part that is not written: do not add a country to a city or to "Austin, TX", do not expand state codes, and do not get a place from the company name, the URL, the language or the currency. If the posting states both a city and a country (the country may come from a separate "Country:" label line) write "City, Country" or "City, State, Country". Write "${GLOBAL_LABEL}" ONLY when the posting explicitly says anywhere / any country; a global company is NOT worldwide. A mere preference such as "ideally in a US time zone" is NOT a location. null ONLY when the posting states no place at all (for example just "Remote"). When several places are listed keep all of them, separated by " | ". Write country names as full English names (US/USA = United States, UK = United Kingdom). Do NOT include the words remote, hybrid or onsite.
- "posted_date": the date this job was posted as YYYY-MM-DD, from a "Posted:" or "First published:" line or a phrase like "Posted on March 3, 2026". A relative phrase such as "Posted 3 Days Ago" or "30+ Days Ago" counts: work the date out from today's date (for "30+ Days Ago" use 30 days). "Updated" and "Valid through" dates do NOT count. null if the posting gives no posting date.
Copy each job's "id" exactly. Do not skip a job. Do not add jobs.
Respond with ONLY this JSON shape, no markdown:
{"results":[{"id":"...","role_description":"...","requirements":"...","nice_to_have":"...","qualifications":"...","compensation":"...","benefits":"...","work_setup":"...","about_company":"...","company_culture":"...","extracted_skills":["..."],"workplace_evidence":null,"workplace_type":null,"employment_evidence":null,"employment_type":null,"location_evidence":null,"location":null,"posted_date":null}]}`;
}

interface SectionResult {
  id?: string;
  role_description?: string;
  requirements?: string;
  nice_to_have?: string;
  qualifications?: string;
  compensation?: string;
  benefits?: string;
  work_setup?: string;
  about_company?: string;
  company_culture?: string;
  extracted_skills?: string[];
  workplace_evidence?: string | null;
  workplace_type?: string | null;
  employment_evidence?: string | null;
  employment_type?: string | null;
  location_evidence?: string | null;
  location?: string | null;
  posted_date?: string | null;
}

interface Normalized {
  role_description: string;
  requirements: string;
  nice_to_have: string;
  qualifications: string;
  compensation: string;
  benefits: string;
  work_setup: string;
  about_company: string;
  company_culture: string;
  full_description: string;
  extracted_skills: string[];
  workplace_type: string | null;
  employment_type: string | null;
  location: string | null;
  posted_at: string | null; // ISO, from the model's posted_date
}

function stripHtmlArtifacts(text: string): string {
  if (!text) return text;
  return text
    .replace(/<[^>]*>/g, "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#39;/g, "'")
    .replace(/&quot;/gi, '"')
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*\n]+)\*/g, "$1")
    .replace(/^[ \t]*[*+•\u2013\u2014-]\s+/gm, "- ")
    .replace(/\*+/g, "")
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function decodeEntities(t: string | null | undefined): string {
  return String(t ?? "").replace(/&amp;/gi, "&").replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"').replace(/&#39;|&apos;/gi, "'").replace(/&nbsp;/gi, " ").trim();
}

// Only turns the model's reply into the shape the database wants. It never second-guesses the model:
// a value outside the allowed list can't be stored, so it becomes null (left empty).
const VALID_WORKPLACE = new Set(["REMOTE", "HYBRID", "ONSITE"]);
const VALID_EMPTYPE = new Set(["FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]);
const NO_ANSWER = new Set(["", "null", "none", "unknown", "n/a", "na", "not stated", "not specified"]);

function enumOrNull(v: unknown, valid: Set<string>): string | null {
  // "On-site", "on site", "full time", "Full-Time" all map to the allowed value; anything else is null
  const squash = (x: string) => x.toUpperCase().replace(/[\s_-]+/g, "");
  const s = squash(String(v ?? ""));
  for (const allowed of valid) if (squash(allowed) === s) return allowed;
  return null;
}

function textOrNull(v: unknown, max: number): string | null {
  const s = String(v ?? "").replace(/\s+/g, " ").trim();
  return NO_ANSWER.has(s.toLowerCase()) ? null : s.slice(0, max);
}

function isoFromDay(v: unknown): string | null {
  const s = String(v ?? "").trim();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return null;
  const t = Date.parse(s + "T00:00:00Z");
  return Number.isFinite(t) ? new Date(t).toISOString() : null;
}

function normalizeSections(raw: SectionResult): Normalized {
  const role_description = stripHtmlArtifacts(raw.role_description ?? "");
  const requirements = stripHtmlArtifacts(raw.requirements ?? "");
  const nice_to_have = stripHtmlArtifacts(raw.nice_to_have ?? "");
  const qualifications = stripHtmlArtifacts(raw.qualifications ?? "");
  const compensation = stripHtmlArtifacts(raw.compensation ?? "");
  const benefits = stripHtmlArtifacts(raw.benefits ?? "");
  const work_setup = stripHtmlArtifacts(raw.work_setup ?? "");
  const about_company = stripHtmlArtifacts(raw.about_company ?? "");
  const company_culture = stripHtmlArtifacts(raw.company_culture ?? "");
  const full_description = [role_description, requirements, nice_to_have, qualifications, compensation, benefits, work_setup, about_company, company_culture]
    .filter(Boolean).join("\n\n");
  const extracted_skills = Array.isArray(raw.extracted_skills)
    ? [...new Set(raw.extracted_skills.map((s) => String(s).trim()).filter(Boolean))]
    : [];
  return {
    role_description, requirements, nice_to_have, qualifications, compensation, benefits,
    work_setup, about_company, company_culture, full_description, extracted_skills,
    workplace_type: enumOrNull(raw.workplace_type, VALID_WORKPLACE),
    employment_type: enumOrNull(raw.employment_type, VALID_EMPTYPE),
    location: textOrNull(raw.location, 2000),
    posted_at: isoFromDay(raw.posted_date),
  };
}

function isAcceptableResult(sections: Normalized): boolean {
  return sections.full_description.length >= 80;
}

function getJobContent(job: any): string {
  if (job.content) return String(job.content);
  if (Array.isArray(job.chunks) && job.chunks.length > 0) {
    return job.chunks.map((c: any) => c?.text ?? "").filter(Boolean).join("\n\n");
  }
  return "";
}

// only used when the model gave no date but the scraper row already carried one
function scrapedPostedIso(v: unknown): string | null {
  if (v == null || v === "") return null;
  if (typeof v === "string" && v.includes("T")) return v;
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return null;
  const ms = n >= 1e11 ? n : n * 1000;
  try { return new Date(ms).toISOString(); } catch { return null; }
}

// ---------------------------------------------------------------- salary
const CURRENCY_SYMBOLS: Record<string, string> = {
  "$": "USD", "US$": "USD", "USD": "USD", "₦": "NGN", "NGN": "NGN",
  "₵": "GHS", "GH₵": "GHS", "GHS": "GHS", "£": "GBP", "GBP": "GBP",
  "€": "EUR", "EUR": "EUR", "C$": "CAD", "CAD": "CAD", "KSh": "KES", "KES": "KES",
  "R": "ZAR", "ZAR": "ZAR",
};

function extractSalaryFromCompensation(compensationText: string) {
  const text = compensationText || "";
  const CUR = "GH₵|KSh|US\\$|C\\$|\\$|USD|CAD|EUR|GBP|NGN|GHS|KES|ZAR|₦|₵|£|€|R";
  const re = new RegExp(
    `(${CUR})?\\s?(\\d{2,3}(?:,\\d{3})?|\\d+)\\s?([kKmM]?)\\s?(?:-|–|to)\\s?(${CUR})?\\s?(\\d{2,3}(?:,\\d{3})?|\\d+)\\s?([kKmM]?)`,
    "i",
  );
  const m = text.match(re);
  if (!m) return { min: null as number | null, max: null as number | null, currency: null as string | null };
  const rawMin = m[2], rawMax = m[5], suffixMin = m[3], suffixMax = m[6];
  const currencyRaw = m[1] || m[4];
  const currency = currencyRaw ? (CURRENCY_SYMBOLS[currencyRaw] || currencyRaw.toUpperCase()) : null;
  if (!rawMin || !rawMax) return { min: null, max: null, currency };
  const hasCurrencyOrSuffix = Boolean(currencyRaw) || Boolean(suffixMin) || Boolean(suffixMax);
  const looksLikeYear = (s: string) => /^(19|20)\d{2}$/.test(s.replace(/,/g, ""));
  if (!hasCurrencyOrSuffix && looksLikeYear(rawMin) && looksLikeYear(rawMax)) {
    return { min: null, max: null, currency: null };
  }
  const sharedSuffix = Boolean(suffixMin) || Boolean(suffixMax);
  const parseAmt = (s: string, suffix: string) => {
    let n = parseFloat(s.replace(/,/g, ""));
    const suf = (suffix || "").toLowerCase();
    if (suf.includes("k") || (sharedSuffix && suffixMin?.toLowerCase().includes("k"))) n *= 1000;
    if (suf.includes("m") || (sharedSuffix && suffixMin?.toLowerCase().includes("m"))) n *= 1000000;
    return isNaN(n) ? null : n;
  };
  return { min: parseAmt(rawMin, suffixMin), max: parseAmt(rawMax, suffixMax), currency };
}

// ---------------------------------------------------------------- compensation guard
// The model occasionally writes one job's pay into another job's slot. Seen live: a
// German posting (icb-gmbh.jobs.personio.de/job/2447659, no salary anywhere on the
// page) came back carrying "a range of $125,500 - $175,700 ... 7.5 - 15% of base
// salary", copied verbatim from a Credit Acceptance posting that shared its batch.
// The text was real, the employer was wrong. So every money figure in the
// compensation section must exist in THIS job's own source text; if one does not,
// the whole section and the salary columns become empty (which is already the
// accepted answer for "the posting states no pay"). Nothing is rewritten.
function moneyValues(text: string): Set<number> {
  const out = new Set<number>();
  const re = /(?:[$€£₦₵]|US\$|C\$|USD|EUR|GBP|NGN|GHS|KES|ZAR|CAD)?\s?(\d[\d,.]*(?:\d))\s?([kKmM])?/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(String(text || "")))) {
    const raw = m[1];
    // 125.000 (European thousands) vs 125,500 vs 54.5
    const cleaned = /^\d{1,3}(\.\d{3})+$/.test(raw) ? raw.replace(/\./g, "") : raw.replace(/,/g, "");
    let n = parseFloat(cleaned);
    if (!Number.isFinite(n)) continue;
    const suf = (m[2] || "").toLowerCase();
    if (suf === "k") n *= 1000;
    if (suf === "m") n *= 1000000;
    if (n < 1000) continue;                       // percentages, 7.5, 15
    if (n >= 1900 && n <= 2100) continue;         // stray years
    out.add(Math.round(n));
  }
  return out;
}

function trustCompensation(comp: string, source: string): { text: string; missing: number[] } {
  const claimed = moneyValues(comp);
  if (!claimed.size) return { text: comp, missing: [] };   // no figures to verify
  const inSource = moneyValues(source);
  const missing = [...claimed].filter((n) => !inSource.has(n));
  return missing.length ? { text: "", missing } : { text: comp, missing: [] };
}

// ---------------------------------------------------------------- prompt building + parsing
function clip(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n);
}

// Keep a call inside MAX_BATCH_CHARS of content: the jobs that do not fit go back to pending for the next call.
function fitBatch(jobs: any[]): { keep: any[]; extra: any[] } {
  const keep: any[] = [], extra: any[] = [];
  let total = 0;
  for (const j of jobs) {
    const n = Math.min(getJobContent(j).length, MAX_CHARS) + 300;
    if (keep.length && total + n > MAX_BATCH_CHARS) extra.push(j);
    else { keep.push(j); total += n; }
  }
  return { keep, extra };
}

// No Location line here on purpose: nothing supplies one. The label lines inside the body are the source.
function buildUserMessage(jobs: any[]): string {
  return jobs.map((job) => {
    const body = clip(getJobContent(job), MAX_CHARS);
    return [
      `JOB id=${job.id}`,
      `Title: ${job.title || ""}`,
      `URL: ${job.job_url || ""}`,
      "",
      body,
    ].join("\n");
  }).join("\n\n-----\n\n");
}

function replyText(payload: any): string {
  const content = payload?.choices?.[0]?.message?.content ?? "";
  return String(content).replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
}

// when the whole reply is not valid JSON, rescue every job whose own block still parses
function salvageResults(text: string, ids: string[]): SectionResult[] {
  const starts: { at: number }[] = [];
  for (const id of ids) {
    const esc = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const m = new RegExp(`\\{\\s*"id"\\s*:\\s*"${esc}"`).exec(text);
    if (m) starts.push({ at: m.index });
  }
  starts.sort((a, b) => a.at - b.at);
  const out: SectionResult[] = [];
  for (let i = 0; i < starts.length; i++) {
    const seg = text.slice(starts[i].at, i + 1 < starts.length ? starts[i + 1].at : text.length).replace(/[\s,]+$/, "");
    for (const candidate of [seg, seg.replace(/\]\s*\}\s*$/, "").replace(/[\s,]+$/, "")]) {
      try { out.push(JSON.parse(candidate)); break; } catch { /* try the next form */ }
    }
  }
  return out;
}

function parseResults(payload: any): SectionResult[] {
  const content = payload?.choices?.[0]?.message?.content ?? "";
  const text = String(content).replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  const parsed = JSON.parse(text);
  if (Array.isArray(parsed)) return parsed;
  if (Array.isArray(parsed?.results)) return parsed.results;
  if (parsed?.id) return [parsed];
  return [];
}

// ---------------------------------------------------------------- db helpers
async function pendingCount(): Promise<number> {
  const { count, error } = await supabase
    .from("jobs_pending")
    .select("id", { count: "exact", head: true })
    .eq("status", "pending");
  if (error) {
    console.error(`[pending-count] ${error.message}`);
    return 0;
  }
  return count ?? 0;
}

async function callsToday(idx: number): Promise<number> {
  const dayStart = new Date().toISOString().slice(0, 10) + "T00:00:00.000Z";
  const { count, error } = await supabase
    .from("llm_usage_log")
    .select("id", { count: "exact", head: true })
    .eq("provider", PROVIDER)
    .eq("key_index", idx)
    .gte("created_at", dayStart);
  if (error) {
    console.error(`[usage] count failed: ${error.message}`);
    return 0;
  }
  return count ?? 0;
}

async function reserveCall(idx: number): Promise<any> {
  const { data, error } = await supabase
    .from("llm_usage_log")
    .insert({ provider: PROVIDER, key_index: idx, tokens_used: 0, success: false, model: MODEL })
    .select("id")
    .single();
  if (error) {
    console.error(`[reserve] ${error.message}`);
    return null;
  }
  return data.id;
}

async function settleCall(id: any, tokens: number, success: boolean) {
  const { error } = await supabase.from("llm_usage_log").update({ tokens_used: tokens, success }).eq("id", id);
  if (error) console.error(`[settle] ${error.message}`);
}

async function dropCall(id: any) {
  const { error } = await supabase.from("llm_usage_log").delete().eq("id", id);
  if (error) console.error(`[drop] ${error.message}`);
}

async function claimBatch(): Promise<any[]> {
  const { data, error } = await supabase.rpc("claim_jobs", { p_limit: BATCH });
  if (error) throw new Error(error.message);
  const jobs = data ?? [];
  if (jobs.length) {
    await supabase.from("jobs_pending").update({ last_error: INFLIGHT }).in("id", jobs.map((j: any) => j.id));
  }
  return jobs;
}

// jobs the model failed to return cleanly: retry up to MAX_MISSES times, then park them as failed
async function releaseOrGiveUp(ids: string[]) {
  const retry: string[] = [];
  const giveUp: string[] = [];
  for (const id of ids) {
    const n = (missCount.get(id) ?? 0) + 1;
    missCount.set(id, n);
    (n >= MAX_MISSES ? giveUp : retry).push(id);
  }
  if (giveUp.length) {
    await supabase.from("jobs_pending")
      .update({ status: "failed", last_error: `gave_up_after_${MAX_MISSES}_bad_replies` })
      .in("id", giveUp);
    for (const id of giveUp) missCount.delete(id);
    console.warn(`[give-up] parked ${giveUp.length} job(s) as failed: ${giveUp.join(", ")}`);
  }
  await release(retry);
}

async function release(ids: string[]) {
  if (!ids.length) return;
  await supabase.from("jobs_pending").update({ status: "pending", last_error: null }).in("id", ids);
}

// on startup nothing is in flight (as long as Inngest is stopped), so free anything left stuck
async function recoverStuck() {
  const { data, error } = await supabase
    .from("jobs_pending")
    .update({ status: "pending", last_error: null })
    .eq("last_error", INFLIGHT)
    .select("id");
  if (error) console.error(`[recover] ${error.message}`);
  else console.log(`[recover] released ${data?.length ?? 0} stuck jobs`);
}

// ---------------------------------------------------------------- write one job
// Returns: "done" | "dropped" | "stale" | "deferred". Throws on a db error so the lane can count a strike.
async function writeOne(job: any, sections: Normalized): Promise<string> {
  const content = getJobContent(job);
  // the model's answers, stored exactly as given (null = the posting does not say)
  const { workplace_type: workplace, employment_type: employment, location } = sections;
  const postedAt = sections.posted_at ?? scrapedPostedIso(job.posted_at);

  // No location = the job is thrown away. Nothing else makes a job get thrown away except a stale date.
  if (!location) {
    const { error } = await supabase.from("jobs_pending")
      .update({ status: "failed", last_error: "no_location" }).eq("id", job.id);
    if (error) throw new Error(`no_location update failed: ${error.message}`);
    return "dropped";
  }

  if (MAX_AGE_DAYS > 0 && postedAt) {
    const ageDays = (Date.now() - Date.parse(postedAt)) / DAY_MS;
    if (Number.isFinite(ageDays) && ageDays > MAX_AGE_DAYS) {
      const { error } = await supabase.from("jobs_pending")
        .update({ status: "failed", last_error: "stale_posting" }).eq("id", job.id);
      if (error) throw new Error(`stale update failed: ${error.message}`);
      return "stale";
    }
  }

  // a pay figure the posting does not contain is a batch-contamination symptom, not data
  const comp = trustCompensation(sections.compensation, content);
  if (comp.missing.length) {
    console.warn(`[comp-guard] ${job.id} (${job.company || "?"}): compensation dropped, figures not in this posting: ${comp.missing.slice(0, 5).join(", ")}`);
  }
  const salary = extractSalaryFromCompensation(comp.text);
  const expiresIso = new Date(Date.now() + EXPIRES_DAYS * DAY_MS).toISOString();
  if (job.job_url_normalized) {
    const { data: sameUrl } = await supabase.from("jobs_master").select("id, dedup_hash").eq("job_url_normalized", job.job_url_normalized).maybeSingle();
    if (sameUrl && sameUrl.dedup_hash !== job.dedup_hash) {
      await supabase.from("jobs_pending").update({ status: "done", last_error: null }).eq("id", job.id);
      return "done";
    }
  }
  const { error: masterErr } = await supabase.from("jobs_master").upsert({
    id: job.id,
    job_url: job.job_url,
    job_url_normalized: job.job_url_normalized,
    dedup_hash: job.dedup_hash,
    company: job.company || "Unknown",
    role: decodeEntities(job.title),
    location,
    job_type: workplace,
    workplace_type: workplace,
    eligible_countries: job.eligible_countries ?? null,
    eligible_cities: job.eligible_cities ?? null,
    employment_type: employment,
    regions: job.regions ?? null,
    is_global: location.toLowerCase() === GLOBAL_LABEL.toLowerCase() ? true : (job.is_global ?? null),
    requires_us_citizenship: job.requires_us_citizenship ?? null,
    role_description: sections.role_description,
    requirements: sections.requirements,
    nice_to_have: sections.nice_to_have,
    qualifications: sections.qualifications,
    compensation: comp.text,
    benefits: sections.benefits,
    work_setup: sections.work_setup,
    about_company: sections.about_company,
    company_culture: sections.company_culture,
    extracted_skills: sections.extracted_skills,
    posted_at: postedAt,
    salary_min: salary.min,
    salary_max: salary.max,
    salary_currency: salary.currency,
    raw_text: STORE_RAW_TEXT ? content : "",
    source: job.source,
    pipe: job.pipe || "pipe1",
    tag_source: "llm-labeled-gemini",
    is_active: true,
    tagged_at: new Date().toISOString(),
    scraped_at: new Date().toISOString(),
    expires_at: expiresIso,
  }, { onConflict: "dedup_hash" });
  if (masterErr) {
    // throw (not silently retry forever): the lane counts a strike and the real reason shows in the logs
    throw new Error(`jobs_master upsert failed: ${masterErr.message}`);
  }
  const { error: freshErr } = await supabase.from(FRESHNESS_TABLE).upsert({
    job_id: job.id,
    job_url: job.job_url,
    dedup_hash: job.dedup_hash,
    source: job.source,
    expires_at: expiresIso,
  }, { onConflict: "job_id" });
  if (freshErr) {
    await supabase.from("jobs_pending").update({ status: "pending", last_error: null }).eq("id", job.id);
    return "deferred";
  }
  await supabase.from("jobs_pending").update({ status: "done", last_error: null }).eq("id", job.id);
  if (job.dedup_hash) {
    try { await markHashesSeen([job.dedup_hash]); } catch (err) { console.error(`[redis] ${err}`); }
  }
  return "done";
}

// ---------------------------------------------------------------- gemini call (direct, no Inngest)
class GeminiError extends Error {
  constructor(public status: number, public body: string) {
    super(`gemini HTTP ${status}: ${body}`);
  }
}

const MAX_TOKENS = Number(Deno.env.get("MAX_TOKENS") ?? (USE === "gemini" ? 65000 : USE === "openrouter" ? 32000 : 16384));
// thinking is OFF and temperature is 0 on every provider
const GEMINI_REASONING = Deno.env.get("GEMINI_REASONING_EFFORT") ?? "none"; // if Gemini answers HTTP 400 to "none", set this env to "minimal" or "low"

function buildBody(messages: unknown[]): Record<string, unknown> {
  if (USE === "nvidia") {
    return { model: MODEL, messages, max_tokens: MAX_TOKENS, temperature: 0, chat_template_kwargs: { enable_thinking: false } };
  }
  if (USE === "openrouter") {
    return { model: MODEL, messages, max_tokens: MAX_TOKENS, temperature: 0, response_format: { type: "json_object" }, reasoning: { enabled: false } };
  }
  return { model: MODEL, messages, max_tokens: MAX_TOKENS, temperature: 0, response_format: { type: "json_object" }, reasoning_effort: GEMINI_REASONING };
}

async function callGemini(apiKey: string, messages: unknown[]): Promise<any> {
  const res = await fetch(GEMINI_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(buildBody(messages)),
    signal: AbortSignal.timeout(GEMINI_TIMEOUT_MS),
  });
  const text = await res.text();
  if (!res.ok) throw new GeminiError(res.status, text.slice(0, 300));
  const json = JSON.parse(text);
  // some providers answer HTTP 200 with an error body (e.g. upstream timeout)
  if (json?.error) throw new GeminiError(Number(json.error.code) || 502, JSON.stringify(json.error).slice(0, 300));
  if (json?.choices?.[0]?.finish_reason === "length") {
    throw new GeminiError(413, "output hit max_tokens (finish_reason=length): lower BATCH or raise MAX_TOKENS");
  }
  return json;
}

// ---------------------------------------------------------------- one lane = one worker loop
async function lane(keyIdx: number, laneNo: number, apiKey: string) {
  const tag = `k${keyIdx}-l${laneNo}`;
  await sleep(laneNo * 700); // stagger so 10 lanes don't hit the db at the same millisecond

  while (true) {
    try {
      if ((pausedUntil.get(keyIdx) ?? 0) > Date.now()) { await sleep(5000); continue; }

      if ((await pendingCount()) < MIN_PENDING) { await sleep(IDLE_MS); continue; }

      const claimed = await claimBatch();
      if (!claimed.length) { await sleep(IDLE_MS); continue; }
      const { keep: jobs, extra } = fitBatch(claimed);
      if (extra.length) await release(extra.map((j: any) => j.id)); // did not fit this call, next call picks them up

      const slotId = await reserveCall(keyIdx);
      if (slotId == null) { await release(jobs.map((j: any) => j.id)); await sleep(5000); continue; }
      const used = await callsToday(keyIdx);
      if (used > DAILY_CAP) {
        await dropCall(slotId);
        await release(jobs.map((j: any) => j.id));
        pausedUntil.set(keyIdx, nextUtcMidnight());
        console.warn(`[${tag}] key ${keyIdx} hit daily cap (${used}), paused until UTC midnight`);
        continue;
      }

      const ids = jobs.map((j: any) => j.id);
      const messages = [
        { role: "system", content: buildInstructions() },
        { role: "user", content: buildUserMessage(jobs) },
      ];

      console.log(`[${tag}] claimed ${jobs.length} jobs -> calling ${USE} (call ${used}/${DAILY_CAP})`);
      state.inflight++;
      const t0 = Date.now();
      let payload: any;
      try {
        payload = await callGemini(apiKey, messages);
      } catch (err) {
        state.inflight--;
        state.batches_failed++;
        await release(ids);
        if (err instanceof GeminiError && (err.status === 401 || err.status === 403)) {
          await dropCall(slotId);
          pausedUntil.set(keyIdx, Date.now() + 60 * 60_000);
          console.error(`[${tag}] key ${keyIdx} rejected (${err.status}), paused 1h: ${err.body.slice(0, 120)}`);
        } else if (err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError")) {
          await dropCall(slotId); // model too slow, don't burn a daily slot
          console.warn(`[${tag}] timed out after ${GEMINI_TIMEOUT_MS / 1000}s, this lane retries in 10s (other lanes keep going)`);
          await sleep(10_000);
        } else if (err instanceof GeminiError && err.status === 413) {
          await dropCall(slotId);
          console.error(`[${tag}] ${err.body}`);
          await sleep(5000);
        } else if (err instanceof GeminiError && [400, 402, 404].includes(err.status)) {
          await dropCall(slotId); // config/credit problem, retrying won't help
          pausedUntil.set(keyIdx, Date.now() + 10 * 60_000);
          console.error(`[${tag}] HTTP ${err.status}, key ${keyIdx} paused 10min: ${err.body.slice(0, 250)}`);
        } else if (err instanceof GeminiError && [500, 502, 503, 504].includes(err.status)) {
          await dropCall(slotId); // provider overloaded, don't burn a daily slot
          const wait = 15_000 + Math.floor(Math.random() * 30_000);
          console.warn(`[${tag}] ${err.status} overloaded, this lane retries in ${Math.round(wait / 1000)}s`);
          await sleep(wait);
        } else if (err instanceof GeminiError && err.status === 429) {
          await dropCall(slotId); // rate limited, don't burn a daily slot
          const daily = /per[\s-]*day|daily/i.test(err.body);
          const until = daily ? nextUtcMidnight() : Date.now() + 60_000;
          pausedUntil.set(keyIdx, until);
          console.warn(`[${tag}] 429 from Gemini, key ${keyIdx} paused until ${new Date(until).toISOString()}`);
        } else {
          await settleCall(slotId, 0, false);
          console.error(`[${tag}] gemini failed: ${String(err).slice(0, 300)}`);
          await sleep(5000);
        }
        continue;
      }
      state.inflight--;
      state.last_gemini_ms = Date.now() - t0;
      console.log(`[${tag}] ${USE} ok in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

      // parse + write
      let results: SectionResult[] = [];
      try {
        results = parseResults(payload);
      } catch (e) {
        const msg = String(e);
        const pos = Number(/position (\d+)/.exec(msg)?.[1] ?? -1);
        const text = replyText(payload);
        console.error(`[${tag}] bad json: ${msg.slice(0, 160)} | near: ${pos >= 0 ? JSON.stringify(text.slice(Math.max(0, pos - 90), pos + 60)) : "?"}`);
        results = salvageResults(text, ids);
        console.warn(`[${tag}] salvaged ${results.length}/${ids.length} jobs from a broken reply`);
        if (!results.length) {
          await release(ids); // nothing usable: plain retry, no strike against the jobs
          await settleCall(slotId, 0, false);
          state.batches_failed++;
          continue;
        }
      }

      const byId = new Map(results.map((r) => [String(r.id || ""), r]));
      let done = 0, dropped = 0, stale = 0;
      const missing: string[] = [];
      for (const job of jobs) {
        const raw = byId.get(String(job.id));
        if (!raw) { missing.push(job.id); continue; }
        const sections = normalizeSections(raw);
        if (!isAcceptableResult(sections)) { missing.push(job.id); continue; }
        try {
          const outcome = await writeOne(job, sections);
          if (outcome === "done") done++;
          else if (outcome === "dropped") dropped++;
          else if (outcome === "stale") stale++;
        } catch (e) {
          console.error(`[${tag}] write error for ${job.id}: ${String(e).slice(0, 200)}`);
          missing.push(job.id);
        }
      }
      const missingSet = new Set(missing);
      for (const j of jobs) if (!missingSet.has(j.id)) missCount.delete(j.id);
      if (results.length) await releaseOrGiveUp(missing); else await release(missing);
      const tokens = (payload?.usage?.prompt_tokens || 0) + (payload?.usage?.completion_tokens || 0);
      await settleCall(slotId, tokens, true);
      state.batches_ok++;
      state.jobs_written += done;
      state.jobs_dropped += dropped;
      state.jobs_stale += stale;
      console.log(`[${tag}] of ${jobs.length}: done=${done} no_location=${dropped} stale=${stale} released=${missing.length}, ${tokens} tokens`);
    } catch (e) {
      console.error(`[${tag}] lane error: ${String(e).slice(0, 300)}`);
      await sleep(5000);
    }
  }
}

// ---------------------------------------------------------------- freshness (was Inngest cron)
async function callFreshness(body: Record<string, unknown>): Promise<any> {
  if (!FRESHNESS_SECRET) throw new Error("FRESHNESS_SECRET is not set");
  const res = await fetch(`${SUPABASE_URL}/functions/v1/check-job-freshness`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      "x-internal-secret": FRESHNESS_SECRET,
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(120_000),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`check-job-freshness HTTP ${res.status}: ${text.slice(0, 300)}`);
  try { return JSON.parse(text); } catch { throw new Error(`check-job-freshness bad JSON: ${text.slice(0, 200)}`); }
}

let freshnessBusy = false;
async function freshnessRun() {
  if (freshnessBusy || !FRESHNESS_SECRET) return;
  freshnessBusy = true;
  try {
    const t = { batches_run: 0, checked: 0, open: 0, unknown: 0, dead: 0, first_strike: 0, expired: 0, sent_dead: 0, sent_expired: 0 };
    let first_reasons: unknown = null;
    for (let i = 0; i < FRESHNESS_BATCHES; i++) {
      const out: any = await callFreshness({ limit: FRESHNESS_LIMIT, dry_run: false });
      if (out.suspicious) {
        console.error(`[freshness] suspicious batch: ${out.closed_total} of ${out.checked} looked closed, nothing applied. ${JSON.stringify(out.by_reason).slice(0, 400)}`);
        return;
      }
      if (out.write_errors?.length) { console.error(`[freshness] write errors: ${JSON.stringify(out.write_errors)}`); return; }
      const notifyErr = out.notify?.error ?? out.notify_expired?.error;
      if (notifyErr) { console.error(`[freshness] Cloudflare notify failed: ${notifyErr}`); return; }
      if (typeof out.notify === "string") { console.error(`[freshness] notify ${out.notify}`); return; }
      t.batches_run++;
      t.checked += out.checked ?? 0;
      t.open += out.open ?? 0;
      t.unknown += out.unknown ?? 0;
      t.dead += out.dead ?? 0;
      t.first_strike += out.first_strike ?? 0;
      t.expired += typeof out.expired === "number" ? out.expired : 0;
      t.sent_dead += out.notify?.sent ?? 0;
      t.sent_expired += out.notify_expired?.sent ?? 0;
      if (i === 0) first_reasons = out.by_reason;
      if (!out.checked || out.done) break;
    }
    state.freshness_last_run = new Date().toISOString();
    console.log("[freshness-run]", JSON.stringify({ ...t, first_reasons }));
  } catch (e) {
    console.error(`[freshness] ${String(e).slice(0, 300)}`);
  } finally {
    freshnessBusy = false;
  }
}

async function freshnessProof() {
  try {
    const since = new Date(Date.now() - 24 * 3600 * 1000).toISOString();
    const since14 = new Date(Date.now() - 14 * 24 * 3600 * 1000).toISOString();
    const count = async (f: (q: any) => any): Promise<number> => {
      const { count, error } = await f(supabase.from("job_freshness").select("job_id", { count: "exact", head: true }));
      if (error) throw new Error(error.message);
      return count ?? 0;
    };
    const stats = {
      active: await count((q) => q.eq("is_active", true)),
      checked_24h: await count((q) => q.gt("last_checked_at", since)),
      dead_unsent: await count((q) => q.eq("is_active", false).is("cf_notified_at", null).not("dead_at", "is", null)),
      expired_unsent: await count((q) => q.eq("is_active", false).is("cf_notified_at", null).is("dead_at", null)),
      dead_sent_14d: await count((q) =>
        q.eq("is_active", false).not("cf_notified_at", "is", null).not("dead_at", "is", null).gt("dead_at", since14)
      ),
    };
    let on_board = 0;
    if (CF_WORKER_URL) {
      const res = await fetch(`${CF_WORKER_URL.replace(/\/$/, "")}/api/expired-jobs`);
      if (!res.ok) throw new Error(`/api/expired-jobs HTTP ${res.status}`);
      const d: any = await res.json();
      on_board = Array.isArray(d?.removedIds) ? d.removedIds.length : 0;
    }
    const problems: string[] = [];
    if (stats.active > 0 && stats.checked_24h === 0) problems.push("no job was checked in the last 24h: freshness is NOT running");
    if (stats.dead_unsent > 50) problems.push(`${stats.dead_unsent} dead jobs never reached Cloudflare`);
    if (stats.expired_unsent > 50) problems.push(`${stats.expired_unsent} expired jobs never reached Cloudflare`);
    if (stats.dead_sent_14d > 0 && on_board === 0) problems.push(`Supabase says ${stats.dead_sent_14d} dead jobs were sent, but the Cloudflare notice board is empty`);
    if (problems.length) console.error(`[freshness-proof] PROBLEMS: ${problems.join(" | ")}`);
    else console.log("[freshness-proof] ok", JSON.stringify({ ...stats, on_board }));
  } catch (e) {
    console.error(`[freshness-proof] ${String(e).slice(0, 300)}`);
  }
}

// ---------------------------------------------------------------- start everything
await recoverStuck();

Deno.serve({ port: Number(Deno.env.get("PORT") ?? 8000) }, async (req) => {
  const url = new URL(req.url);
  if (url.pathname === "/wake") return new Response("awake"); // scraper pings this to wake a sleeping free instance
  const pending = await pendingCount();
  return new Response(JSON.stringify({ ...state, pending, paused_keys: [...pausedUntil.keys()] }, null, 2), {
    headers: { "Content-Type": "application/json" },
  });
});

if (!KEYS.length) { console.error(`[init] NO KEYS FOUND for LLM=${USE}: set ${KEY_PREFIX}_1`); Deno.exit(1); }
for (const k of KEYS) {
  for (let l = 1; l <= LANES_PER_KEY; l++) {
    state.lanes_total++;
    lane(k.idx, l, k.key); // not awaited: every lane runs forever in parallel
  }
}

setInterval(freshnessRun, FRESHNESS_EVERY_MS);
freshnessRun();

let lastProofDay = "";
setInterval(() => {
  const now = new Date();
  const day = now.toISOString().slice(0, 10);
  if (now.getUTCHours() === 7 && lastProofDay !== day) {
    lastProofDay = day;
    freshnessProof();
  }
}, 60_000);

// heartbeat so you always see it alive in the Render logs
setInterval(async () => {
  const pending = await pendingCount();
  console.log(`[status] pending=${pending} inflight=${state.inflight}/${state.lanes_total} ok=${state.batches_ok} failed=${state.batches_failed} written=${state.jobs_written} no_location=${state.jobs_dropped} stale=${state.jobs_stale} lastGemini=${(state.last_gemini_ms / 1000).toFixed(1)}s`);
}, 30_000);

console.log(`[init] worker up: ${state.lanes_total} lanes`);