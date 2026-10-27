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
// The stale pre-check throws most of a claim away before the model sees it, so
// claiming only BATCH leaves 1-2 jobs per paid call. Claim more, keep BATCH.
const CLAIM_FACTOR = Math.max(1, Number(Deno.env.get("CLAIM_FACTOR") ?? 3));
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
  stale_precheck: 0,
  jobs_unresolved: 0,
  loc_reasoned: 0,
  loc_resolved: 0,
  loc_none: 0,
  loc_unavailable: 0,
  loc_unusable: 0,
  loc_skipped_cap: 0,
  loc_circuit: 0,
  global_unverified: 0,
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
Also read the fields below from the WHOLE posting. Your answer is FINAL: it is stored exactly as you give it and nothing in code checks or corrects it. So answer only from what the posting says; a guessed workplace_type or employment_type is a failure. ONE exception: location MAY be resolved — from the posting, from the company's own office name, and from the job URL — see the location rule below. If the posting does not state a field, return null. A null workplace_type or employment_type is simply left empty, and a null location (nothing points to a place at all) means the job is thrown away, so never invent a place just to save a job. The work arrangement or employment type is often one sentence far from the top. Today's date is ${today}.
- "workplace_evidence": FIRST find the exact short phrase (under 25 words, copied word-for-word from the posting) that states whether the role is remote, hybrid or onsite. A label line such as "Workplace type: OnSite" or "Remote status: hybrid" is valid evidence; a label whose value is blank, unspecified or not specified is NOT. null if the posting has no such phrase. Fill this BEFORE workplace_type.
- "workplace_type": one of "REMOTE", "HYBRID", "ONSITE", or null. Set it ONLY from workplace_evidence. If workplace_evidence is null this MUST be null. A city/office address alone is NOT a statement of arrangement. If the phrase says "hybrid" (even next to "onsite" or "in office"), the answer is HYBRID. Ignore "remote monitoring", "remote sensing", "hybrid cloud" or "hybrid vehicles". A company-level blurb such as "we are a remote-first company" is NOT evidence about this role unless the posting says this role itself is remote.
- "employment_evidence": the exact short phrase (under 25 words, copied word-for-word) that states the type of employment. A label line such as "Employment type: FullTime" is valid evidence. null if there is none. Fill this BEFORE employment_type.
- "employment_type": one of "FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY", "INTERNSHIP", "VOLUNTEER", or null. Set it ONLY from employment_evidence. Map label values the obvious way: FullTime or Full-time = FULL_TIME, PartTime or Part-time = PART_TIME, Contract, Contractor or Freelance = CONTRACT, Temp or Temporary = TEMPORARY, Intern or Internship = INTERNSHIP. "contract assignment" or "fixed-term contract position" means CONTRACT. "employment contract" or "subject to contract" is NOT evidence. "40 hours per week" about THIS role counts as FULL_TIME. A title containing the whole word "Intern" or "Internship" counts as INTERNSHIP. Do not infer from salary or seniority.
- "location_evidence": the exact short phrase(s) (copied word-for-word) from the "Location", "All locations", "Office" or "Country" label lines, the title, the description or the job URL that state where the candidate must be based, or that the job is open worldwide. If the place comes from an office name in the text or from a place written into the URL, quote that. null if nothing points to a place. Fill this BEFORE location.
- "location": where the candidate must be based, or null. Start from whatever the posting states, then RESOLVE it to a usable place. You MUST name the country for a city, town, state or office you recognise: "Dresden" -> "Dresden, Germany"; "Berlin" -> "Berlin, Germany"; "Bangalore HQ" -> "Bangalore, India"; "Chennai Regional Office" -> "Chennai, India"; "Atlanta, Georgia" -> "Atlanta, Georgia, United States"; "Austin, TX" -> "Austin, TX, United States"; "Palo Alto" -> "Palo Alto, California, United States"; "North Kingstown, RI" -> "North Kingstown, Rhode Island, United States". Use everything available for this: the label lines, the description, the company's own office name and the job URL (Workday links often carry the place, e.g. ".../Malaysia--Kuala-Lumpur-SSC/..."). If the posting gives only a country, keep the country and do not invent a city for it. Write "City, Country" or "City, State, Country". Write "${GLOBAL_LABEL}" ONLY when the posting explicitly says anywhere / any country; a global company is NOT worldwide. A mere preference such as "ideally in a US time zone" is NOT a location. null ONLY when nothing in the posting, the office name or the URL points to a place (for example "Remote" with no country anywhere); "Remote" plus a country is just the country. When several places are listed keep all of them, separated by " | ". Write country names as full English names (US/USA = United States, UK = United Kingdom). Do NOT include the words remote, hybrid or onsite.
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
    // decimals matter: "$110,000.00 - $150,000.00" and "$54.5K - $68K" both used to
    // parse as 0 / 5,000 because the old pattern stopped at the dot and backtracked.
    `(${CUR})?\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s?([kKmM]?)\\s?(?:-|–|to)\\s?(${CUR})?\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s?([kKmM]?)`,
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
    // a 0 here is always a parse artefact, never a real salary
    return !Number.isFinite(n) || n <= 0 ? null : Math.round(n);
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
    if (keep.length >= BATCH) extra.push(j); // count cap: the claim is deliberately oversized
    else if (keep.length && total + n > MAX_BATCH_CHARS) extra.push(j);
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
  const { data, error } = await supabase.rpc("claim_jobs", { p_limit: BATCH * CLAIM_FACTOR });
  if (error) throw new Error(error.message);
  const jobs = data ?? [];
  if (jobs.length) {
    const ids = jobs.map((j: any) => j.id);
    const { error: markErr } = await supabase.from("jobs_pending").update({ last_error: INFLIGHT }).in("id", ids);
    if (markErr) {
      // claim_jobs already set status=processing. Without the marker these rows are
      // invisible to recoverStuck and would sit there forever, so put them straight
      // back rather than leave a hole.
      await release(ids);
      throw new Error(`inflight mark failed: ${markErr.message}`);
    }
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

// On startup nothing is in flight (as long as Inngest is stopped), so free anything left stuck.
// Two sweeps, because claim_jobs writes status=processing before the marker lands:
//   (a) the normal case — the marker is there, free it whatever its age
//   (b) the hole — processing with no marker at all, older than any real call can
//       take (LLM_TIMEOUT_MS is 120s, so 15 minutes is never a live call)
async function recoverStuck() {
  const { data, error } = await supabase
    .from("jobs_pending")
    .update({ status: "pending", last_error: null })
    .eq("last_error", INFLIGHT)
    .select("id");
  if (error) console.error(`[recover] ${error.message}`);
  else console.log(`[recover] released ${data?.length ?? 0} inflight jobs`);

  const cutoff = new Date(Date.now() - 15 * 60_000).toISOString();
  const { data: orphans, error: orphanErr } = await supabase
    .from("jobs_pending")
    .update({ status: "pending", last_error: null })
    .eq("status", "processing")
    .is("last_error", null)
    .lt("updated_at", cutoff)
    .select("id");
  if (orphanErr) console.error(`[recover] orphan sweep: ${orphanErr.message}`);
  else if (orphans?.length) console.warn(`[recover] released ${orphans.length} marker-less stuck jobs`);
}

// ---------------------------------------------------------------- write one job
// Returns: "done" | "dropped" | "stale" | "deferred". Throws on a db error so the lane can count a strike.
async function writeOne(job: any, sections: Normalized, noLocationReason = "no_location"): Promise<string> {
  const content = getJobContent(job);
  // the model's answers, stored exactly as given (null = the posting does not say)
  const { workplace_type: workplace, employment_type: employment, location } = sections;
  const postedAt = sections.posted_at ?? scrapedPostedIso(job.posted_at);

  // No location = the job is not shown to anyone, so it is not kept. The row
  // stays in jobs_pending as status=failed, under one of two markers:
  //   no_location     the posting really names no place — nothing to revive
  //   loc_unresolved  we never learned where it is (cap, timeout, junk answer)
  // The second is not a verdict, so it stays queryable and is safe to requeue
  // once there is budget. Nothing else throws a job away except a stale date.
  if (!location) {
    const { error } = await supabase.from("jobs_pending")
      .update({ status: "failed", last_error: noLocationReason }).eq("id", job.id);
    if (error) throw new Error(`${noLocationReason} update failed: ${error.message}`);
    return noLocationReason === "no_location" ? "dropped" : "unresolved";
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
    // Never derived from the string alone: a "Worldwide" that survived the check
    // above is evidence-backed, and anything else is not global.
    is_global: isGlobalClaim(location),
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

// Hybrid location reasoning. The bulk call always runs with thinking OFF.
// A second, thinking-ON call is made only for the jobs whose location came back
// unresolved — no country at all, an office name, or null. Cheap path for the
// obvious ones, reasoning for the ones that need it.
const LOCATION_REASONING = (Deno.env.get("LOCATION_REASONING") ?? "on").toLowerCase() !== "off";
const LOCATION_REASON_TOKENS = Number(Deno.env.get("LOCATION_REASON_TOKENS") ?? 3000);
// Places are resolved for a whole batch in ONE call, the same way the bulk pass
// works. One call per job was costing ~0.4 calls/job — about 2x the bulk pass —
// to return a single field.
const LOCATION_BATCH = Number(Deno.env.get("LOCATION_BATCH") ?? 15);
const LOCATION_ITEM_CHARS = Number(Deno.env.get("LOCATION_REASON_CHARS") ?? 900);
// This endpoint answers the BULK call in 25-165s on the same key, so a place call
// needs real headroom: the prompt is tiny but the wall clock is queue + decode.
// 180s sat just above the slowest bulk completion and timed out under load.
const LOCATION_REASON_TIMEOUT_MS = Number(Deno.env.get("LOCATION_REASON_TIMEOUT_MS") ?? 300_000);
// Thinking on nvidia means chat_template_kwargs.enable_thinking, which
// nemotron-3.5-lightning answers with HTTP 500. Off unless explicitly enabled.
const LOCATION_REASON_THINKING = (Deno.env.get("LOCATION_REASON_THINKING") ?? "off").toLowerCase() === "on";
// The reasoning pass is only worth paying for if it is better at places than the
// bulk model. Point it at a real reasoning model (same provider + key, different
// model id). Empty = use MODEL, i.e. the old behaviour.
const LOCATION_REASON_MODEL = (Deno.env.get("LOCATION_REASON_MODEL") ?? "").trim();

function buildBody(messages: unknown[], opts: { maxTokens?: number; thinking?: boolean; model?: string } = {}): Record<string, unknown> {
  const max_tokens = opts.maxTokens ?? MAX_TOKENS;
  const think = opts.thinking === true;
  const model = opts.model || MODEL;
  if (USE === "nvidia") {
    return { model, messages, max_tokens, temperature: 0, chat_template_kwargs: { enable_thinking: think } };
  }
  if (USE === "openrouter") {
    return { model, messages, max_tokens, temperature: 0, response_format: { type: "json_object" }, reasoning: { enabled: think } };
  }
  return { model, messages, max_tokens, temperature: 0, response_format: { type: "json_object" }, reasoning_effort: think ? "low" : GEMINI_REASONING };
}

async function callGemini(apiKey: string, messages: unknown[], opts: { maxTokens?: number; thinking?: boolean; model?: string; timeoutMs?: number } = {}): Promise<any> {
  const res = await fetch(GEMINI_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(buildBody(messages, opts)),
    signal: AbortSignal.timeout(opts.timeoutMs ?? GEMINI_TIMEOUT_MS),
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

// ---------------------------------------------------------------- hybrid location reasoning
const COUNTRY_RE = /(United States|\bUSA\b|\bU\.S\.A?\.?\b|United Kingdom|\bU\.K\.?\b|England|Scotland|Wales|Ireland|Canada|India|Germany|Deutschland|France|Netherlands|Holland|Spain|Portugal|Poland|Italy|Sweden|Norway|Denmark|Finland|Switzerland|Austria|Belgium|Australia|New Zealand|Singapore|Japan|China|South Korea|Korea|Taiwan|Hong Kong|Malaysia|Indonesia|Philippines|Vietnam|Thailand|Brazil|Brasil|Mexico|Argentina|Colombia|Chile|Peru|Uruguay|Egypt|Nigeria|Kenya|Ghana|South Africa|Morocco|Tunisia|Israel|Turkey|United Arab Emirates|\bUAE\b|Saudi Arabia|Qatar|Pakistan|Bangladesh|Sri Lanka|Ukraine|Romania|Czech|Hungary|Greece|Bulgaria|Serbia|Croatia|Estonia|Latvia|Lithuania|Luxembourg|Iceland|Worldwide)/i;
const OFFICE_WORD_RE = /\b(HQ|Headquarters|Head Office|Office|Campus|Site|Plant|Factory|Remote)\b/i;
const ARRANGEMENT_RE = /\b(fully\s+remote|remote|hybrid|on-?site)\b/gi;

/**
 * Strip the words "Remote" / "Hybrid" / "On-site" out of a place. The model is
 * told not to write them, but it does ("Remote - United States", "... Area Hybrid").
 * What is left is the place itself: "United States", "California - San Francisco Bay Area".
 */
function cleanLocationAnswer(value: unknown): string {
  return String(value ?? "")
    .split("|")
    .map((seg) =>
      seg
        .replace(ARRANGEMENT_RE, " ")
        .replace(/^\s*[-,;:]+\s*/, "")
        .replace(/\s*[-,;:]+\s*$/, "")
        .replace(/\s{2,}/g, " ")
        .trim()
    )
    .filter(Boolean)
    .join(" | ");
}

// "Worldwide" is a claim, and a claim has to be paid for by the posting text.
// The shared-taxonomy detectGlobal() cannot do this check here: it assumes the
// location field came from the feed, so feeding it the model's own "Worldwide"
// matches itself and always says yes. These are the body-evidence rules from that
// same file, applied to content only.
const GLOBAL_EVIDENCE = [
  /\b(?:fully\s+)?(?:remote\s*[-–—,(]?\s*)?(?:world\s?wide|globally?|global)\b/,
  /\bwork\s+from\s+anywhere\b/, /\bremote\s*\(\s*(?:anywhere|global|worldwide)\s*\)/,
  /\banywhere\s+in\s+the\s+world\b/, /\bany\s+(?:country|location|time\s*zone)\b/,
  /\blocation\s*:?\s*(?:anywhere|global|worldwide|remote\s*-\s*global)\b/,
  /\bno\s+location\s+restriction/, /\bfully\s+distributed\b/, /\bopen\s+to\s+all\s+locations\b/,
];
const GLOBAL_NEGATION = [
  /\bglobal\s+(?:team|company|leader|brand|reach|presence|impact|player|organi[sz]ation|scale|network|client|customer|footprint|business|mission|headquarters)\b/,
  /\bglobally\s+(?:distributed\s+team|recogni[sz]ed|renowned)\b/,
];
const GLOBAL_CLAIM_RE = /^(?:worldwide|anywhere|global|remote\s*[-–—]\s*(?:global|worldwide|anywhere))$/i;
const GLOBAL_RESTRICTION_RE = /\b(must be (?:based|located|residing|authorized|eligible)|only (?:open to|available to|consider)|eligible to work in|work authori[sz]ation|right to work in|authorized to work|(?:US|USA|UK|EU|EMEA|APAC|CANADA)[- ]?(?:based|only|residents|citizens)|within the (?:US|UK|EU))\b/i;

function isGlobalClaim(location: unknown): boolean {
  return GLOBAL_CLAIM_RE.test(String(location ?? "").trim());
}

/**
 * A "Worldwide" answer is only kept when the posting itself says so — an explicit
 * anywhere/worldwide phrase, or no restriction language at all. Bare "Remote" is
 * not global: most of those are US-only, which is exactly what the shared taxonomy
 * has warned about all along.
 */
function globalClaimSupported(location: unknown, content: string): boolean {
  if (!isGlobalClaim(location)) return true; // not a global claim: nothing to verify
  const body = String(content || "").toLowerCase().slice(0, 1500);
  if (GLOBAL_NEGATION.some((r) => r.test(body))) return false;
  if (GLOBAL_RESTRICTION_RE.test(body)) return false;
  return GLOBAL_EVIDENCE.some((r) => r.test(body));
}

const BARE_OFFICE_RE = /^(HQ|Headquarters|Head Office|Office|Campus|Site|Plant|Factory|Multiple|Various)$/i;
const BARE_PLACEHOLDER_RE = /^(all states|all locations|multiple locations|various locations|nationwide|anywhere)$/i;

/**
 * Is this a place the product can actually show? Not a gazetteer and not a
 * country resolver (geo.ts owns that) — only a shape gate: it must not be an
 * office label or a placeholder. It deliberately does NOT require a country from
 * a hand-written list: that list is partial (no Iran, Iraq, Bolivia, Slovenia…)
 * and it was rejecting correct answers like "Tehran, Iran".
 * "Belgium" passes. "Bangalore HQ", "Headquarters" and "Location - All States" do not.
 */
function isUsableLocation(value: unknown): boolean {
  const segs = cleanLocationAnswer(value).split("|").map((s) => s.trim()).filter(Boolean);
  if (!segs.length) return false;
  return segs.every((seg) => {
    if (BARE_OFFICE_RE.test(seg)) return false;
    if (BARE_PLACEHOLDER_RE.test(seg.replace(/^location\s*[-–:]\s*/i, ""))) return false;
    if (OFFICE_WORD_RE.test(seg) && seg.split(",").length < 2) return false;
    return true;
  });
}

/** True when the answer still is not a usable "place, country". */
function locationNeedsReasoning(loc: unknown): boolean {
  const s = cleanLocationAnswer(loc).trim();
  if (!s) return true;
  return s
    .split("|")
    .map((x) => x.trim())
    .filter(Boolean)
    .some((seg) => !COUNTRY_RE.test(seg) || (OFFICE_WORD_RE.test(seg) && seg.split(",").length < 2));
}

/**
 * What the second call concluded. These three are NOT interchangeable:
 *  - resolved    the answer, ready to store
 *  - none        the posting genuinely names no place  -> a truthful drop
 *  - unusable    we did look, and the model still gave something we cannot show
 *  - unavailable we never got an answer at all (cap, timeout, bad JSON, HTTP)
 * Only `none` means "no_location". The other two mean "we do not know yet", and
 * the job must not be parked as if the posting had no location.
 */
type LocVerdict =
  | { kind: "resolved"; location: string }
  | { kind: "none" }
  | { kind: "unusable"; answer: string }
  | { kind: "unavailable"; why: string };

/** The board's own "Location: / Office: / Country:" lines — the cheapest possible signal. */
function locationLines(content: string): string {
  const hits: string[] = [];
  for (const line of String(content || "").split(/\n+/)) {
    if (hits.length >= 5) break;
    if (/^\s*(?:all locations?|location|office|country|city|region|workplace|remote status|job location)\s*:/i.test(line)) {
      hits.push(line.trim().slice(0, 120));
    }
  }
  return hits.join("\n");
}

/**
 * Resolve the places for a whole batch in ONE call. Same idea as the bulk pass:
 * N jobs in, N answers out. Returns a verdict per job id; anything the model
 * skipped comes back as "unavailable", never as a guess.
 * Never throws — a failed batch must not cost the jobs.
 */
// The place call is the one that keeps answering HTTP 500 from the provider while
// the identical bulk call succeeds. Retry it once whatever the thinking flag says,
// and if it keeps failing, stop paying for failures for a while.
let locFailStreak = 0;
let locPausedUntil = 0;
const LOC_FAIL_LIMIT = Math.max(1, Number(Deno.env.get("LOCATION_FAIL_LIMIT") ?? 5));
const LOC_PAUSE_MS = Number(Deno.env.get("LOCATION_PAUSE_MS") ?? 600_000);
// 15 lanes each firing a place call plus a retry is a lot of simultaneous small
// requests, and the provider answers 500 rather than 429 when it is unhappy.
// Keep the place pass to a few in flight; it is not on the critical path.
let locInflight = 0;
const LOC_MAX_CONCURRENT = Math.max(1, Number(Deno.env.get("LOCATION_MAX_CONCURRENT") ?? 3));

async function reasonLocationsBatch(
  keyIdx: number,
  apiKey: string,
  items: { id: string; job: any; current: any }[]
): Promise<Map<string, LocVerdict>> {
  const out = new Map<string, LocVerdict>();
  if (!items.length) return out;
  if (locPausedUntil > Date.now()) {
    state.loc_circuit += items.length;
    for (const it of items) out.set(String(it.id), { kind: "unavailable", why: "circuit_open" });
    return out;
  }

  const blocks = items.map((it) => {
    const content = getJobContent(it.job);
    const lines = locationLines(content);
    return [
      `### id: ${it.id}`,
      `Company: ${it.job.company ?? ""}`,
      `Title: ${it.job.role ?? it.job.title ?? ""}`,
      `Job URL: ${it.job.job_url ?? it.job.url ?? ""}`,
      it.current ? `Fast pass answered: ${JSON.stringify(String(it.current))}` : "Fast pass found no place.",
      lines ? `Board labels:\n${lines}` : "Board labels: (none)",
      "Text:",
      String(content || "").slice(0, LOCATION_ITEM_CHARS),
    ].join("\n");
  });

  const prompt = [
    `Below are ${items.length} job postings. For EACH id, say where the candidate is based.`,
    "",
    "Answer with JSON only:",
    '{"results":[{"id":"<id>","location":"<place>"}]}',
    "One entry per id, same ids, nothing else.",
    "",
    "Rules:",
    '- location is "City, Country", or "City, State, Country" for the US.',
    "- Resolve office names, company office labels and well-known cities/states to their country:",
    '  "Bangalore HQ" -> "Bangalore, India"; "Dresden" -> "Dresden, Germany"; "Austin, TX" -> "Austin, TX, United States".',
    '- A country on its own is a complete answer: "Location: Belgium" -> "Belgium". Never add a city it does not name.',
    '- "Remote - United States" -> "United States". A remote job that names a country still has that country.',
    '- A city district or campus is that city: "Nantes" -> "Nantes, France".',
    "- Use the board labels, the posting text, the company's offices and the job URL (Workday links carry the place).",
    '- Several places: join them with " | ", each resolved the same way.',
    '- Nothing anywhere points to a place: {"id":"<id>","location":null}. Bare "Remote" with no country is null.',
    "- Never invent a place. Never answer with an office label such as HQ or Office.",
    "",
    "JOBS:",
    blocks.join("\n\n"),
  ].join("\n");

  const attempt = async (thinking: boolean) => {
    while (locInflight >= LOC_MAX_CONCURRENT) await sleep(250);
    locInflight++;
    try {
      const slotId = await reserveCall(keyIdx);
      if (slotId == null) { state.loc_skipped_cap++; throw new Error("cap"); }
      try {
        const json = await callGemini(apiKey, [{ role: "user", content: prompt }], {
          // 1,200 was too tight: the provider answers HTTP 500 instead of a
          // truncated finish when generation does not complete inside the budget.
          maxTokens: Math.min(LOCATION_REASON_TOKENS, Math.max(2500, 140 * items.length)),
          thinking,
          model: LOCATION_REASON_MODEL || undefined,
          timeoutMs: LOCATION_REASON_TIMEOUT_MS,
        });
        const tokens = (json?.usage?.prompt_tokens || 0) + (json?.usage?.completion_tokens || 0);
        await settleCall(slotId, tokens, true);
        return json;
      } catch (e) {
        await dropCall(slotId);
        throw e;
      }
    } finally {
      locInflight--; // always release the slot, success or failure
    }
  };

  // One retry, always — flat, after a short pause. A 500 from the provider is
  // usually transient; the old code only retried when thinking was enabled.
  const plans = LOCATION_REASON_THINKING ? [true, false] : [false, false];
  let json: any = null;
  let lastErr: unknown = null;
  for (const think of plans) {
    try {
      json = await attempt(think);
      lastErr = null;
      break;
    } catch (e) {
      lastErr = e;
      if (/cap$/.test(String(e))) break; // out of budget: retrying cannot help
      await sleep(1500);
    }
  }
  if (!json) {
    locFailStreak++;
    console.warn(`[loc-reason] batch failed (${locFailStreak}/${LOC_FAIL_LIMIT}): ${String(lastErr).slice(0, 160)}`);
    if (locFailStreak >= LOC_FAIL_LIMIT) {
      locPausedUntil = Date.now() + LOC_PAUSE_MS;
      locFailStreak = 0;
      console.error(`[loc-reason] place pass paused for ${Math.round(LOC_PAUSE_MS / 60000)}min after ${LOC_FAIL_LIMIT} failures in a row`);
    }
  } else {
    locFailStreak = 0;
  }
  if (!json) {
    for (const it of items) out.set(String(it.id), { kind: "unavailable", why: "batch_failed" });
    return out;
  }

  const raw = String(replyText(json) ?? "").replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  let parsed: any = null;
  try { parsed = JSON.parse(raw); } catch { parsed = null; }
  const list: any[] = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.results) ? parsed.results : [];
  if (!list.length) {
    console.warn(`[loc-reason] unparsable batch reply, ${raw.length} chars: ${raw.slice(0, 120)}`);
    for (const it of items) out.set(String(it.id), { kind: "unavailable", why: "bad_json" });
    return out;
  }

  for (const it of items) {
    const hit = list.find((r) => String(r?.id ?? "") === String(it.id));
    if (!hit || !("location" in hit)) { out.set(String(it.id), { kind: "unavailable", why: "missing_from_reply" }); continue; }
    if (hit.location === null || hit.location === "") { out.set(String(it.id), { kind: "none" }); continue; }
    if (typeof hit.location !== "string") { out.set(String(it.id), { kind: "unavailable", why: "not_a_string" }); continue; }
    const clean = cleanLocationAnswer(hit.location);
    if (!clean) { out.set(String(it.id), { kind: "none" }); continue; }
    if (!isUsableLocation(clean)) { out.set(String(it.id), { kind: "unusable", answer: clean }); continue; }
    out.set(String(it.id), { kind: "resolved", location: clean });
  }
  return out;
}

// ---------------------------------------------------------------- posted date, read without a model
// The board prints its own date ("Posted: 2026-05-01"). That is a deterministic
// field: a regex reads it, and the model is never paid to discover that a posting
// is six months old. Conservative on purpose — no label or no year means null,
// and null means the model decides exactly as before.
const POSTED_LABEL_RE = /\b(?:date\s+posted|posted(?:\s+on)?|published(?:\s+on)?|publication\s+date|date\s+published)\b\s*[:\-\u2013]?\s*([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{4})/i;
const POSTED_REL_RE = /\bposted\s+(?:about\s+)?(\d+)\s+(day|week|month)s?\s+ago\b/i;

/** The posting's own publish date in ms, or null when we are not certain. */
function readPostedDate(content: string): number | null {
  const head = String(content || "").slice(0, 2500);
  const label = POSTED_LABEL_RE.exec(head);
  if (label) {
    const raw = label[1].replace(/(\d+)(st|nd|rd|th)\b/i, "$1");
    const t = Date.parse(raw);
    if (Number.isFinite(t)) {
      if (t - Date.now() > 2 * DAY_MS) return null; // a "posted" date in the future is a parse accident
      return t;
    }
  }
  const rel = POSTED_REL_RE.exec(head);
  if (rel) {
    const n = Number(rel[1]);
    const unit = rel[2].toLowerCase();
    const ms = unit === "day" ? n * DAY_MS : unit === "week" ? n * 7 * DAY_MS : n * 30 * DAY_MS;
    return Date.now() - ms;
  }
  return null;
}

/**
 * Throw out postings that are already too old BEFORE a model call is spent on them.
 * Distinct marker (stale_precheck) so a wrong regex can be requeued on its own
 * without touching anything the model judged. Fails open: if the update errors,
 * every job goes through as before.
 */
async function dropStaleEarly(tag: string, jobs: any[]): Promise<any[]> {
  if (!jobs.length || MAX_AGE_DAYS <= 0) return jobs;
  const keep: any[] = [];
  const stale: string[] = [];
  for (const job of jobs) {
    const t = readPostedDate(getJobContent(job));
    if (t != null && (Date.now() - t) / DAY_MS > MAX_AGE_DAYS) { stale.push(job.id); continue; }
    keep.push(job);
  }
  if (!stale.length) return jobs;
  const { error } = await supabase.from("jobs_pending")
    .update({ status: "failed", last_error: "stale_precheck" }).in("id", stale);
  if (error) {
    console.error(`[${tag}] stale_precheck update failed, processing all: ${error.message}`);
    return jobs;
  }
  state.stale_precheck += stale.length;
  console.log(`[${tag}] stale_precheck: dropped ${stale.length}/${jobs.length} before calling the model`);
  return keep;
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

      // Date first, model second: a posting the board itself dates as too old is
      // dropped here, for free. Done before fitBatch so the release decision below
      // is made on jobs that are actually worth a call.
      const fresh = await dropStaleEarly(tag, claimed);
      if (!fresh.length) { await sleep(500); continue; }

      const { keep: jobs, extra } = fitBatch(fresh);
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

      // (a) normalise, and collect the places the bulk pass could not finish
      type Staged = { job: any; sections: Normalized; noLocationReason: string };
      const staged: Staged[] = [];
      const needPlace: { id: string; job: any; current: any }[] = [];
      const missing: string[] = [];
      for (const job of jobs) {
        const raw = byId.get(String(job.id));
        if (!raw) { missing.push(job.id); continue; }
        const sections = normalizeSections(raw);
        if (!isAcceptableResult(sections)) { missing.push(job.id); continue; }

        // The model is told not to write "Remote"/"Hybrid" inside a place, but it
        // does. Strip it before judging the answer, so "Remote - United States"
        // counts as "United States" instead of sending the job back to reasoning.
        sections.location = sections.location ? cleanLocationAnswer(sections.location) : sections.location;

        // An unsupported "Worldwide" is not a location, it is a guess. Null it here
        // so the place pass gets a chance to find a real country; if it cannot, the
        // job is marked loc_unresolved rather than published as open to everyone.
        if (sections.location && !globalClaimSupported(sections.location, getJobContent(job))) {
          state.global_unverified++;
          sections.location = null;
        }

        // A date we read off the board beats the model's reading of the same date:
        // the regex cannot invent a month, so it stops a fresh posting being killed
        // as stale on a bad guess as well as making the stored posted_at exact.
        const detDate = readPostedDate(getJobContent(job));
        if (detDate != null) sections.posted_at = new Date(detDate).toISOString();

        const item: Staged = { job, sections, noLocationReason: "no_location" };
        staged.push(item);
        if (LOCATION_REASONING && locationNeedsReasoning(sections.location)) {
          needPlace.push({ id: String(job.id), job, current: sections.location });
        }
      }

      // (b) one call for every unresolved place in this batch, in slices
      if (needPlace.length) {
        const verdicts = new Map<string, LocVerdict>();
        for (let i = 0; i < needPlace.length; i += LOCATION_BATCH) {
          const slice = needPlace.slice(i, i + LOCATION_BATCH);
          const got = await reasonLocationsBatch(keyIdx, apiKey, slice);
          for (const [k, v] of got) verdicts.set(k, v);
        }
        state.loc_reasoned += needPlace.length;
        const byJobId = new Map(staged.map((s) => [String(s.job.id), s]));
        for (const item of needPlace) {
          const v = verdicts.get(item.id) ?? ({ kind: "unavailable", why: "missing" } as LocVerdict);
          const s = byJobId.get(item.id)!;
          const bulk = s.sections.location;
          if (v.kind === "resolved") { s.sections.location = v.location; state.loc_resolved++; }
          else if (v.kind === "none") { s.sections.location = null; state.loc_none++; } // honest drop
          else {
            // we looked and never got a usable answer. Keep whatever the bulk pass
            // said; if it said nothing, this job is unknown, not location-less.
            s.sections.location = bulk;
            if (v.kind === "unusable") state.loc_unusable++;
            else state.loc_unavailable++;
            if (!bulk) s.noLocationReason = "loc_unresolved";
          }
        }
      }

      // (c) write
      let done = 0, dropped = 0, stale = 0, unresolved = 0;
      for (const item of staged) {
        try {
          const outcome = await writeOne(item.job, item.sections, item.noLocationReason);
          if (outcome === "done") done++;
          else if (outcome === "dropped") dropped++;
          else if (outcome === "stale") stale++;
          else if (outcome === "unresolved") unresolved++;
        } catch (e) {
          console.error(`[${tag}] write error for ${item.job.id}: ${String(e).slice(0, 200)}`);
          missing.push(item.job.id);
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
      state.jobs_unresolved += unresolved;
      const locCalls = needPlace.length ? Math.ceil(needPlace.length / LOCATION_BATCH) : 0;
      console.log(`[${tag}] of ${jobs.length}: done=${done} no_location=${dropped} unresolved=${unresolved} stale=${stale} released=${missing.length}, loc asked=${needPlace.length} calls=${locCalls}, ${tokens} tokens`);
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
  console.log(`[status] pending=${pending} inflight=${state.inflight}/${state.lanes_total} ok=${state.batches_ok} failed=${state.batches_failed} written=${state.jobs_written} no_location=${state.jobs_dropped} unresolved=${state.jobs_unresolved} stale=${state.jobs_stale} stale_pre=${state.stale_precheck} loc[asked=${state.loc_reasoned} resolved=${state.loc_resolved} none=${state.loc_none} unusable=${state.loc_unusable} unavailable=${state.loc_unavailable} cap=${state.loc_skipped_cap} circuit=${state.loc_circuit}] lastGemini=${(state.last_gemini_ms / 1000).toFixed(1)}s`);
}, 30_000);

console.log(`[init] worker up: ${state.lanes_total} lanes`);