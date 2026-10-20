// worker.ts - runs on Render (Docker + Deno). No Inngest.
// - LANES_PER_KEY lanes per Gemini key, each lane = claim -> Gemini -> write -> repeat
// - freshness check every 10 min + daily proof
// - every step is logged so you can SEE it working in the Render logs
// Copy ../_shared/redis.ts next to this file (same folder) for markHashesSeen.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { markHashesSeen } from "./redis.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

const USE = (Deno.env.get("LLM") ?? "gemini").toLowerCase(); // "gemini" or "openrouter"
const MODEL = USE === "openrouter"
      ? (Deno.env.get("OPENROUTER_MODEL") ?? "nvidia/nemotron-3-super-120b-a12b:free")
  : (Deno.env.get("GEMINI_CLEAN_MODEL") ?? "gemini-3.8-flash");
const PROVIDER = USE === "openrouter" ? "openrouter-clean" : "gemini-clean";
const GEMINI_URL = USE === "openrouter"
  ? "https://openrouter.ai/api/v1/chat/completions"
  : "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const BATCH = Number(Deno.env.get("BATCH") ?? 60);
const MIN_PENDING = Number(Deno.env.get("MIN_PENDING") ?? 20); // don't waste a call on tiny batches
const DAILY_CAP = Number(Deno.env.get("DAILY_CAP") ?? 1500);   // requests per key per day
const LANES_PER_KEY = Number(Deno.env.get("LANES_PER_KEY") ?? 10);
const IDLE_MS = 15_000;
const GEMINI_TIMEOUT_MS = 300_000;
const MAX_CHARS = 10000;
const EXPIRES_DAYS = 90;
const REQUIRE_REAL_LOCATION = true;
const STORE_RAW_TEXT = false;
const GLOBAL_LABEL = "Worldwide";
const FRESHNESS_TABLE = "job_freshness";
const REVIEW_REQUIRED = new Set(["workplace_type", "employment_type", "location"]);
const INFLIGHT = "openrouter_inflight";

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
    const k = (Deno.env.get(`${USE === "openrouter" ? "OPENROUTER_API_KEY" : "GEMINI_API_KEY"}_${i}`) ?? "").trim();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push({ idx: i, key: k });
  }
    console.log(`[init] provider=${USE} model=${MODEL || "(NOT SET)"} url=${GEMINI_URL} keys=${out.length} lanes_per_key=${LANES_PER_KEY} batch=${BATCH}`);
  return out;
}
const KEYS = loadKeys();

function nextUtcMidnight(now = Date.now()): number {
  const d = new Date(now);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() + 1, 0, 5, 0);
}

// in-memory pause per key (429 / daily cap)
const pausedUntil = new Map<number, number>();

// live counters, shown in logs + /health
const state = {
  started_at: new Date().toISOString(),
  lanes_total: 0,
  inflight: 0,
  batches_ok: 0,
  batches_failed: 0,
  jobs_written: 0,
  last_gemini_ms: 0,
  freshness_last_run: null as string | null,
};

// ---------------------------------------------------------------- prompt
const SECTION_INSTRUCTIONS = `You receive several FULL job postings (already cleaned plain text, no HTML). Classify EACH job separately.
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
Also extract these fields by reading the WHOLE posting for an explicit statement — never guess or infer from context. The work arrangement or employment type is often one sentence far from the top.
- "workplace_evidence": FIRST find the exact short phrase (under 25 words, copied word-for-word from the posting) that states whether the role is remote, hybrid or onsite. null if the posting has no such phrase. Fill this BEFORE detected_workplace_type.
- "detected_workplace_type": one of "REMOTE", "HYBRID", "ONSITE", or null. Set it ONLY from workplace_evidence. If workplace_evidence is null this MUST be null. A city/office address alone is NOT a statement of arrangement. If the phrase says "hybrid" (even next to "onsite" or "in office"), the answer is HYBRID. Ignore "remote monitoring", "remote sensing", "hybrid cloud" or "hybrid vehicles". A company-level blurb such as "we are a remote-first company" is NOT evidence about this role unless the posting says this role itself is remote.
- "employment_evidence": the exact short phrase (under 25 words, copied word-for-word) that states the type of employment. null if there is none.
- "detected_employment_type": one of "FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY", "INTERNSHIP", "VOLUNTEER", or null. Set it ONLY from employment_evidence. "contract assignment" or "fixed-term contract position" means CONTRACT. "employment contract" or "subject to contract" is NOT evidence. "40 hours per week" about THIS role counts as FULL_TIME. A title containing the whole word "Intern" or "Internship" counts as INTERNSHIP. Do not infer from salary or seniority.
- "location_evidence": the exact short phrase (under 25 words, copied word-for-word from the Location line, the title, the description, or the job URL) that says where the candidate must be based, or that the job is open worldwide. null if nothing says it.
- "location_scope": one of "WORLDWIDE", "RESTRICTED", "PLACE", "UNKNOWN". WORLDWIDE only when the posting explicitly says anywhere / worldwide. RESTRICTED when it hard-limits countries. PLACE when it names a city, office or country. UNKNOWN when nothing is stated. A global company is NOT WORLDWIDE.
- "location_value": for RESTRICTED or PLACE only. FULL ENGLISH COUNTRY NAMES, never US/USA/UK. Format "City, Country" or "City, State, Country". You may use the URL host with the posting (galvestoncountytx plus "County of Galveston" → "Galveston, Texas, United States"). If you can name the city but not the country, return the city only. Never guess a country when the name exists in more than one country. Never infer a country from the company name alone. Do NOT include the words remote, hybrid or onsite. null for WORLDWIDE and UNKNOWN.
Copy each job's "id" exactly. Do not skip a job. Do not add jobs.
Respond with ONLY this JSON shape, no markdown:
{"results":[{"id":"...","role_description":"...","requirements":"...","nice_to_have":"...","qualifications":"...","compensation":"...","benefits":"...","work_setup":"...","about_company":"...","company_culture":"...","extracted_skills":["..."],"workplace_evidence":null,"detected_workplace_type":null,"employment_evidence":null,"detected_employment_type":null,"location_evidence":null,"location_scope":"UNKNOWN","location_value":null}]}`;

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
  detected_workplace_type?: string | null;
  employment_evidence?: string | null;
  detected_employment_type?: string | null;
  location_evidence?: string | null;
  location_scope?: string | null;
  location_value?: string | null;
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
  workplace_evidence: string | null;
  detected_workplace_type: string | null;
  employment_evidence: string | null;
  detected_employment_type: string | null;
  location_evidence: string | null;
  location_scope: string | null;
  location_value: string | null;
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

const VALID_WORKPLACE = new Set(["REMOTE", "HYBRID", "ONSITE"]);
const VALID_EMPTYPE = new Set(["FULL_TIME", "PART_TIME", "CONTRACT", "TEMPORARY", "INTERNSHIP", "VOLUNTEER"]);

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
  const wp = String(raw.detected_workplace_type || "").toUpperCase().trim();
  const et = String(raw.detected_employment_type || "").toUpperCase().trim();
  const scope = String(raw.location_scope || "").toUpperCase().trim();
  return {
    role_description, requirements, nice_to_have, qualifications, compensation, benefits,
    work_setup, about_company, company_culture, full_description, extracted_skills,
    workplace_evidence: raw.workplace_evidence ? String(raw.workplace_evidence) : null,
    detected_workplace_type: VALID_WORKPLACE.has(wp) ? wp : null,
    employment_evidence: raw.employment_evidence ? String(raw.employment_evidence) : null,
    detected_employment_type: VALID_EMPTYPE.has(et) ? et : null,
    location_evidence: raw.location_evidence ? String(raw.location_evidence) : null,
    location_scope: ["WORLDWIDE", "RESTRICTED", "PLACE", "UNKNOWN"].includes(scope) ? scope : "UNKNOWN",
    location_value: raw.location_value ? String(raw.location_value).slice(0, 200) : null,
  };
}

function isAcceptableResult(sections: Normalized): boolean {
  return sections.full_description.length >= 80;
}

// ---------------------------------------------------------------- location logic
const OPEN_KEYS = new Set(["worldwide", "world-wide", "anywhere", "global"]);
const REGION_WORDS = new Set([
  "africa", "sub-saharan africa", "emea", "mea", "west africa", "east africa", "north africa",
  "central africa", "southern africa", "europe", "north america", "americas", "south america",
  "latam", "latin america", "asia", "apac", "oceania",
]);
const CITY_WORDS = new Set([
  "lagos", "abuja", "ile-ife", "ibadan", "kano", "port harcourt", "nairobi", "mombasa", "accra",
  "kumasi", "johannesburg", "cape town", "durban", "kigali", "kampala", "dakar", "addis ababa",
  "london", "manchester", "birmingham", "new york", "san francisco", "seattle", "austin", "chicago",
  "boston", "los angeles", "toronto", "vancouver", "berlin", "paris", "amsterdam", "madrid",
  "stockholm", "dubai", "mumbai", "bangalore", "bengaluru", "hyderabad", "singapore", "sydney",
  "melbourne", "sao paulo", "dublin", "edinburgh", "zurich", "lisbon", "rome", "milan", "warsaw",
  "prague", "vienna", "brussels", "copenhagen", "helsinki", "oslo", "tokyo", "seoul", "taipei",
  "hong kong", "shanghai", "beijing", "shenzhen", "tel aviv", "cairo", "casablanca", "tunis",
  "pretoria", "abu dhabi", "doha", "riyadh", "delhi", "pune", "chennai", "gurgaon",
  "mountain view", "palo alto", "sunnyvale", "redmond", "denver", "atlanta", "miami", "dallas",
  "houston", "philadelphia", "washington", "concord", "el segundo", "galveston",
]);
const COUNTRY_WORDS = new Set(`
afghanistan albania algeria andorra angola argentina armenia australia austria azerbaijan
bahamas bahrain bangladesh barbados belarus belgium belize benin bhutan bolivia bosnia botswana
brazil brunei bulgaria burkina cameroon canada cape verde chad chile china colombia congo
costa rica croatia cuba cyprus czech republic czechia denmark djibouti ecuador egypt
el salvador eritrea estonia eswatini ethiopia fiji finland france gabon gambia georgia germany
ghana greece grenada guatemala guinea guyana haiti honduras hungary iceland india indonesia
iran iraq ireland israel italy jamaica japan jordan kazakhstan kenya kuwait kyrgyzstan laos
latvia lebanon lesotho liberia libya liechtenstein lithuania luxembourg madagascar malawi
malaysia maldives mali malta mauritania mauritius mexico moldova monaco mongolia montenegro
morocco mozambique myanmar namibia nepal netherlands new zealand nicaragua niger nigeria
north macedonia norway oman pakistan panama papua new guinea paraguay peru philippines poland
portugal qatar romania russia rwanda samoa san marino saudi arabia senegal serbia seychelles
sierra leone singapore slovakia slovenia somalia south africa south korea spain sri lanka
sudan sweden switzerland syria taiwan tajikistan tanzania thailand timor-leste togo tunisia
turkey turkmenistan uganda ukraine united arab emirates united kingdom united states uruguay
uzbekistan venezuela vietnam yemen zambia zimbabwe usa uk britain england scotland wales
uae holland
`.split(/\s+/).map((s) => s.trim()).filter(Boolean).concat([
  "united states", "united kingdom", "united states of america", "united arab emirates",
  "south africa", "south korea", "new zealand", "saudi arabia", "sri lanka", "costa rica",
  "el salvador", "sierra leone", "czech republic", "dominican republic", "north macedonia",
  "papua new guinea", "burkina faso", "cape verde", "great britain", "northern ireland",
]));

const LOC_NOISE_RE = /\b(remote|hybrid|on-?site|fully|wfh|work from home|based|only|within|in)\b/gi;
const WORK_MODE_RE = /\b(remote|hybrid|on-?site|onsite|in-?office|in\s+office|work\s+from\s+home|wfh|telecommut\w*|days?\s+per\s+week\s+(?:in\s+)?(?:the\s+)?office|\d+\s+days?\s+(?:on-?site|onsite|in\s+(?:the\s+)?office|remote)|office\s+based|fully\s+remote|remote-?first)\b/i;
const BARE_PLACE_RE = /^(?:(?:fully\s+)?remote|hybrid|on-?site|wfh|work[\s-]*from[\s-]*home)$/i;

function normText(t: string): string {
  return (t || "").toLowerCase().match(/[a-z0-9]+/g)?.join(" ") ?? "";
}

function evidenceOk(evidence: string | null | undefined, hay: string): boolean {
  if (!evidence || typeof evidence !== "string") return false;
  const pieces = evidence.split(/\.\.\.|…/).map(normText).filter(Boolean);
  return pieces.length > 0 && pieces.every((p) => hay.includes(p));
}

function workplaceEvidenceOk(evidence: string | null | undefined, hay: string): boolean {
  return evidenceOk(evidence, hay) && WORK_MODE_RE.test(evidence || "");
}

function locKeys(text: string): string[] {
  const t = (text || "").toLowerCase().replace(LOC_NOISE_RE, " ");
  const out: string[] = [];
  for (const part of t.split(/\s[-–—]\s|[,;|/()]+|\s(?:or|and)\s/)) {
    const cleaned = part.replace(/\s+/g, " ").trim();
    if (cleaned) out.push(cleaned);
  }
  return out;
}

function locKnown(key: string): boolean {
  return OPEN_KEYS.has(key) || REGION_WORDS.has(key) || CITY_WORDS.has(key) || COUNTRY_WORDS.has(key);
}

function locClass(text: string): "EMPTY" | "GLOBAL_OK" | "SPECIFIC_OK" | "UNREADABLE" {
  const t = (text || "").trim();
  if (!t || /^\d+\s+locations?$/i.test(t)) return "EMPTY";
  if (BARE_PLACE_RE.test(t)) return "EMPTY";
  const keys = locKeys(t);
  if (keys.some((k) => OPEN_KEYS.has(k))) return "GLOBAL_OK";
  if (keys.some(locKnown)) return "SPECIFIC_OK";
  return "UNREADABLE";
}

function isRealPlace(text: string): boolean {
  const t = (text || "").trim();
  if (!t) return false;
  const c = locClass(t);
  return c === "GLOBAL_OK" || c === "SPECIFIC_OK";
}

function getJobContent(job: any): string {
  if (job.content) return String(job.content);
  if (Array.isArray(job.chunks) && job.chunks.length > 0) {
    return job.chunks.map((c: any) => c?.text ?? "").filter(Boolean).join("\n\n");
  }
  return "";
}

function decideLocation(job: any, sections: Normalized): { final: string; changed: boolean; note: string } {
  const hay = normText(`${job.title || ""} ${job.location || ""} ${job.job_url || ""} ${getJobContent(job)}`);
  const scraped = String(job.location || "").trim();
  const cls = locClass(scraped);
  const scope = sections.location_scope;
  if (!scope || !["WORLDWIDE", "RESTRICTED", "PLACE"].includes(scope) || !evidenceOk(sections.location_evidence, hay)) {
    return { final: scraped, changed: false, note: "" };
  }
  let value = String(sections.location_value || "").replace(/\b(remote|hybrid|on-?site)\b/gi, " ");
  value = value.replace(/\s*[-–—]\s*(?=,|$)/g, "").replace(/\s+/g, " ").replace(/^[\s,;|/-]+|[\s,;|/-]+$/g, "");
  const valueOk = Boolean(value) && locClass(value) === "SPECIFIC_OK";
  if (scope === "WORLDWIDE") {
    if (cls === "GLOBAL_OK") return { final: scraped, changed: false, note: "" };
    return { final: GLOBAL_LABEL, changed: true, note: "posting says worldwide" };
  }
  if (valueOk) {
    const scrapedKeys = new Set(locKeys(scraped));
    if (cls === "SPECIFIC_OK" && locKeys(value).some((k) => scrapedKeys.has(k))) {
      return { final: scraped, changed: false, note: "" };
    }
    return { final: value.slice(0, 200), changed: true, note: "posting names a place" };
  }
  return { final: scraped, changed: false, note: "" };
}

function postedIso(v: unknown): string | null {
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

// ---------------------------------------------------------------- prompt building + parsing
function clip(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n);
}

function buildUserMessage(jobs: any[]): string {
  return jobs.map((job) => {
    const body = clip(getJobContent(job), MAX_CHARS);
    return [
      `JOB id=${job.id}`,
      `Title: ${job.title || ""}`,
      `URL: ${job.job_url || ""}`,
      `Location: ${job.location || ""}`,
      "",
      body,
    ].join("\n");
  }).join("\n\n-----\n\n");
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

async function skipNoLocation(job: any) {
  const { error } = await supabase.from("jobs_pending").update({
    status: "failed",
    updated_at: new Date().toISOString(),
    last_error: "bare_remote_or_empty_location",
  }).eq("id", job.id);
  if (error) await supabase.from("jobs_pending").update({ status: "failed" }).eq("id", job.id);
}

// ---------------------------------------------------------------- write one job
async function writeOne(job: any, sections: Normalized): Promise<string> {
  const hay = normText(`${job.title || ""} ${job.location || ""} ${job.job_url || ""} ${getJobContent(job)}`);
  let detectedWorkplace = sections.detected_workplace_type;
  let detectedEmp = sections.detected_employment_type;
  if (!workplaceEvidenceOk(sections.workplace_evidence, hay)) detectedWorkplace = null;
  else if (sections.workplace_evidence && (/\bhybrid\b/i.test(sections.workplace_evidence) || (/\bremote\b/i.test(sections.workplace_evidence) && /\b(on-?site|onsite|office)\b/i.test(sections.workplace_evidence)))) {
    detectedWorkplace = "HYBRID";
  }
  if (!evidenceOk(sections.employment_evidence, hay)) detectedEmp = null;
  const scrapeWorkplace = String(job.workplace_type || job.job_type || "").toUpperCase() || null;
  const scrapeEmp = String(job.employment_type || "").toUpperCase() || null;
  let finalWorkplace = (detectedWorkplace || (VALID_WORKPLACE.has(scrapeWorkplace || "") ? scrapeWorkplace : null)) as string | null;
  let finalEmp = (detectedEmp || (VALID_EMPTYPE.has(scrapeEmp || "") ? scrapeEmp : null)) as string | null;
  const decided = decideLocation(job, sections);
  let finalLocation = decided.final;
  if (finalLocation && locClass(finalLocation) === "EMPTY") finalLocation = "";
  if (!isRealPlace(finalLocation)) {
    let alt = String(sections.location_value || "").replace(/\b(remote|hybrid|on-?site)\b/gi, " ");
    alt = alt.replace(/\s+/g, " ").replace(/^[\s,;|/-]+|[\s,;|/-]+$/g, "");
    if (isRealPlace(alt)) finalLocation = alt.slice(0, 200);
  }
  if (REQUIRE_REAL_LOCATION && !isRealPlace(finalLocation)) {
    await skipNoLocation(job);
    return "skipped_loc";
  }
  const body = `${job.title || ""} ${getJobContent(job)}`;
  const hasWorkMode = /\b(remote|hybrid|on-?site|work from home|wfh|telecommut)\b/i.test(body);
  const placeClass = locClass(finalLocation);
  if (!finalWorkplace) {
    if (placeClass === "GLOBAL_OK") finalWorkplace = "REMOTE";
    else if (placeClass === "SPECIFIC_OK" && !hasWorkMode) finalWorkplace = "ONSITE";
  }
  if (!finalEmp) finalEmp = "FULL_TIME";
  const missing = [
    ["workplace_type", Boolean(finalWorkplace)],
    ["employment_type", Boolean(finalEmp)],
    ["location", Boolean(finalLocation) && placeClass !== "EMPTY"],
  ].filter(([name, found]) => !found && REVIEW_REQUIRED.has(String(name))).map(([name]) => name);
  if (missing.length) {
    const { error } = await supabase.from("jobs_needs_review").upsert({
      job_url: job.job_url,
      job_url_normalized: job.job_url_normalized,
      dedup_hash: job.dedup_hash,
      company: job.company || "Unknown",
      title: job.title,
      location: job.location || "",
      content: getJobContent(job),
      source: job.source,
      pipe: job.pipe || "pipe1",
      posted_at: job.posted_at ?? null,
      employment_type: finalEmp,
      reason: "missing_after_llm: " + missing.join(", "),
    }, { onConflict: "dedup_hash" });
    if (!error) {
      await supabase.from("jobs_pending").update({ status: "done", last_error: null }).eq("id", job.id);
      return "needs_review";
    }
  }
  const salary = extractSalaryFromCompensation(sections.compensation);
  const expiresIso = new Date(Date.now() + EXPIRES_DAYS * 24 * 60 * 60 * 1000).toISOString();
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
    role: job.title,
    location: finalLocation,
    job_type: finalWorkplace,
    workplace_type: finalWorkplace,
    eligible_countries: job.eligible_countries ?? null,
    eligible_cities: job.eligible_cities ?? null,
    employment_type: finalEmp,
    regions: job.regions ?? null,
    is_global: placeClass === "GLOBAL_OK" ? true : (job.is_global ?? null),
    requires_us_citizenship: job.requires_us_citizenship ?? null,
    role_description: sections.role_description,
    requirements: sections.requirements,
    nice_to_have: sections.nice_to_have,
    qualifications: sections.qualifications,
    compensation: sections.compensation,
    benefits: sections.benefits,
    work_setup: sections.work_setup,
    about_company: sections.about_company,
    company_culture: sections.company_culture,
    extracted_skills: sections.extracted_skills,
    posted_at: postedIso(job.posted_at),
    salary_min: salary.min,
    salary_max: salary.max,
    salary_currency: salary.currency,
    raw_text: STORE_RAW_TEXT ? getJobContent(job) : "",
    source: job.source,
    pipe: job.pipe || "pipe1",
    tag_source: "llm-labeled-gemini",
    is_active: true,
    tagged_at: new Date().toISOString(),
    scraped_at: new Date().toISOString(),
    expires_at: expiresIso,
  }, { onConflict: "dedup_hash" });
  if (masterErr) {
    await supabase.from("jobs_pending").update({ status: "pending", last_error: null }).eq("id", job.id);
    return "deferred";
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

async function callGemini(apiKey: string, messages: unknown[]): Promise<any> {
  const res = await fetch(GEMINI_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: MODEL,
      messages,
      max_tokens: USE === "openrouter" ? 16000 : 65000,
      response_format: { type: "json_object" },
      ...(USE === "openrouter" ? {} : { reasoning_effort: "low" }),
    }),
    signal: AbortSignal.timeout(GEMINI_TIMEOUT_MS),
  });
  const text = await res.text();
  if (!res.ok) throw new GeminiError(res.status, text.slice(0, 300));
  return JSON.parse(text);
}

// ---------------------------------------------------------------- one lane = one worker loop
async function lane(keyIdx: number, laneNo: number, apiKey: string) {
  const tag = `k${keyIdx}-l${laneNo}`;
  await sleep(laneNo * 700); // stagger so 10 lanes don't hit the db at the same millisecond

  while (true) {
    try {
      if ((pausedUntil.get(keyIdx) ?? 0) > Date.now()) { await sleep(5000); continue; }

      if ((await pendingCount()) < MIN_PENDING) { await sleep(IDLE_MS); continue; }

      const jobs = await claimBatch();
      if (!jobs.length) { await sleep(IDLE_MS); continue; }

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
        { role: "system", content: SECTION_INSTRUCTIONS },
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
        } else if (err instanceof GeminiError && [500, 502, 503, 504].includes(err.status)) {
          await dropCall(slotId); // provider overloaded, don't burn a daily slot
          pausedUntil.set(keyIdx, Date.now() + 30_000);
          console.warn(`[${tag}] ${err.status} overloaded, key ${keyIdx} backing off 30s`);
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
        console.error(`[${tag}] bad json: ${String(e).slice(0, 200)} | ${JSON.stringify(payload).slice(0, 300)}`);
        await release(ids);
        await settleCall(slotId, 0, false);
        state.batches_failed++;
        continue;
      }

      const byId = new Map(results.map((r) => [String(r.id || ""), r]));
      let done = 0;
      const missing: string[] = [];
      for (const job of jobs) {
        const raw = byId.get(String(job.id));
        if (!raw) { missing.push(job.id); continue; }
        const sections = normalizeSections(raw);
        if (!isAcceptableResult(sections)) { missing.push(job.id); continue; }
        try {
          const outcome = await writeOne(job, sections);
          if (outcome === "done" || outcome === "needs_review" || outcome === "skipped_loc") done++;
        } catch (e) {
          console.error(`[${tag}] write error for ${job.id}: ${String(e).slice(0, 200)}`);
          missing.push(job.id);
        }
      }
      await release(missing);
      const tokens = (payload?.usage?.prompt_tokens || 0) + (payload?.usage?.completion_tokens || 0);
      await settleCall(slotId, tokens, true);
      state.batches_ok++;
      state.jobs_written += done;
      console.log(`[${tag}] wrote ${done}/${jobs.length} (released ${missing.length}), ${tokens} tokens`);
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

if (!KEYS.length) { console.error(`[init] NO KEYS FOUND for LLM=${USE}`); Deno.exit(1); }
if (USE === "openrouter" && !MODEL) { console.error("[init] OPENROUTER_MODEL is not set"); Deno.exit(1); }
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
  console.log(`[status] pending=${pending} inflight=${state.inflight}/${state.lanes_total} ok=${state.batches_ok} failed=${state.batches_failed} written=${state.jobs_written} lastGemini=${(state.last_gemini_ms / 1000).toFixed(1)}s`);
}, 30_000);

console.log(`[init] worker up: ${state.lanes_total} lanes`);