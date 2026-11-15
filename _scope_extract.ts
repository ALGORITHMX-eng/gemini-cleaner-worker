const COUNTRY_RE = /(United States|\bUSA\b|\bU\.S\.A?\.?\b|United Kingdom|\bU\.K\.?\b|England|Scotland|Wales|Ireland|Canada|India|Germany|Deutschland|France|Netherlands|Holland|Spain|Portugal|Poland|Italy|Sweden|Norway|Denmark|Finland|Switzerland|Austria|Belgium|Australia|New Zealand|Singapore|Japan|China|South Korea|Korea|Taiwan|Hong Kong|Malaysia|Indonesia|Philippines|Vietnam|Thailand|Brazil|Brasil|Mexico|Argentina|Colombia|Chile|Peru|Uruguay|Egypt|Nigeria|Kenya|Ghana|South Africa|Morocco|Tunisia|Israel|Turkey|United Arab Emirates|\bUAE\b|Saudi Arabia|Qatar|Pakistan|Bangladesh|Sri Lanka|Ukraine|Romania|Czech|Hungary|Greece|Bulgaria|Serbia|Croatia|Estonia|Latvia|Lithuania|Luxembourg|Iceland|Worldwide)/i;
type RemoteScope = { kind: "worldwide" | "country" | "remote_unknown" | "unknown"; country?: string };

const SCOPE_LINE_RE = /^\s*(location|all locations|office|country|workplace type|job location type|applicant countries?|applicant location requirements?)\s*[:\-–]\s*(.+)$/gim;
const SHORT_COUNTRY_RE = /(?:^|[\s,|(/-])(usa|u\.s\.a?|us|uk|uae|north america|emea|apac|latam)(?=$|[\s,|)/-])/i;
// "Remote, Global" and "Global - Remote" both mean open world; the comma form used to
// slip through and land in remote_unknown, which would have demoted 21 correct rows.
const OPEN_WORLD_RE = /\b(worldwide|world wide|any country|any countries|anywhere in the world|all countries|global)\b|remote\s*[,\-–]\s*(?:global|worldwide)/i;
// The publisher stated a place, and the place is "remote" with no country attached.
const BARE_REMOTE_RE = /\b(remote|work from home|wfh|telecommute|home[- ]?based|fully distributed|distributed team)\b/i;

// "USA", "US" and "United States of America" are one country, not three.
function canonicalCountry(raw: string): string {
  const v = raw.trim();
  if (/^(usa|u\.s\.?a?\.?|us|united states.*)$/i.test(v)) return "United States";
  if (/^(uk|u\.k\.?|united kingdom|england|scotland|wales)$/i.test(v)) return "United Kingdom";
  if (/^(uae|united arab emirates)$/i.test(v)) return "United Arab Emirates";
  return v.replace(/\b\w/g, (c) => c.toUpperCase());
}

function countriesIn(text: string): string[] {
  const out: string[] = [];
  const push = (raw: string) => {
    if (/^worldwide$/i.test(raw)) return;
    const v = canonicalCountry(raw);
    if (!out.some((c) => c.toLowerCase() === v.toLowerCase())) out.push(v);
  };
  for (const m of text.matchAll(new RegExp(COUNTRY_RE.source, "gi"))) push(m[0]);
  for (const m of text.matchAll(new RegExp(SHORT_COUNTRY_RE.source, "gi"))) push(m[1]);
  return out;
}

function remoteScopeOf(content: string): RemoteScope {
  const head = String(content || "").slice(0, 1500);
  const lines = [...head.matchAll(SCOPE_LINE_RE)].map((m) => `${m[1].toLowerCase()}: ${m[2].trim()}`);
  if (!lines.length) return { kind: "unknown" };
  const header = lines.join("\n");

  const applicant = /applicant (?:countries?|location requirements?)\s*:\s*(.+)/i.exec(header)?.[1] ?? "";
  if (applicant) {
    if (OPEN_WORLD_RE.test(applicant)) return { kind: "worldwide" };
    const list = countriesIn(applicant);
    // Several countries is not one place: no single value can be shown honestly.
    return list.length === 1 ? { kind: "country", country: list[0] } : { kind: "country" };
  }

  // A bare "TELECOMMUTE" is deliberately NOT read as worldwide: publishers omit
  // applicantLocationRequirements even when the role is US-only, so absence there
  // proves nothing. Only an explicit open-world phrase does.
  const named = countriesIn(header);
  if (named.length === 1) return { kind: "country", country: named[0] };
  if (named.length > 1) return { kind: "country" };

  if (OPEN_WORLD_RE.test(header)) return { kind: "worldwide" };

  // The publisher named a place and the place is just "Remote". No country, no
  // applicant list, no open-world phrase: the scope is genuinely unstated, so the
  // model's "Worldwide" is not allowed to stand. Distinct from "unknown", where the
  // posting carries no location line at all and silence still leaves it to the model.
  if (BARE_REMOTE_RE.test(header)) return { kind: "remote_unknown" };

  return { kind: "unknown" };
}
const GLOBAL_CLAIM_RE = /^(?:worldwide|anywhere|global|remote\s*[-–—]\s*(?:global|worldwide|anywhere))$/i;

// Not a validator: it only labels the model's own answer so the row can carry
// is_global = true. The model decided the word; this reads that word back.
function isGlobalClaim(location: unknown): boolean {
  return GLOBAL_CLAIM_RE.test(String(location ?? "").trim());
}
export { remoteScopeOf, isGlobalClaim };
