import fs from "node:fs";
const src = fs.readFileSync("worker.ts", "utf8").split(/\r?\n/);
// COUNTRY_RE may span lines; take from its start to the line that ends with ";"
const startC = src.findIndex(l => l.startsWith("const COUNTRY_RE"));
let endC = startC;
while (endC < src.length && !src[endC].trim().endsWith(";")) endC++;
const countryRe = src.slice(startC, endC + 1).join("\n");
const startS = src.findIndex(l => l.startsWith("type RemoteScope"));
const startG = src.findIndex(l => l.startsWith("function isGlobalClaim"));
let endG = startG;
while (endG < src.length && src[endG] !== "}") endG++;
const block = src.slice(startS, endG + 1).join("\n");
const out = countryRe + "\n" + block + "\nexport { remoteScopeOf, isGlobalClaim };\n";
fs.writeFileSync("_scope_extract.ts", out, "utf8");
console.log("extracted lines:", (countryRe + block).split("\n").length);
console.log("has remoteScopeOf:", /function remoteScopeOf/.test(out), "| has isGlobalClaim:", /function isGlobalClaim/.test(out), "| has COUNTRY_RE:", /const COUNTRY_RE/.test(out));
