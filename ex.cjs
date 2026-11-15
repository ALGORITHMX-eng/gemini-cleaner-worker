const fs = require("fs");
const src = fs.readFileSync("worker.ts", "utf8");
const s = src.indexOf("const COUNTRY_RE");
const e = src.indexOf("Second, thinking-ON call for one job");
const start2 = src.lastIndexOf("/**", e);
const snip = src.slice(s, start2);
fs.writeFileSync("C:/Users/Johnson Olalere/AccioWork/2026-10-25-05-18-28-094-11f74204/loccheck.ts", snip + "\nexport { locationNeedsReasoning };\n");
console.log("snippet chars", snip.length, "| end marker at", e);
