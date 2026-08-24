import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { canonicalDigest, canonicalJson, ContractError } from "./spine_v1.ts";

const path = new URL("../synth-spine-v1/fixtures/valid/canonical-values.json", import.meta.url);
const corpus = JSON.parse(readFileSync(path, "utf8"));
for (const testCase of corpus.cases) {
  assert.equal(canonicalJson(testCase.value, testCase.omit_self_digest).toString("utf8"), testCase.canonical);
  assert.equal(canonicalDigest(testCase.value, testCase.omit_self_digest), testCase.digest);
}
assert.throws(() => canonicalDigest({ n: 1.5 }), (error: unknown) => error instanceof ContractError && error.code === "canonical_float_forbidden");
console.log(`typescript parity: ${corpus.cases.length} golden cases`);
