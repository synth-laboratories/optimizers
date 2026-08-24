import { createHash } from "node:crypto";

export const DOMAIN = Buffer.from("synth.canonical-json.v1\0", "utf8");

export class ContractError extends Error {
  code: string;

  constructor(code: string, detail = "") {
    super(`${code}${detail ? `: ${detail}` : ""}`);
    this.code = code;
  }
}

function validate(value: unknown): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new ContractError("canonical_non_finite");
    if (!Number.isSafeInteger(value)) {
      throw new ContractError(Number.isInteger(value) ? "canonical_integer_range" : "canonical_float_forbidden");
    }
    return;
  }
  if (Array.isArray(value)) { value.forEach(validate); return; }
  if (typeof value === "object") { Object.values(value as Record<string, unknown>).forEach(validate); return; }
  throw new ContractError("canonical_type_forbidden", typeof value);
}

function encode(value: unknown): string {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(encode).join(",")}]`;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
  return `{${keys.map((key) => `${encode(key)}:${encode(record[key])}`).join(",")}}`;
}

export function canonicalJson(value: unknown, omitSelfDigest = false): Buffer {
  if (omitSelfDigest) {
    if (value === null || Array.isArray(value) || typeof value !== "object") {
      throw new ContractError("canonical_self_digest_root");
    }
    const { digest: _digest, ...rest } = value as Record<string, unknown>;
    value = rest;
  }
  validate(value);
  return Buffer.from(encode(value), "utf8");
}

export function canonicalDigest(value: unknown, omitSelfDigest = false): string {
  return `sha256:${createHash("sha256").update(DOMAIN).update(canonicalJson(value, omitSelfDigest)).digest("hex")}`;
}
