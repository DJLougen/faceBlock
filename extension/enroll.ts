/**
 * Pure enrollment-state helpers shared by the background worker and the
 * offscreen document. No DOM, no chrome APIs — every function here is
 * unit-testable in Bun.
 *
 * The enrollment contract (see protocol.ts):
 *   RESOLVE_PREVIEW {name, identityId?} -> EnrollPreview (never persists)
 *   CONFIRM_ENROLL {name, faces, identityId?} -> SavedIdentity (persists)
 * An explicit identityId must name an existing saved identity — it is the
 * refresh/replace path. Without one the id is derived from the confirmed
 * name (canonical references.json id when the name is curated, slug
 * otherwise).
 */

import type { BlockedIdentity } from "../src/shared/types.ts";
import type { BlockList, EnrollPreviewFace, SavedIdentity } from "./protocol.ts";

/**
 * Embedding width of the bundled w600k_mbf ArcFace model. This IS pinned to
 * the model: a stored vector of any other width cannot be cosine-compared
 * against what the bundled embedder emits, so it is malformed by definition.
 * (The synthetic benchmark in bench/ declares its own 128-d protocol and
 * never passes through this validator.)
 */
export const MODEL_EMBED_DIM = 512;

/**
 * Experimental operating point for the real w600k_mbf model. NOT calibrated —
 * false negatives are expected; see extension/protocol.ts contract.
 */
export const MATCH_THRESHOLD = 0.4;

/** How far from unit length a stored embedding may drift before it is rejected. */
const NORM_TOLERANCE = 1e-3;

export function normalizeName(s: string): string {
  return s.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * The id a confirmed identity gets from its name alone: the same kebab-case
 * slug references.json ids are written in, with a deterministic hash
 * fallback for names that slugify to nothing.
 */
export function identityId(name: string): string {
  const slug = normalizeName(name)
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (slug) return slug;
  let hash = 0;
  for (const ch of name) hash = (hash * 31 + ch.codePointAt(0)!) >>> 0;
  return `person-${hash.toString(36)}`;
}

/**
 * Validate one stored/confirmed embedding against the bundled model's
 * contract: exactly MODEL_EMBED_DIM components, finite AFTER float32
 * conversion (a finite f64 like 1e40 overflows to Infinity in f32), nonzero,
 * and L2-normalized — embedAligned always normalizes, so an unnormalized
 * vector did not come from the model and cannot be trusted at the cosine
 * gate. Returns a fresh number[] copy, or null when malformed.
 */
export function toValidEmbedding(raw: unknown): number[] | null {
  if (!Array.isArray(raw) || raw.length !== MODEL_EMBED_DIM) return null;
  const f32 = new Float32Array(MODEL_EMBED_DIM);
  let sumSq = 0;
  for (let i = 0; i < MODEL_EMBED_DIM; i++) {
    const v = raw[i];
    if (typeof v !== "number") return null;
    f32[i] = v;
    const f = f32[i]!;
    if (!Number.isFinite(f)) return null;
    sumSq += f * f;
  }
  if (sumSq === 0) return null;
  if (Math.abs(Math.sqrt(sumSq) - 1) > NORM_TOLERANCE) return null;
  return Array.from(f32);
}

/**
 * Defensively validate confirmed preview faces — they crossed a message
 * boundary. Entries with a malformed embedding are dropped rather than
 * trusted, so one bad face cannot poison the identity.
 */
export function toPreviewFaces(raw: unknown): EnrollPreviewFace[] {
  if (!Array.isArray(raw)) return [];
  const out: EnrollPreviewFace[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const f = item as Partial<EnrollPreviewFace>;
    const embedding = toValidEmbedding(f.embedding);
    if (!embedding) continue;
    out.push({
      url: typeof f.url === "string" ? f.url : "",
      thumbUrl: typeof f.thumbUrl === "string" ? f.thumbUrl : undefined,
      filename: typeof f.filename === "string" ? f.filename : "",
      source: typeof f.source === "string" ? f.source : "",
      score: typeof f.score === "number" && Number.isFinite(f.score) ? f.score : 0,
      embedding,
    });
  }
  return out;
}

/**
 * Validate the identities payload that crosses into the offscreen document.
 * Malformed vectors are dropped per-entry; an identity left with none is
 * dropped whole and counted in `droppedIdentities` so the caller can surface
 * a local diagnosis — one corrupt identity must never poison matching for
 * the valid ones.
 */
export function toBlockedIdentities(raw: unknown): {
  identities: BlockedIdentity[];
  droppedIdentities: number;
  degradedIdentities: number;
} {
  const out: BlockedIdentity[] = [];
  let droppedIdentities = 0;
  let degradedIdentities = 0;
  if (!Array.isArray(raw)) return { identities: out, droppedIdentities, degradedIdentities };
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const s = item as Partial<SavedIdentity>;
    if (typeof s.id !== "string" || !Array.isArray(s.embeddings)) continue;
    const embeddings: Float32Array[] = [];
    let rejectedAny = false;
    for (const e of s.embeddings) {
      const valid = toValidEmbedding(e);
      if (valid) embeddings.push(new Float32Array(valid));
      else rejectedAny = true;
    }
    if (embeddings.length === 0) {
      droppedIdentities += 1;
      continue;
    }
    // Survived but lost vectors: still a diagnosis worth surfacing.
    if (rejectedAny) degradedIdentities += 1;
    out.push({
      id: s.id,
      displayName: typeof s.name === "string" ? s.name : undefined,
      embeddings,
      threshold: Number.isFinite(s.threshold) ? (s.threshold as number) : MATCH_THRESHOLD,
      createdAt: typeof s.createdAt === "number" ? s.createdAt : 0,
    });
  }
  return { identities: out, droppedIdentities, degradedIdentities };
}

/**
 * Decide which existing identity a preview targets, if any.
 *
 * Order: an explicit identityId (the refresh path — must exist, else this is
 * an error the caller must surface), then the canonical references.json id
 * carried on a curated preview, then the name's slug, then an exact
 * normalized-name match. Returns {} for a genuinely new person.
 */
export function selectPreviewIdentityId(
  state: BlockList,
  opts: { explicitId?: string | null; canonicalId?: string | null; name: string },
): { identityId?: string; error?: string } {
  const explicit =
    typeof opts.explicitId === "string" && opts.explicitId !== "" ? opts.explicitId : null;
  if (explicit) {
    const hit = state.identities.find((i) => i.id === explicit);
    if (!hit) {
      return { error: `faceBlock: no saved identity has id "${explicit}"` };
    }
    return { identityId: hit.id };
  }
  const canonical = typeof opts.canonicalId === "string" ? opts.canonicalId : null;
  if (canonical) {
    const hit = state.identities.find((i) => i.id === canonical);
    if (hit) return { identityId: hit.id };
  }
  const slug = identityId(opts.name);
  const bySlug = state.identities.find((i) => i.id === slug);
  if (bySlug) return { identityId: bySlug.id };
  const want = normalizeName(opts.name);
  const byName = state.identities.find((i) => normalizeName(i.name) === want);
  if (byName) return { identityId: byName.id };
  return {};
}

/**
 * Merge a confirmed identity into the blocklist. An existing identity with
 * the same id is REPLACED — its embeddings, name, and sources come from the
 * confirmation — but its id, its stored threshold (a user-visible operating
 * point, never silently reset), and its createdAt survive the refresh.
 */
export function mergeConfirmedIdentity(state: BlockList, identity: SavedIdentity): BlockList {
  const idx = state.identities.findIndex((i) => i.id === identity.id);
  if (idx === -1) {
    return { ...state, identities: [...state.identities, identity] };
  }
  const existing = state.identities[idx]!;
  const merged: SavedIdentity = {
    ...identity,
    threshold: Number.isFinite(existing.threshold) ? existing.threshold : identity.threshold,
    createdAt: existing.createdAt > 0 ? existing.createdAt : identity.createdAt,
  };
  const identities = state.identities.slice();
  identities[idx] = merged;
  return { ...state, identities };
}
