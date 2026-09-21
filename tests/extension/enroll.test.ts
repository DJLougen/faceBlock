import { describe, expect, test } from "bun:test";
import {
  MODEL_EMBED_DIM,
  identityId,
  mergeConfirmedIdentity,
  normalizeName,
  selectPreviewIdentityId,
  toBlockedIdentities,
  toPreviewFaces,
  toValidEmbedding,
} from "../../extension/enroll.ts";
import type { BlockList, SavedIdentity } from "../../extension/protocol.ts";

/** A valid model embedding: 512 components, finite, unit L2 norm. */
function unitEmbedding(seed = 1): number[] {
  const v = new Array<number>(MODEL_EMBED_DIM).fill(0);
  v[seed % MODEL_EMBED_DIM] = 1;
  return v;
}

function identity(partial: Partial<SavedIdentity> = {}): SavedIdentity {
  return {
    id: "theo-browne",
    name: "Theo Browne",
    embeddings: [unitEmbedding()],
    threshold: 0.55,
    sources: ["a"],
    createdAt: 1000,
    ...partial,
  };
}

function stateWith(...identities: SavedIdentity[]): BlockList {
  return { identities, enabled: true, revision: 0 };
}

describe("identityId", () => {
  test("slugifies names the way references.json ids are written", () => {
    expect(identityId("Theo Browne")).toBe("theo-browne");
    expect(identityId("  Dwarkesh   Patel ")).toBe("dwarkesh-patel");
  });

  test("falls back to a deterministic hash for names that slugify to nothing", () => {
    const a = identityId("!!!");
    expect(a.startsWith("person-")).toBe(true);
    expect(identityId("!!!")).toBe(a);
  });
});

describe("toValidEmbedding", () => {
  test("accepts a real 512-d unit vector", () => {
    const v = unitEmbedding(7);
    const out = toValidEmbedding(v);
    expect(out).not.toBeNull();
    expect(out!.length).toBe(MODEL_EMBED_DIM);
    expect(out![7]).toBe(1);
  });

  test("rejects wrong widths, including the benchmark's 128-d protocol", () => {
    expect(toValidEmbedding(new Array(128).fill(0.1))).toBeNull();
    expect(toValidEmbedding([1])).toBeNull();
    expect(toValidEmbedding([])).toBeNull();
  });

  test("rejects non-finite values after float32 conversion", () => {
    const nan = unitEmbedding();
    nan[3] = Number.NaN;
    expect(toValidEmbedding(nan)).toBeNull();
    // Finite in f64 but overflows to Infinity in f32.
    const overflow = unitEmbedding();
    overflow[3] = 1e40;
    expect(toValidEmbedding(overflow)).toBeNull();
  });

  test("rejects zero and unnormalized vectors", () => {
    expect(toValidEmbedding(new Array(MODEL_EMBED_DIM).fill(0))).toBeNull();
    const scaled = unitEmbedding();
    scaled[0] = 2; // norm 2
    expect(toValidEmbedding(scaled)).toBeNull();
  });

  test("rejects non-array and non-number entries", () => {
    expect(toValidEmbedding("nope")).toBeNull();
    const bad = unitEmbedding();
    bad[4] = "x" as unknown as number;
    expect(toValidEmbedding(bad)).toBeNull();
  });
});

describe("toPreviewFaces", () => {
  test("keeps valid faces and drops malformed embeddings", () => {
    const faces = toPreviewFaces([
      { url: "u1", filename: "a.jpg", source: "commons", score: 0.9, embedding: unitEmbedding() },
      { url: "u2", filename: "b.jpg", source: "commons", score: 0.8, embedding: [1, 2, 3] },
      "garbage",
      { url: "u3", embedding: unitEmbedding(2) },
    ]);
    expect(faces.length).toBe(2);
    expect(faces[0]!.url).toBe("u1");
    expect(faces[1]!.url).toBe("u3");
  });

  test("returns [] for non-array input", () => {
    expect(toPreviewFaces(null)).toEqual([]);
    expect(toPreviewFaces({})).toEqual([]);
  });
});

describe("toBlockedIdentities", () => {
  test("one malformed identity does not poison valid matching", () => {
    const { identities, droppedIdentities } = toBlockedIdentities([
      identity({ id: "good" }),
      { id: "bad", name: "Bad", embeddings: [[1, 2, 3]], threshold: 0.4 },
      { id: "mixed", name: "Mixed", embeddings: [unitEmbedding(), [0]], threshold: 0.4 },
    ]);
    expect(identities.map((i) => i.id)).toEqual(["good", "mixed"]);
    expect(droppedIdentities).toBe(1);
    // The malformed vector inside "mixed" is dropped, the valid one survives.
    expect(identities[1]!.embeddings.length).toBe(1);
  });

  test("non-array input yields empty", () => {
    expect(toBlockedIdentities(undefined).identities).toEqual([]);
    expect(toBlockedIdentities(undefined).droppedIdentities).toBe(0);
  });
});

describe("selectPreviewIdentityId", () => {
  test("explicit id must exist — fails closed otherwise", () => {
    const state = stateWith(identity());
    expect(selectPreviewIdentityId(state, { explicitId: "ghost", name: "X" }).error).toBeTruthy();
    expect(
      selectPreviewIdentityId(state, { explicitId: "theo-browne", name: "X" }).identityId,
    ).toBe("theo-browne");
  });

  test("canonical id wins when it is already saved", () => {
    const state = stateWith(identity());
    expect(
      selectPreviewIdentityId(state, { canonicalId: "theo-browne", name: "theo" }).identityId,
    ).toBe("theo-browne");
  });

  test("canonical id that is not saved falls through to slug/name matching", () => {
    const state = stateWith(identity({ id: "theo-browne" }));
    // canonicalId "custom-id" not saved; slug of "Theo Browne" hits.
    expect(
      selectPreviewIdentityId(state, { canonicalId: "custom-id", name: "Theo Browne" }).identityId,
    ).toBe("theo-browne");
  });

  test("alias-typed name does not duplicate a curated identity", () => {
    const state = stateWith(identity());
    expect(selectPreviewIdentityId(state, { name: "Theo Browne" }).identityId).toBe("theo-browne");
  });

  test("new person yields no identityId", () => {
    const state = stateWith(identity());
    expect(selectPreviewIdentityId(state, { name: "Someone Else" }).identityId).toBeUndefined();
  });
});

describe("mergeConfirmedIdentity", () => {
  test("appends a genuinely new identity", () => {
    const state = stateWith(identity());
    const fresh = identity({ id: "new-person", name: "New Person", threshold: 0.4 });
    const next = mergeConfirmedIdentity(state, fresh);
    expect(next.identities.length).toBe(2);
    expect(next.identities[1]!.id).toBe("new-person");
  });

  test("refresh replaces references but preserves id, stored threshold, createdAt", () => {
    const existing = identity({ threshold: 0.61, createdAt: 42 });
    const state = stateWith(existing);
    const confirmed = identity({
      name: "Theo Browne",
      embeddings: [unitEmbedding(3), unitEmbedding(4)],
      threshold: 0.4, // the default the offscreen document stamps
      sources: ["new-a", "new-b"],
      createdAt: 9999,
    });
    const next = mergeConfirmedIdentity(state, confirmed);
    expect(next.identities.length).toBe(1);
    const merged = next.identities[0]!;
    expect(merged.id).toBe("theo-browne");
    expect(merged.threshold).toBe(0.61); // stored operating point survives
    expect(merged.createdAt).toBe(42);
    expect(merged.embeddings.length).toBe(2);
    expect(merged.sources).toEqual(["new-a", "new-b"]);
  });
});

describe("normalizeName", () => {
  test("collapses whitespace and case", () => {
    expect(normalizeName("  Theo   Browne ")).toBe("theo browne");
  });
});
