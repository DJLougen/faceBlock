import { beforeAll, describe, expect, test } from "bun:test";
import { MODEL_EMBED_DIM } from "../../extension/enroll.ts";
import type { BlockList, EnrollPreview, SavedIdentity } from "../../extension/protocol.ts";

/**
 * Drives the real background.ts message handler with a stubbed chrome API.
 * The offscreen document is replaced by a per-test handler so the contract —
 * preview-before-save, refresh merge, fail-closed identityId — is exercised
 * end to end through the actual listener.
 */

type Message = Record<string, unknown>;
type Listener = (
  message: unknown,
  sender: { id?: string; url?: string; tab?: { id?: number; url?: string } },
  sendResponse: (r: unknown) => void,
) => boolean | undefined;

const EXT_ID = "test-ext";
const PAGE_URL = `chrome-extension://${EXT_ID}/options.html`;

let store: Record<string, unknown> = {};
let listener: Listener | null = null;
let offscreenHandler: (msg: Message) => Promise<Message> = async () => ({
  ok: false,
  error: "no offscreen handler installed",
});

function unitEmbedding(seed = 1): number[] {
  const v = new Array<number>(MODEL_EMBED_DIM).fill(0);
  v[seed % MODEL_EMBED_DIM] = 1;
  return v;
}

function previewFor(name: string, identityId?: string): EnrollPreview {
  return {
    name,
    identityId,
    candidatesTried: 2,
    facesFound: 1,
    kept: [
      {
        url: "https://upload.wikimedia.org/x/face.jpg",
        filename: "face.jpg",
        source: "commons",
        score: 0.9,
        embedding: unitEmbedding(),
      },
    ],
    rejected: [],
  };
}

function send(message: Message, fromPage = true): Promise<Message> {
  const sender = fromPage
    ? { id: EXT_ID, url: PAGE_URL }
    : { id: EXT_ID, url: "https://example.com/page", tab: { id: 1 } };
  const { promise, resolve } = Promise.withResolvers<Message>();
  const keep = listener!(message, sender, (r: unknown) => resolve(r as Message));
  expect(keep).toBe(true);
  return promise;
}

function savedIdentities(): SavedIdentity[] {
  const state = store["faceblockState"] as BlockList | undefined;
  return state?.identities ?? [];
}

beforeAll(async () => {
  (globalThis as Record<string, unknown>)["chrome"] = {
    storage: {
      local: {
        get: async (key: string) => ({ [key]: store[key] }),
        set: async (items: Record<string, unknown>) => {
          Object.assign(store, items);
        },
      },
    },
    offscreen: {
      hasDocument: async () => true,
      createDocument: async () => undefined,
      closeDocument: async () => undefined,
    },
    runtime: {
      id: EXT_ID,
      onMessage: {
        addListener(cb: Listener) {
          listener = cb;
        },
      },
      sendMessage: async (msg: unknown) => offscreenHandler(msg as Message),
    },
    tabs: {
      query: async () => [],
      sendMessage: async () => undefined,
    },
  };
  // Dynamic import is required: the chrome stub must exist before
  // background.ts evaluates, since the module registers its listener at load.
  await import("../../extension/background.ts");
});

describe("enrollment contract", () => {
  test("RESOLVE_PREVIEW returns a preview and never persists", async () => {
    offscreenHandler = async (msg) =>
      msg.type === "RESOLVE_PREVIEW" ? { ok: true, preview: previewFor("Ada Lovelace") } : {};
    const res = await send({ target: "background", type: "RESOLVE_PREVIEW", name: "Ada Lovelace" });
    expect(res.ok).toBe(true);
    expect(res.preview).toBeTruthy();
    expect(savedIdentities().some((i) => i.name === "Ada Lovelace")).toBe(false);
  });

  test("BLOCK_NAME is the legacy preview alias — it does not save", async () => {
    offscreenHandler = async (msg) =>
      msg.type === "RESOLVE_PREVIEW" ? { ok: true, preview: previewFor("Grace Hopper") } : {};
    const res = await send({ target: "background", type: "BLOCK_NAME", name: "Grace Hopper" });
    expect(res.ok).toBe(true);
    expect(res.preview).toBeTruthy();
    expect(savedIdentities().some((i) => i.name === "Grace Hopper")).toBe(false);
  });

  test("RESOLVE_PREVIEW with a nonexistent explicit identityId fails closed", async () => {
    let offscreenCalls = 0;
    offscreenHandler = async () => {
      offscreenCalls += 1;
      return { ok: true, preview: previewFor("X") };
    };
    const res = await send({
      target: "background",
      type: "RESOLVE_PREVIEW",
      name: "X",
      identityId: "ghost-id",
    });
    expect(res.ok).toBe(false);
    expect(offscreenCalls).toBe(0); // rejected before the offscreen round-trip
  });

  test("a preview for an existing name carries its identityId for refresh", async () => {
    // Seed a saved identity via a first confirm.
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        return {
          ok: true,
          identity: {
            id: "katherine-johnson",
            name: "Katherine Johnson",
            embeddings: [unitEmbedding()],
            threshold: 0.4,
            sources: ["a"],
            createdAt: 111,
          },
        };
      }
      return { ok: true, preview: previewFor("Katherine Johnson") };
    };
    const confirmed = await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "Katherine Johnson",
      faces: previewFor("Katherine Johnson").kept,
    });
    expect(confirmed.ok).toBe(true);
    const res = await send({
      target: "background",
      type: "RESOLVE_PREVIEW",
      name: "Katherine Johnson",
    });
    expect(res.ok).toBe(true);
    expect((res.preview as EnrollPreview).identityId).toBe("katherine-johnson");
  });

  test("CONFIRM_ENROLL persists only on confirm", async () => {
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        return {
          ok: true,
          identity: {
            id: "new-person",
            name: "New Person",
            embeddings: [unitEmbedding()],
            threshold: 0.4,
            sources: ["s"],
            createdAt: 5,
          },
        };
      }
      return { ok: true, preview: previewFor("New Person") };
    };
    await send({ target: "background", type: "RESOLVE_PREVIEW", name: "New Person" });
    expect(savedIdentities().some((i) => i.id === "new-person")).toBe(false);
    const res = await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "New Person",
      faces: previewFor("New Person").kept,
    });
    expect(res.ok).toBe(true);
    expect(savedIdentities().some((i) => i.id === "new-person")).toBe(true);
  });

  test("CONFIRM_ENROLL with a nonexistent explicit identityId fails closed", async () => {
    let offscreenCalls = 0;
    offscreenHandler = async () => {
      offscreenCalls += 1;
      return {};
    };
    const res = await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "X",
      faces: previewFor("X").kept,
      identityId: "ghost-id",
    });
    expect(res.ok).toBe(false);
    expect(offscreenCalls).toBe(0);
  });

  test("refresh preserves the stored threshold and createdAt", async () => {
    // Seed an identity with a customized threshold.
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        return {
          ok: true,
          identity: {
            id: "custom-thresh",
            name: "Custom Thresh",
            embeddings: [unitEmbedding()],
            threshold: 0.4,
            sources: ["a"],
            createdAt: 7,
          },
        };
      }
      return {};
    };
    await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "Custom Thresh",
      faces: previewFor("Custom Thresh").kept,
    });
    // Simulate a user-tuned threshold stored on the identity.
    const state = store["faceblockState"] as BlockList;
    state.identities = state.identities.map((i) =>
      i.id === "custom-thresh" ? { ...i, threshold: 0.66 } : i,
    );

    // Refresh with new faces: offscreen stamps the default threshold, but the
    // stored operating point must survive.
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        return {
          ok: true,
          identity: {
            id: "custom-thresh",
            name: "Custom Thresh",
            embeddings: [unitEmbedding(2), unitEmbedding(3)],
            threshold: 0.4,
            sources: ["b", "c"],
            createdAt: 999,
          },
        };
      }
      return {};
    };
    const res = await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "Custom Thresh",
      faces: previewFor("Custom Thresh").kept,
      identityId: "custom-thresh",
    });
    expect(res.ok).toBe(true);
    const saved = savedIdentities().find((i) => i.id === "custom-thresh")!;
    expect(saved.threshold).toBe(0.66);
    expect(saved.createdAt).toBe(7);
    expect(saved.embeddings.length).toBe(2);
    expect(savedIdentities().filter((i) => i.id === "custom-thresh").length).toBe(1);
  });

  test("a REMOVE that lands during confirm prevents resurrection", async () => {
    // Seed the identity.
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        return {
          ok: true,
          identity: {
            id: "race-victim",
            name: "Race Victim",
            embeddings: [unitEmbedding()],
            threshold: 0.4,
            sources: ["a"],
            createdAt: 1,
          },
        };
      }
      return {};
    };
    await send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "Race Victim",
      faces: previewFor("Race Victim").kept,
    });
    expect(savedIdentities().some((i) => i.id === "race-victim")).toBe(true);

    // Now a refresh whose offscreen leg is gated: REMOVE lands while the
    // confirmation is in flight, and the serialized mutate must refuse to
    // resurrect the deleted identity.
    const gate = Promise.withResolvers<void>();
    offscreenHandler = async (msg) => {
      if (msg.type === "CONFIRM_ENROLL") {
        await gate.promise;
        return {
          ok: true,
          identity: {
            id: "race-victim",
            name: "Race Victim",
            embeddings: [unitEmbedding(5)],
            threshold: 0.4,
            sources: ["b"],
            createdAt: 2,
          },
        };
      }
      return {};
    };
    const confirm = send({
      target: "background",
      type: "CONFIRM_ENROLL",
      name: "Race Victim",
      faces: previewFor("Race Victim").kept,
      identityId: "race-victim",
    });
    const removed = await send({ target: "background", type: "REMOVE", id: "race-victim" });
    expect(removed.ok).toBe(true);
    gate.resolve();
    const res = await confirm;
    expect(res.ok).toBe(false);
    expect(savedIdentities().some((i) => i.id === "race-victim")).toBe(false);
  });

  test("content scripts cannot reach the enrollment routes", async () => {
    const res = await send(
      { target: "background", type: "RESOLVE_PREVIEW", name: "X" },
      false,
    );
    expect(res.ok).toBe(false);
    expect(res.error).toBe("Extension pages only.");
  });

  test("earnEnabled defaults false and SET_EARN_ENABLED persists", async () => {
    const boot = store["faceblockState"] as BlockList;
    expect(boot.earnEnabled).toBe(false);

    const on = await send({ target: "background", type: "SET_EARN_ENABLED", earnEnabled: true });
    expect(on.ok).toBe(true);
    expect((on.state as BlockList).earnEnabled).toBe(true);

    const off = await send({ target: "background", type: "SET_EARN_ENABLED", earnEnabled: false });
    expect(off.ok).toBe(true);
    expect((off.state as BlockList).earnEnabled).toBe(false);
  });


  test("SET_EARN_ENABLED does not bump revision", async () => {
    const before = store["faceblockState"] as BlockList;
    const rev = before.revision;
    const on = await send({ target: "background", type: "SET_EARN_ENABLED", earnEnabled: true });
    expect(on.ok).toBe(true);
    expect((on.state as BlockList).revision).toBe(rev);
    expect((on.state as BlockList).earnEnabled).toBe(true);
    const off = await send({ target: "background", type: "SET_EARN_ENABLED", earnEnabled: false });
    expect((off.state as BlockList).revision).toBe(rev);
  });
  test("content GET_STATE includes earnEnabled but not identities", async () => {
    await send({ target: "background", type: "SET_EARN_ENABLED", earnEnabled: true });
    const res = await send({ target: "background", type: "GET_STATE" }, false);
    expect(res.ok).toBe(true);
    const state = res.state as BlockList;
    expect(state.earnEnabled).toBe(true);
    expect(state.identities).toEqual([]);
  });
});
