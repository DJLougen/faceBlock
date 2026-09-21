/**
 * FaceBlock offscreen document: the only place inference runs in the real
 * extension. Background forwards ANALYZE/ANALYZE_FRAME/RESOLVE_PREVIEW/
 * CONFIRM_ENROLL here; this file owns the YuNet detector and the ArcFace
 * ONNX session as lazy singletons.
 *
 * Two scheduling lanes keep the model serialized without letting the network
 * stall it (see jobqueue.ts): downloads and decodes run in a bounded pool,
 * while detect+embed work is enqueued one unit at a time — so a fresh video
 * frame can run between enrollment candidates, and at most one model
 * execution is ever in flight.
 *
 * All model/runtime assets load via chrome.runtime.getURL — nothing leaves
 * the extension. ANALYZE fetches the target image bytes (read-only download,
 * credentials omitted) and never uploads pixels or embeddings anywhere.
 * ANALYZE_FRAME takes a JPEG the content script sampled from a <video>;
 * frames are decoded, analysed on device, and discarded — never cached,
 * persisted, or uploaded (plan §14/§22).
 *
 * The enrollment flow (RESOLVE_PREVIEW/CONFIRM_ENROLL) transmits ONLY the
 * typed name to Wikimedia endpoints, and only when the name is not in the
 * curated references.json — curated references stay fully local. Downloaded
 * reference photos are decoded, embedded, and discarded in memory — never
 * persisted and never uploaded (plan §11/§14). Only embeddings plus the
 * minimal EnrollPreviewFace metadata survive, and only on confirm.
 */

import { createYuNetDetector, detectFacesYuNet, type YuNetDetector } from "../src/cv/yunet.ts";
import { createEmbedder, embedAligned, type Embedder } from "../src/cv/embedder.ts";
import { alignFace, alignmentSourceRect } from "../src/cv/align.ts";
import { imageToRasterRegion } from "../src/cv/raster.ts";
import { matchFace } from "../src/matching/matcher.ts";
import { cosineNormalized } from "../src/matching/cosine.ts";
import { resolveCandidates } from "../src/resolve/resolve.ts";
import { clusterEmbeddings } from "../src/resolve/cluster.ts";
import type { CandidateImage } from "../src/resolve/types.ts";
import { BOX_SCALE_X, BOX_SCALE_Y, MIN_REFERENCE_IMAGES } from "../src/shared/config.ts";
import type { BlockedIdentity, Box, FaceDetection } from "../src/shared/types.ts";
import type {
  EnrollPreview,
  EnrollPreviewFace,
  FaceDiagnostics,
  FrameResult,
  ImageResult,
  ReferencePerson,
  SavedIdentity,
} from "./protocol.ts";
import {
  MATCH_THRESHOLD,
  identityId,
  normalizeName,
  toBlockedIdentities,
  toPreviewFaces,
} from "./enroll.ts";
import { createJobQueue, createPool } from "./jobqueue.ts";
import { checkImageType, errText, fetchImageBytes, base64ToBytes, MAX_IMAGE_BYTES } from "./net.ts";
import * as ort from "onnxruntime-web/wasm";

/**
 * MV3 extension_pages CSP forbids blob: workers, so ORT cannot spawn its
 * default proxy worker inside the offscreen document. proxy=false runs
 * inference on this document's thread; numThreads=1 (set in createEmbedder)
 * keeps the threaded wasm build from needing pthread workers.
 */
ort.env.wasm.proxy = false;

/** Narrow chrome surface this file uses; keeps the module free of @types/chrome. */
declare const chrome: {
  runtime: {
    id: string;
    getURL(path: string): string;
    onMessage: {
      addListener(
        cb: (
          message: unknown,
          sender: { id?: string; tab?: unknown; url?: string },
          sendResponse: (response: unknown) => void,
        ) => boolean | void,
      ): void;
    };
  };
};

/* ---------- limits ---------- */
const MAX_PIXELS = 12_000_000; // 12 MP decode cap, enforced after decode on every path
const DECODE_TIMEOUT_MS = 30_000;
const JOB_TIMEOUT_MS = 120_000; // hard ceiling per serialized inference job
const MAX_QUEUE_WAIT_MS = 60_000; // queued jobs older than this are dropped
const MAX_QUEUE_PENDING = 32;
const MAX_RESOLVE_FETCH = 48; // candidates actually downloaded per RESOLVE_PREVIEW
const NETWORK_CONCURRENCY = 4; // bounded download+decode pool
const NETWORK_MAX_PENDING = 64; // download waiters beyond this are rejected
const MAX_PREVIEW_PROTOTYPES = 8; // diverse faces offered for confirmation
const DUPLICATE_COSINE = 0.98; // near-identical crops collapse to one face
const MIN_AGREEMENTS = 1; // matchFace agreement count used by ANALYZE paths

/* ---------- scheduling lanes ---------- */

/**
 * The serial model lane: at most one detect/embed unit runs at a time.
 * Frame jobs are prioritized and supersede still-pending frames.
 */
const inferenceQueue = createJobQueue({
  maxWaitMs: MAX_QUEUE_WAIT_MS,
  maxPending: MAX_QUEUE_PENDING,
  jobTimeoutMs: JOB_TIMEOUT_MS,
});

/** The network lane: downloads and decodes never hold the model lane. */
const networkPool = createPool(NETWORK_CONCURRENCY, NETWORK_MAX_PENDING);

/**
 * The candidate lane: bounds how many decoded candidate bitmaps are alive at
 * once. A candidate holds its decoded image from download through its
 * serialized inference unit, so without this cap a 48-candidate resolve
 * could keep dozens of multi-MP bitmaps resident while they wait on the
 * model lane. Live video frames bypass this lane entirely.
 */
const candidatePool = createPool(NETWORK_CONCURRENCY, NETWORK_MAX_PENDING);

/* ---------- small helpers ---------- */
function extUrl(path: string): string {
  return chrome.runtime.getURL(path.replace(/^\/+/, ""));
}

function withTimeout<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  let timer: number | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = window.setTimeout(
      () => reject(new Error(`faceBlock: ${what} timed out after ${Math.round(ms / 1000)}s`)),
      ms,
    );
  });
  return Promise.race([work, timeout]).finally(() => clearTimeout(timer));
}

/* ---------- lazy singleton models ---------- */
let embedderPromise: Promise<Embedder> | null = null;
let yunetPromise: Promise<YuNetDetector> | null = null;

/**
 * YuNet is the only detector on every path, deliberately. Enrolment and
 * matching must agree on landmark conventions: aligning enrolments with one
 * model and queries with another produces differently-cropped 112x112 chips
 * and drops same-person similarity below the match threshold (measured:
 * masks fell from 9 to 6 while detection rose).
 */
function getYuNet(): Promise<YuNetDetector> {
  if (!yunetPromise) {
    yunetPromise = createYuNetDetector(
      extUrl("models/face_detection_yunet_2026may.onnx"),
      extUrl("ort/"),
    ).catch((e) => {
      yunetPromise = null;
      throw e;
    });
  }
  return yunetPromise;
}

function getEmbedder(): Promise<Embedder> {
  if (!embedderPromise) {
    embedderPromise = createEmbedder(
      extUrl("models/w600k_mbf.onnx"),
      extUrl("ort/"),
    ).catch((e) => {
      embedderPromise = null;
      throw e;
    });
  }
  return embedderPromise;
}

/* ---------- image decode ---------- */
function loadImage(url: string): Promise<HTMLImageElement> {
  return withTimeout(
    new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("faceBlock: image failed to decode"));
      img.src = url;
    }),
    DECODE_TIMEOUT_MS,
    "image decode",
  );
}

/**
 * Decode a Blob into an <img> via an object URL. The post-decode pixel cap
 * lives here so EVERY path — fetched photos, bundled references, sampled
 * video frames — enforces the same 12 MP bound. Caller MUST call release()
 * on every path — it revokes the URL and drops the decoded bitmap reference.
 */
async function decodeImage(
  blob: Blob,
  what: string,
): Promise<{ img: HTMLImageElement; release: () => void }> {
  checkImageType(blob.type, what);
  if (blob.size === 0) throw new Error(`faceBlock: ${what} is empty`);
  if (blob.size > MAX_IMAGE_BYTES) {
    throw new Error(`faceBlock: ${what} exceeds the 12 MB limit`);
  }
  const url = URL.createObjectURL(blob);
  let img: HTMLImageElement;
  const release = () => {
    URL.revokeObjectURL(url);
    img.onload = null;
    img.onerror = null;
    img.src = "";
  };
  try {
    img = await loadImage(url);
  } catch (e) {
    URL.revokeObjectURL(url);
    throw new Error(`faceBlock: ${what} is not a decodable image — ${errText(e)}`);
  }
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (w === 0 || h === 0) {
    release();
    throw new Error(`faceBlock: ${what} decoded to an empty image`);
  }
  if (w * h > MAX_PIXELS) {
    release();
    throw new Error(
      `faceBlock: ${what} is ${w}x${h} (${((w * h) / 1e6).toFixed(1)} MP) — over the 12 MP limit`,
    );
  }
  return { img, release };
}

/* ---------- references ---------- */

async function loadReferences(): Promise<ReferencePerson[]> {
  const res = await fetch(extUrl("references.json"));
  if (!res.ok) {
    throw new Error(`faceBlock: cannot load references.json (HTTP ${res.status})`);
  }
  const raw: unknown = await res.json();
  if (!Array.isArray(raw)) {
    throw new Error("faceBlock: references.json is not an array of ReferencePerson");
  }
  return raw as ReferencePerson[];
}

function findPerson(people: ReferencePerson[], name: string): ReferencePerson | null {
  const want = normalizeName(name);
  for (const p of people) {
    if (!p || typeof p !== "object") continue;
    const names = [p.name, ...(Array.isArray(p.aliases) ? p.aliases : [])];
    if (names.some((n) => typeof n === "string" && normalizeName(n) === want)) return p;
  }
  return null;
}

/**
 * Detect faces for alignment. ONE detector on every path — see getYuNet.
 */
async function detectForAlignment(img: HTMLImageElement): Promise<FaceDetection[]> {
  return detectFacesYuNet(await getYuNet(), img);
}

/**
 * Embed one detected face. The raster covers exactly the region alignFace
 * can read (alignmentSourceRect), so the chip is identical to aligning on
 * the full image — verified by tests/cv/region-align.test.ts — while large
 * photos never pay a full-frame getImageData.
 */
async function embedDetectedFace(
  embedder: Embedder,
  img: HTMLImageElement,
  det: FaceDetection,
): Promise<Float32Array> {
  const { raster, offsetX, offsetY } = imageToRasterRegion(
    img,
    alignmentSourceRect(det),
    0,
  );
  const local: FaceDetection = {
    ...det,
    box: { ...det.box, x: det.box.x - offsetX, y: det.box.y - offsetY },
    landmarks: det.landmarks?.map((pt) => ({ x: pt.x - offsetX, y: pt.y - offsetY })),
  };
  return embedAligned(embedder, alignFace(raster, local));
}

/* ---------- enrollment preview ---------- */
interface EmbeddedCandidate {
  candidate: CandidateImage;
  embedding: Float32Array;
}

/**
 * Download + decode one image off the model lane, then enqueue exactly one
 * serialized detect+embed unit. Requires exactly one face: a face picked out
 * of a group photo would poison the identity.
 */
async function prepareCandidate(
  url: string,
  what: string,
): Promise<{ img: HTMLImageElement; release: () => void }> {
  const blob = await networkPool.run(() => fetchImageBytes(url, what));
  return decodeImage(blob, what);
}

async function detectAndEmbedOne(
  img: HTMLImageElement,
  embedder: Embedder,
): Promise<Float32Array> {
  const dets = await detectForAlignment(img);
  if (dets.length !== 1) {
    throw new Error(`${dets.length} faces`);
  }
  const embedding = await embedDetectedFace(embedder, img, dets[0]!);
  if (embedding.length === 0) {
    throw new Error("empty embedding");
  }
  return embedding;
}

/**
 * Curated references.json hit: preview the bundled reference set under the
 * person's canonical name/id. Fully local — no Wikimedia requests — and
 * nothing is persisted; CONFIRM_ENROLL is the only write path.
 */
async function curatedPreview(
  person: ReferencePerson,
  embedder: Embedder,
): Promise<EnrollPreview> {
  const kept: EnrollPreviewFace[] = [];
  const rejected: { url: string; reason: string }[] = [];
  const refs = person.references ?? [];
  let facesFound = 0;
  await Promise.all(
    refs.map(async (ref) => {
      if (!ref || typeof ref.path !== "string") return;
      const url = /^https?:\/\//i.test(ref.path) ? ref.path : extUrl(ref.path);
      const what = `reference "${ref.path}"`;
      // The whole unit — download, decode, inference, release — occupies one
      // candidatePool slot, so resident decoded bitmaps stay bounded.
      await candidatePool.run(async () => {
        let img: HTMLImageElement;
        let release: () => void;
        try {
          ({ img, release } = await prepareCandidate(url, what));
        } catch (e) {
          rejected.push({ url, reason: errText(e) });
          return;
        }
        // `released` resolves when the inference job's cleanup actually ran —
        // including on pre-run rejection or after a caller-facing timeout —
        // so the pool slot stays held while the bitmap is still in use.
        const released = Promise.withResolvers<void>();
        try {
          const embedding = await inferenceQueue.enqueue(
            () => detectAndEmbedOne(img, embedder),
            "inference",
            () => {
              try {
                release();
              } finally {
                released.resolve();
              }
            },
          );
          facesFound++;
          kept.push({
            url,
            thumbUrl: url,
            filename: ref.path.split("/").pop() ?? ref.path,
            source: typeof ref.source === "string" ? ref.source : "curated",
            score: 1,
            embedding: Array.from(embedding),
          });
        } catch (e) {
          rejected.push({ url, reason: errText(e) });
        } finally {
          await released.promise;
        }
      });
    }),
  );
  return {
    name: person.name,
    identityId: person.id,
    candidatesTried: refs.length,
    facesFound,
    kept,
    rejected,
  };
}

/** At most this many RESOLVE_PREVIEW resolutions run concurrently. */
const MAX_CONCURRENT_PREVIEWS = 2;
let previewInFlight = 0;

/**
 * Bounded admission for the preview path: each resolution fans out into the
 * network pool and the inference queue, so a flood of requests is rejected
 * rather than stacked.
 */
async function resolvePreview(name: unknown): Promise<{ preview: EnrollPreview }> {
  if (previewInFlight >= MAX_CONCURRENT_PREVIEWS) {
    throw new Error("faceBlock: another enrollment preview is already in progress — try again shortly");
  }
  previewInFlight += 1;
  // The caller-facing deadline wraps the work; admission and resource slots
  // stay held until the work ACTUALLY settles — a timed-out preview may still
  // hold candidate bitmaps while its inference jobs drain.
  const work = buildPreview(name).finally(() => {
    previewInFlight -= 1;
  });
  return await withTimeout(work, JOB_TIMEOUT_MS, "enrollment preview");
}

/**
 * Resolve a typed name to candidate photos, embed the usable ones, and
 * return a diverse preview set for user confirmation. Nothing is persisted
 * here — embeddings stay in this document until CONFIRM_ENROLL.
 */
async function buildPreview(name: unknown): Promise<{ preview: EnrollPreview }> {
  if (typeof name !== "string" || name.trim() === "") {
    throw new Error('faceBlock: RESOLVE_PREVIEW requires a non-empty "name"');
  }
  const trimmed = name.trim();
  const embedder = await getEmbedder();

  // Curated directory first: a references.json hit previews the bundled set
  // under its canonical name/id and never touches the network beyond the
  // bundled (or explicitly http(s)) reference files.
  const person = findPerson(await loadReferences(), trimmed);
  if (person) {
    return { preview: await curatedPreview(person, embedder) };
  }

  const { candidates, timedOut } = await resolveCandidates(trimmed, { limit: MAX_RESOLVE_FETCH });
  const rejected: { url: string; reason: string }[] = [];
  // Deadline honesty: a timed-out search is not proof that no photos exist.
  if (timedOut) {
    rejected.push({
      url: "",
      reason: "photo search hit its deadline — results below may be incomplete",
    });
  }
  if (candidates.length === 0) {
    return {
      preview: {
        name: trimmed,
        candidatesTried: 0,
        facesFound: 0,
        kept: [],
        rejected: timedOut
          ? [
              {
                url: "",
                reason:
                  `photo search for "${trimmed}" timed out before all sources answered — ` +
                  `try again, or use a more specific name`,
              },
            ]
          : [
              {
                url: "",
                reason:
                  `no candidate photos found for "${trimmed}" — ` +
                  `try the person's full name or a common alias`,
              },
            ],
      },
    };
  }

  const embedded: (EmbeddedCandidate | undefined)[] = [];
  let facesFound = 0;
  await Promise.all(
    candidates.map(async (candidate, i) => {
      const what = `candidate "${candidate.filename}"`;
      // The whole unit — download, decode, inference, release — occupies one
      // candidatePool slot, so resident decoded bitmaps stay bounded.
      await candidatePool.run(async () => {
        let img: HTMLImageElement;
        let release: () => void;
        try {
          // Downloads prefer the thumbnail: Commons/Wikipedia originals
          // routinely exceed the 12 MP decode cap, while 640px is ample for a
          // detector that aligns to 112x112.
          ({ img, release } = await prepareCandidate(
            candidate.thumbUrl ?? candidate.url,
            what,
          ));
        } catch (e) {
          // A bad candidate must not abort the whole resolve.
          rejected.push({ url: candidate.url, reason: errText(e) });
          return;
        }
        // `released` resolves when the inference job's cleanup actually ran —
        // including on pre-run rejection or after a caller-facing timeout —
        // so the pool slot stays held while the bitmap is still in use.
        const released = Promise.withResolvers<void>();
        try {
          const embedding = await inferenceQueue.enqueue(
            () => detectAndEmbedOne(img, embedder),
            "inference",
            () => {
              try {
                release();
              } finally {
                released.resolve();
              }
            },
          );
          embedded[i] = { candidate, embedding };
          facesFound++;
        } catch (e) {
          rejected.push({ url: candidate.url, reason: errText(e) });
        } finally {
          await released.promise;
        }
      });
    }),
  );

  // Drop near-duplicates before clustering: Commons often hosts a portrait
  // and a slightly cropped copy of it, which would waste prototype slots.
  const unique: EmbeddedCandidate[] = [];
  for (const item of embedded) {
    if (!item) continue;
    if (unique.some((u) => cosineNormalized(u.embedding, item.embedding) >= DUPLICATE_COSINE)) {
      rejected.push({ url: item.candidate.url, reason: "duplicate" });
      continue;
    }
    unique.push(item);
  }

  const cluster = clusterEmbeddings(
    unique.map((u) => u.embedding),
    { maxPrototypes: MAX_PREVIEW_PROTOTYPES },
  );
  for (const rej of cluster.rejected) {
    rejected.push({ url: unique[rej.index]?.candidate.url ?? "", reason: rej.reason });
  }
  // kept is the cluster's prototypes, not every kept embedding, so the
  // stored set spans poses and lighting rather than the first usable photos.
  const kept: EnrollPreviewFace[] = [];
  for (const idx of cluster.prototypes) {
    const item = unique[idx];
    if (!item) continue;
    kept.push({
      url: item.candidate.url,
      thumbUrl: item.candidate.thumbUrl,
      filename: item.candidate.filename,
      source: item.candidate.source,
      score: item.candidate.score,
      embedding: Array.from(item.embedding),
    });
  }
  if (kept.length < MIN_REFERENCE_IMAGES) {
    rejected.push({
      url: "",
      reason:
        `too-few-consistent-faces: only ${kept.length} distinct face(s) survived ` +
        `dedup/clustering (need ${MIN_REFERENCE_IMAGES}) — confirm only if these look right`,
    });
  }
  return {
    preview: {
      name: trimmed,
      candidatesTried: candidates.length,
      facesFound,
      kept,
      rejected,
    },
  };
}

/**
 * Turn the user's confirmed preview faces into a SavedIdentity. An explicit
 * identityId is the refresh path — background has already verified it names
 * a saved identity, and it is used verbatim. Otherwise the id is the
 * canonical references.json id for curated names, else the name's slug.
 */
async function confirmEnroll(
  name: unknown,
  rawFaces: unknown,
  explicitId: unknown,
): Promise<{ identity: SavedIdentity }> {
  if (typeof name !== "string" || name.trim() === "") {
    throw new Error('faceBlock: CONFIRM_ENROLL requires a non-empty "name"');
  }
  const faces = toPreviewFaces(rawFaces);
  if (faces.length === 0) {
    throw new Error(
      'faceBlock: CONFIRM_ENROLL requires "faces" with at least one valid embedding',
    );
  }
  const trimmed = name.trim();
  let id: string;
  if (typeof explicitId === "string" && explicitId !== "") {
    id = explicitId;
  } else {
    const person = findPerson(await loadReferences(), trimmed);
    id = person?.id ?? identityId(trimmed);
  }
  return {
    identity: {
      id,
      name: trimmed,
      embeddings: faces.map((f) => f.embedding),
      threshold: MATCH_THRESHOLD,
      sources: faces.map((f) => f.url),
      createdAt: Date.now(),
    },
  };
}

/* ---------- analyze ---------- */

/** Grow a detection box by the configured scale around its center, clamped to the image. */
function expandBox(box: Box, w: number, h: number): Box {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const hw = (box.width * BOX_SCALE_X) / 2;
  const hh = (box.height * BOX_SCALE_Y) / 2;
  const x0 = Math.max(0, Math.min(cx - hw, w));
  const y0 = Math.max(0, Math.min(cy - hh, h));
  const x1 = Math.max(0, Math.min(cx + hw, w));
  const y1 = Math.max(0, Math.min(cy + hh, h));
  return { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
}

/** Unthresholded best cosine over every gallery vector, for diagnostics. */
function bestCosine(
  embedding: Float32Array,
  identities: readonly BlockedIdentity[],
): { bestCosine: number | null; bestIdentityId: string | null } {
  let best: number | null = null;
  let bestId: string | null = null;
  for (const identity of identities) {
    const gallery =
      identity.prototypes && identity.prototypes.length > 0
        ? identity.prototypes
        : identity.embeddings;
    for (const g of gallery) {
      const s = cosineNormalized(embedding, g);
      if (best === null || s > best) {
        best = s;
        bestId = identity.id;
      }
    }
  }
  return { bestCosine: best, bestIdentityId: bestId };
}

interface AnalysisOutcome {
  regions: { x: number; y: number; width: number; height: number; confidence: number; identityId: string }[];
  dets: FaceDetection[];
  diagFaces: FaceDiagnostics["faces"];
  droppedIdentities: number;
  degradedIdentities: number;
  width: number;
  height: number;
}

/**
 * The serialized inference unit shared by ANALYZE and ANALYZE_FRAME:
 * detect, embed each face, match. Diagnostics are computed only for
 * extension-page senders and carry numbers, never embeddings.
 */
async function analyzeDecoded(
  img: HTMLImageElement,
  rawIdentities: unknown,
  wantDiagnostics: boolean,
): Promise<AnalysisOutcome> {
  const { identities, droppedIdentities, degradedIdentities } = toBlockedIdentities(rawIdentities);
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const dets = await detectForAlignment(img);
  const regions: AnalysisOutcome["regions"] = [];
  const diagFaces: FaceDiagnostics["faces"] = [];
  if (dets.length > 0 && identities.length > 0) {
    const embedder = await getEmbedder();
    for (const det of dets) {
      const embedding = await embedDetectedFace(embedder, img, det);
      const match = matchFace(embedding, identities, { minAgreements: MIN_AGREEMENTS });
      if (match) {
        regions.push({
          ...expandBox(det.box, w, h),
          confidence: match.score,
          identityId: match.identityId,
        });
      }
      if (wantDiagnostics) {
        const best = bestCosine(embedding, identities);
        diagFaces.push({
          box: { ...det.box },
          bestCosine: best.bestCosine,
          bestIdentityId: best.bestIdentityId,
          matched: match !== null,
          matchedIdentityId: match?.identityId ?? null,
        });
      }
    }
  } else if (wantDiagnostics) {
    // No gallery to score against: report the detections with null cosines
    // rather than paying for embeddings that cannot match anything.
    for (const det of dets) {
      diagFaces.push({
        box: { ...det.box },
        bestCosine: null,
        bestIdentityId: null,
        matched: false,
        matchedIdentityId: null,
      });
    }
  }
  return { regions, dets, diagFaces, droppedIdentities, degradedIdentities, width: w, height: h };
}

function makeDiagnostics(outcome: AnalysisOutcome): FaceDiagnostics {
  return {
    faceCount: outcome.dets.length,
    threshold: MATCH_THRESHOLD,
    minAgreements: MIN_AGREEMENTS,
    droppedIdentities: outcome.droppedIdentities,
    degradedIdentities: outcome.degradedIdentities,
    faces: outcome.diagFaces,
  };
}

async function analyze(
  url: unknown,
  rawIdentities: unknown,
  wantDiagnostics: boolean,
): Promise<{ result: ImageResult }> {
  if (typeof url !== "string" || !/^https?:\/\//i.test(url)) {
    throw new Error('faceBlock: ANALYZE requires an http(s) "url"');
  }
  // Download + decode on the network lane; only detect+embed holds the
  // serialized model lane.
  const { img, release } = await prepareCandidate(url, `image "${url}"`);
  // release is the job's cleanup: it fires after the inference unit actually
  // settles, never while the model is still reading pixels.
  const outcome = await inferenceQueue.enqueue(
    () => analyzeDecoded(img, rawIdentities, wantDiagnostics),
    "inference",
    release,
  );
  const result: ImageResult = {
    width: outcome.width,
    height: outcome.height,
    faceCount: outcome.dets.length,
    regions: outcome.regions,
  };
  if (wantDiagnostics) result.diagnostics = makeDiagnostics(outcome);
  return { result };
}

/**
 * Analyse one sampled video frame. The JPEG arrives base64-encoded from the
 * content script (runtime messaging is JSON, not structured clone) — unlike
 * ANALYZE nothing is fetched; the frame is decoded, matched, and released.
 * Regions use the DECODED frame's pixel space; dimensions reported by the
 * content script are never trusted.
 */
async function analyzeFrame(
  jpegBase64: unknown,
  rawIdentities: unknown,
  wantDiagnostics: boolean,
): Promise<{ result: FrameResult }> {
  // The frame arrives base64-encoded: runtime.sendMessage JSON-serializes its
  // payload, so raw bytes would be lost in transit. The string is capped
  // BEFORE decoding so an oversized payload never becomes a decoded blob.
  const bytes = base64ToBytes(jpegBase64);
  if (!bytes || bytes.length === 0) {
    throw new Error(
      'faceBlock: ANALYZE_FRAME requires a non-empty "jpegBase64" string within the 12 MB cap',
    );
  }
  const { img, release } = await decodeImage(
    new Blob([bytes], { type: "image/jpeg" }),
    "video frame",
  );
  // release is the job's cleanup: it fires after the frame's inference unit
  // actually settles, never while the model is still reading pixels.
  const outcome = await inferenceQueue.enqueue(
    () => analyzeDecoded(img, rawIdentities, wantDiagnostics),
    "frame",
    release,
  );
  const result: FrameResult = {
    width: outcome.width,
    height: outcome.height,
    regions: outcome.regions,
  };
  if (wantDiagnostics) result.diagnostics = makeDiagnostics(outcome);
  return { result };
}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Only same-extension callers: the background service worker and extension
  // pages. Authenticate on the sender URL's origin — extension pages opened
  // in tabs still carry sender.tab, so the tab check would reject them,
  // while a content script's sender.url is the web page it runs on.
  let senderUrl: URL | null = null;
  try {
    senderUrl = typeof sender.url === "string" ? new URL(sender.url) : null;
  } catch {
    senderUrl = null;
  }
  const fromExtension =
    sender.id === chrome.runtime.id &&
    // The service worker has no tab; an extension page in a tab carries a
    // chrome-extension sender.url; a content script's sender.url is the web
    // page it runs on, so it fails both halves.
    (!sender.tab ||
      (senderUrl !== null &&
        senderUrl.protocol === "chrome-extension:" &&
        senderUrl.host === chrome.runtime.id));
  if (!fromExtension) return false;

  if (!message || typeof message !== "object") return false;
  const m = message as {
    target?: string;
    type?: string;
    name?: string;
    url?: string;
    jpegBase64?: string;
    identities?: SavedIdentity[];
    faces?: EnrollPreviewFace[];
    identityId?: string;
  };
  if (m.target !== "offscreen") return false;
  // Diagnostics are attached only for direct extension-PAGE senders (the
  // options/verify pages end in .html). The background service worker's
  // forwards never get them, so nothing diagnostic reaches content scripts.
  const wantDiagnostics = senderUrl !== null && senderUrl.pathname.endsWith(".html");
  let work: Promise<
    | { identity: SavedIdentity }
    | { result: ImageResult }
    | { result: FrameResult }
    | { preview: EnrollPreview }
  >;
  if (m.type === "ANALYZE") {
    work = analyze(m.url, m.identities, wantDiagnostics);
  } else if (m.type === "ANALYZE_FRAME") {
    work = analyzeFrame(m.jpegBase64, m.identities, wantDiagnostics);
  } else if (m.type === "RESOLVE_PREVIEW") {
    work = resolvePreview(m.name);
  } else if (m.type === "CONFIRM_ENROLL") {
    work = confirmEnroll(m.name, m.faces, m.identityId);
  } else {
    sendResponse({
      ok: false,
      error: `faceBlock: unknown offscreen message type "${String(m.type)}"`,
    });
    return false;
  }
  work
    .then((payload) => sendResponse({ ok: true, ...payload }))
    .catch((e) => sendResponse({ ok: false, error: errText(e) }));
  return true; // keep the channel open for the async response
});
