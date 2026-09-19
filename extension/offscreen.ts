/**
 * FaceBlock offscreen document: the only place inference runs in the real
 * extension. Background forwards ENROLL/ANALYZE here; this file owns the
 * MediaPipe landmarker and the ArcFace ONNX session as lazy singletons and
 * serializes every job through one bounded queue.
 *
 * All model/runtime assets load via chrome.runtime.getURL — nothing leaves
 * the extension. ANALYZE fetches the target image bytes (read-only download,
 * credentials omitted) and never uploads pixels or embeddings anywhere.
 * ANALYZE_FRAME takes a JPEG the content script sampled from a <video>;
 * frames are decoded, analysed on device, and discarded — never cached,
 * persisted, or uploaded (plan §14/§22).
 *
 * The self-seed flow (RESOLVE_PREVIEW/CONFIRM_ENROLL) transmits ONLY the
 * typed name to Wikimedia endpoints to find candidate photos. Downloaded
 * reference photos are decoded, embedded, and discarded in memory — never
 * persisted and never uploaded (plan §11/§14). Only embeddings plus the
 * minimal EnrollPreviewFace metadata survive, and only on confirm.
 */

import { createYuNetDetector, detectFacesYuNet, type YuNetDetector } from "../src/cv/yunet.ts";
import { createEmbedder, embedAligned, type Embedder } from "../src/cv/embedder.ts";
import { alignFace } from "../src/cv/align.ts";
import { imageToRaster, imageToRasterRegion } from "../src/cv/raster.ts";
import { matchFace } from "../src/matching/matcher.ts";
import { cosineNormalized } from "../src/matching/cosine.ts";
import { resolveCandidates } from "../src/resolve/resolve.ts";
import { clusterEmbeddings } from "../src/resolve/cluster.ts";
import type { CandidateImage } from "../src/resolve/types.ts";
import { BOX_SCALE_X, BOX_SCALE_Y, MIN_REFERENCE_IMAGES } from "../src/shared/config.ts";
import type { BlockedIdentity, Box, FaceDetection } from "../src/shared/types.ts";
import type { EnrollPreview, EnrollPreviewFace, FrameResult, ImageResult, ReferencePerson, SavedIdentity } from "./protocol.ts";
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
          sender: { id?: string; tab?: unknown },
          sendResponse: (response: unknown) => void,
        ) => boolean | void,
      ): void;
    };
  };
};

/* ---------- limits ---------- */
const MAX_IMAGE_BYTES = 12 * 1024 * 1024; // 12 MiB streamed cap
const MAX_PIXELS = 12_000_000; // 12 MP decode cap, enforced before canvas
const FETCH_TIMEOUT_MS = 15_000;
const DECODE_TIMEOUT_MS = 30_000;
const JOB_TIMEOUT_MS = 120_000; // hard ceiling per ENROLL/ANALYZE job
const MAX_QUEUE_WAIT_MS = 60_000; // queued jobs older than this are dropped
const MAX_QUEUE_PENDING = 32;
const MAX_RESOLVE_FETCH = 48; // candidates actually downloaded per RESOLVE_PREVIEW
const RESOLVE_FETCH_CONCURRENCY = 4; // bounded download+embed worker pool
const MAX_PREVIEW_PROTOTYPES = 8; // diverse faces offered for confirmation
const DUPLICATE_COSINE = 0.98; // near-identical crops collapse to one face

/**
 * Experimental operating point for the real w600k_mbf model. NOT calibrated —
 * false negatives are expected; see extension/protocol.ts contract.
 */
const MATCH_THRESHOLD = 0.4;

/* ---------- small helpers ---------- */
function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

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
 * YuNet is the DETECTOR. It replaces the landmarker's detection role because
 * that model is tuned for near-frontal faces and returned nothing for profile
 * views or faces that are small within a large photo — misses that happen
 * before matching, so no similarity filtering can recover them.
 *
 * The landmarker is retained only as a fallback: if YuNet fails to load or
 * finds nothing, detection still works rather than silently disabling the
 * extension.
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

/* ---------- bounded serial queue ---------- */
interface Job {
  run: () => Promise<unknown>;
  resolve: (v: unknown) => void;
  reject: (e: unknown) => void;
  enqueuedAt: number;
}

const queue: Job[] = [];
let pumping = false;

function enqueue<T>(run: () => Promise<T>): Promise<T> {
  const now = Date.now();
  while (queue.length > 0 && now - queue[0]!.enqueuedAt > MAX_QUEUE_WAIT_MS) {
    queue.shift()!.reject(
      new Error("faceBlock: request expired waiting in the inference queue"),
    );
  }
  if (queue.length >= MAX_QUEUE_PENDING) {
    return Promise.reject(
      new Error(
        `faceBlock: inference queue full (${MAX_QUEUE_PENDING} pending) — try again shortly`,
      ),
    );
  }
  return new Promise<T>((resolve, reject) => {
    queue.push({
      run: run as () => Promise<unknown>,
      resolve: resolve as (v: unknown) => void,
      reject,
      enqueuedAt: now,
    });
    void pump();
  });
}

async function pump(): Promise<void> {
  if (pumping) return;
  pumping = true;
  try {
    while (queue.length > 0) {
      const job = queue.shift()!;
      if (Date.now() - job.enqueuedAt > MAX_QUEUE_WAIT_MS) {
        job.reject(new Error("faceBlock: request expired waiting in the inference queue"));
        continue;
      }
      // Serialization guard: the caller is rejected at JOB_TIMEOUT_MS, but the
      // queue stays locked until the underlying run() actually settles — ORT
      // sessions and the landmarker must never overlap.
      const work = job.run();
      const settled = work.then(
        () => undefined,
        () => undefined,
      );
      try {
        job.resolve(await withTimeout(work, JOB_TIMEOUT_MS, "inference job"));
      } catch (e) {
        job.reject(e);
      }
      await settled;
    }
  } finally {
    pumping = false;
  }
}

/* ---------- image fetch + decode ---------- */
function checkImageType(type: string, what: string): void {
  const t = type.split(";")[0]!.trim().toLowerCase();
  if (t === "image/svg+xml") {
    throw new Error(
      `faceBlock: ${what} is SVG — refused because SVG can pull external resources`,
    );
  }
  if (t && t !== "application/octet-stream" && !t.startsWith("image/")) {
    throw new Error(`faceBlock: ${what} has unsupported media type "${t}"`);
  }
}

/** Fetch image bytes with no credentials, an abort timeout, and a streamed 12 MiB cap. */
async function fetchImageBytes(url: string, what: string): Promise<Blob> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    let res: Response;
    try {
      res = await fetch(url, {
        credentials: "omit",
        redirect: "follow",
        signal: ctrl.signal,
      });
    } catch (e) {
      throw new Error(
        `faceBlock: fetch of ${what} failed — ${ctrl.signal.aborted ? "timed out" : errText(e)}`,
      );
    }
    if (!res.ok) {
      throw new Error(`faceBlock: fetch of ${what} returned HTTP ${res.status}`);
    }
    checkImageType(res.headers.get("content-type") ?? "", what);
    const declared = Number(res.headers.get("content-length") ?? 0);
    if (declared > MAX_IMAGE_BYTES) {
      throw new Error(
        `faceBlock: ${what} is ${(declared / 1048576).toFixed(1)} MB — over the 12 MB limit`,
      );
    }
    const type = (res.headers.get("content-type") ?? "").split(";")[0]!.trim();
    if (!res.body) {
      const blob = await res.blob();
      if (blob.size > MAX_IMAGE_BYTES) {
        throw new Error(`faceBlock: ${what} exceeds the 12 MB limit`);
      }
      return blob;
    }
    const reader = res.body.getReader();
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let total = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_IMAGE_BYTES) {
        await reader.cancel().catch(() => undefined);
        throw new Error(`faceBlock: ${what} exceeds the 12 MB limit (streamed)`);
      }
      chunks.push(new Uint8Array(value));
    }
    return new Blob(chunks, { type: type || "application/octet-stream" });
  } finally {
    clearTimeout(timer);
  }
}

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
 * Decode a Blob into an <img> via an object URL. Caller MUST call release()
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
  if (img.naturalWidth === 0 || img.naturalHeight === 0) {
    release();
    throw new Error(`faceBlock: ${what} decoded to an empty image`);
  }
  return { img, release };
}

/* ---------- references ---------- */
function normalizeName(s: string): string {
  return s.trim().replace(/\s+/g, " ").toLowerCase();
}

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
 * Embed exactly one face from one reference image. References with zero or
 * multiple faces are rejected — enrolling a face picked out of a group photo
 * would poison the identity.
 */
/**
 * Detect faces for alignment.
 *
 * ONE detector on every path, deliberately. Enrolment and matching must agree
 * on landmark conventions: aligning enrolments with one model and queries with
 * another produces differently-cropped 112x112 chips and drops same-person
 * similarity below the match threshold (measured: masks fell from 9 to 6 while
 * detection rose).
 */
async function detectForAlignment(img: HTMLImageElement): Promise<FaceDetection[]> {
  return detectFacesYuNet(await getYuNet(), img);
}

async function embedReference(
  refPath: string,
  embedder: Embedder,
): Promise<Float32Array> {
  const url = /^https?:\/\//i.test(refPath) ? refPath : extUrl(refPath);
  const blob = await fetchImageBytes(url, `reference "${refPath}"`);
  const { img, release } = await decodeImage(blob, `reference "${refPath}"`);
  try {
    const dets = await detectForAlignment(img);
    if (dets.length !== 1) {
      throw new Error(
        `faceBlock: reference "${refPath}" has ${dets.length} faces — ` +
          `a reference must contain exactly one face`,
      );
    }
    const raster = imageToRaster(img);
    const embedding = await embedAligned(embedder, alignFace(raster, dets[0]!));
    if (embedding.length === 0) {
      throw new Error(`faceBlock: reference "${refPath}" produced an empty embedding`);
    }
    return embedding;
  } finally {
    release();
  }
}

async function enroll(name: unknown): Promise<{ identity: SavedIdentity }> {
  if (typeof name !== "string" || name.trim() === "") {
    throw new Error('faceBlock: ENROLL requires a non-empty "name"');
  }
  const people = await loadReferences();
  const person = findPerson(people, name);
  if (!person) {
    const known = people
      .map((p) => `${p.name} (aliases: ${p.aliases?.join(", ") || "none"})`)
      .join("; ");
    throw new Error(
      `faceBlock: no reference set matches "${name.trim()}". ` +
        `Enrollable people: ${known || "none"}. ` +
        `To block someone else, add them to extension/references.json first.`,
    );
  }
  const embedder = await getEmbedder();
  const embeddings: number[][] = [];
  const sources: string[] = [];
  const failures: string[] = [];
  for (const ref of person.references ?? []) {
    if (!ref || typeof ref.path !== "string") continue;
    try {
      const emb = await embedReference(ref.path, embedder);
      embeddings.push(Array.from(emb));
      if (typeof ref.source === "string") sources.push(ref.source);
    } catch (e) {
      failures.push(`${ref.path}: ${errText(e)}`);
    }
  }
  if (embeddings.length === 0) {
    throw new Error(
      `faceBlock: could not enroll "${person.name}" — no usable reference image. ` +
        failures.join(" | "),
    );
  }
  return {
    identity: {
      id: person.id,
      name: person.name,
      embeddings,
      threshold: MATCH_THRESHOLD,
      sources,
      createdAt: Date.now(),
    },
  };
}

/* ---------- self-seed resolve ---------- */
interface EmbeddedCandidate {
  candidate: CandidateImage;
  embedding: Float32Array;
}

/**
 * Fetch and embed one resolved candidate. Downloads the thumbnail when the
 * source offers one — Commons/Wikipedia originals routinely exceed the
 * 12 MP decode cap, while 640px is ample for a detector that aligns to
 * 112x112. Requires exactly one face for the same reason embedReference
 * does: a face picked out of a group photo would poison the identity.
 */
async function embedCandidate(
  candidate: CandidateImage,
  embedder: Embedder,
): Promise<Float32Array> {
  const what = `candidate "${candidate.filename}"`;
  const blob = await fetchImageBytes(candidate.thumbUrl ?? candidate.url, what);
  const { img, release } = await decodeImage(blob, what);
  try {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w * h > MAX_PIXELS) {
      throw new Error(
        `${((w * h) / 1e6).toFixed(1)} MP — over the 12 MP limit`,
      );
    }
    const dets = await detectForAlignment(img);
    if (dets.length !== 1) {
      throw new Error(`${dets.length} faces`);
    }
    const embedding = await embedAligned(embedder, alignFace(imageToRaster(img), dets[0]!));
    if (embedding.length === 0) {
      throw new Error("empty embedding");
    }
    return embedding;
  } finally {
    release();
  }
}

/**
 * Resolve a typed name to candidate photos, embed the usable ones, and
 * return a diverse preview set for user confirmation. Nothing is persisted
 * here — embeddings stay in this document until CONFIRM_ENROLL.
 */
async function resolvePreview(name: unknown): Promise<{ preview: EnrollPreview }> {
  if (typeof name !== "string" || name.trim() === "") {
    throw new Error('faceBlock: RESOLVE_PREVIEW requires a non-empty "name"');
  }
  const trimmed = name.trim();
  const { candidates } = await resolveCandidates(trimmed, { limit: MAX_RESOLVE_FETCH });
  const rejected: { url: string; reason: string }[] = [];
  if (candidates.length === 0) {
    return {
      preview: {
        name: trimmed,
        candidatesTried: 0,
        facesFound: 0,
        kept: [],
        rejected: [
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

  const embedder = await getEmbedder();
  const embedded: (EmbeddedCandidate | undefined)[] = [];
  let facesFound = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    for (;;) {
      const i = next++;
      if (i >= candidates.length) return;
      const candidate = candidates[i]!;
      try {
        const embedding = await embedCandidate(candidate, embedder);
        embedded[i] = { candidate, embedding };
        facesFound++;
      } catch (e) {
        // A bad candidate must not abort the whole resolve.
        rejected.push({ url: candidate.url, reason: errText(e) });
      }
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(RESOLVE_FETCH_CONCURRENCY, candidates.length) }, () => worker()),
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
 * Defensively validate the confirmed preview faces — they crossed a message
 * boundary — in the same spirit as toBlockedIdentities. Entries with a
 * malformed embedding are dropped rather than trusted.
 *
 * The width is NOT pinned to a constant: it is a property of whichever model
 * file is bundled (w600k_mbf emits 512-d, while the synthetic benchmark in
 * bench/ declares 128 for its own protocol). Pinning it to the benchmark's
 * EMBED_DIM silently rejected every real embedding. What must hold is that the
 * vectors are non-empty, finite, and mutually comparable — mixed widths cannot
 * be cosine-compared, so those are dropped.
 */
function toPreviewFaces(raw: unknown): EnrollPreviewFace[] {
  if (!Array.isArray(raw)) return [];
  const out: EnrollPreviewFace[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const f = item as Partial<EnrollPreviewFace>;
    const emb = f.embedding;
    if (!Array.isArray(emb) || emb.length === 0 || !emb.every((v) => Number.isFinite(v))) {
      continue;
    }
    const width = out[0]?.embedding.length ?? emb.length;
    if (emb.length !== width) continue;
    out.push({
      url: typeof f.url === "string" ? f.url : "",
      thumbUrl: typeof f.thumbUrl === "string" ? f.thumbUrl : undefined,
      filename: typeof f.filename === "string" ? f.filename : "",
      source: typeof f.source === "string" ? f.source : "",
      score: typeof f.score === "number" && Number.isFinite(f.score) ? f.score : 0,
      embedding: emb.slice(),
    });
  }
  return out;
}

/**
 * Turn the user's confirmed preview faces into a SavedIdentity. The id is
 * derived from the name the same way references.json ids are written
 * (kebab-case slug), with a deterministic hash fallback for names that
 * slugify to nothing.
 */
function identityId(name: string): string {
  const slug = normalizeName(name)
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (slug) return slug;
  let hash = 0;
  for (const ch of name) hash = (hash * 31 + ch.codePointAt(0)!) >>> 0;
  return `person-${hash.toString(36)}`;
}

async function confirmEnroll(
  name: unknown,
  rawFaces: unknown,
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
  return {
    identity: {
      id: identityId(trimmed),
      name: trimmed,
      embeddings: faces.map((f) => f.embedding),
      threshold: MATCH_THRESHOLD,
      sources: faces.map((f) => f.url),
      createdAt: Date.now(),
    },
  };
}

/* ---------- analyze ---------- */
function toBlockedIdentities(raw: unknown): BlockedIdentity[] {
  if (!Array.isArray(raw)) return [];
  const out: BlockedIdentity[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const s = item as Partial<SavedIdentity>;
    if (typeof s.id !== "string" || !Array.isArray(s.embeddings)) continue;
    const embeddings = s.embeddings
      .filter(
        (e): e is number[] =>
          Array.isArray(e) && e.length > 0 && e.every((v) => Number.isFinite(v)),
      )
      .map((e) => new Float32Array(e));
    if (embeddings.length === 0) continue;
    out.push({
      id: s.id,
      displayName: typeof s.name === "string" ? s.name : undefined,
      embeddings,
      threshold: Number.isFinite(s.threshold) ? (s.threshold as number) : MATCH_THRESHOLD,
      createdAt: typeof s.createdAt === "number" ? s.createdAt : 0,
    });
  }
  return out;
}

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

async function analyze(url: unknown, rawIdentities: unknown): Promise<{ result: ImageResult }> {
  if (typeof url !== "string" || !/^https?:\/\//i.test(url)) {
    throw new Error('faceBlock: ANALYZE requires an http(s) "url"');
  }
  const identities = toBlockedIdentities(rawIdentities);
  const blob = await fetchImageBytes(url, `image "${url}"`);
  const { img, release } = await decodeImage(blob, `image "${url}"`);
  try {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w * h > MAX_PIXELS) {
      throw new Error(
        `faceBlock: image is ${w}x${h} (${((w * h) / 1e6).toFixed(1)} MP) — over the 12 MP limit`,
      );
    }
    // YuNet detects; the landmarker is kept only as a fallback so a model
    // failure degrades to the old behaviour instead of masking nothing.
    // Same detector as enrolment, so alignment conventions match.
    const dets = await detectForAlignment(img);
    const regions: ImageResult["regions"] = [];
    if (dets.length > 0 && identities.length > 0) {
      const embedder = await getEmbedder();
      for (const det of dets) {
        // Crop to the face: alignment needs a 112px chip, not a 23 MB bitmap.
        const { raster, offsetX, offsetY } = imageToRasterRegion(img, det.box);
        const local: FaceDetection = {
          ...det,
          box: { ...det.box, x: det.box.x - offsetX, y: det.box.y - offsetY },
          landmarks: det.landmarks?.map((pt) => ({ x: pt.x - offsetX, y: pt.y - offsetY })),
        };
        const embedding = await embedAligned(embedder, alignFace(raster, local));
        const match = matchFace(embedding, identities, { minAgreements: 1 });
        if (match) {
          regions.push({
            ...expandBox(det.box, w, h),
            confidence: match.score,
            identityId: match.identityId,
          });
        }
      }
    }
    return { result: { width: w, height: h, faceCount: dets.length, regions } };
  } finally {
    release();
  }
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
): Promise<{ result: FrameResult }> {
  // The frame arrives base64-encoded: runtime.sendMessage JSON-serializes its
  // payload, so raw bytes would be lost in transit.
  const b64 = typeof jpegBase64 === "string" ? jpegBase64 : "";
  let bytes: Uint8Array<ArrayBuffer> | null = null;
  if (b64.length > 0) {
    try {
      const binary = atob(b64);
      const out = new Uint8Array(new ArrayBuffer(binary.length));
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
      bytes = out;
    } catch {
      bytes = null;
    }
  }
  if (!bytes || bytes.length === 0) {
    throw new Error('faceBlock: ANALYZE_FRAME requires a non-empty "jpegBase64" string');
  }
  const identities = toBlockedIdentities(rawIdentities);
  const { img, release } = await decodeImage(
    new Blob([bytes], { type: "image/jpeg" }),
    "video frame",
  );
  try {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w * h > MAX_PIXELS) {
      throw new Error(
        `faceBlock: video frame is ${w}x${h} (${((w * h) / 1e6).toFixed(1)} MP) — over the 12 MP limit`,
      );
    }
    const dets: FaceDetection[] = await detectForAlignment(img);
    const regions: FrameResult["regions"] = [];
    if (dets.length > 0 && identities.length > 0) {
      const embedder = await getEmbedder();
      for (const det of dets) {
        // Crop to the face: alignment needs a 112px chip, not a 23 MB bitmap.
        const { raster, offsetX, offsetY } = imageToRasterRegion(img, det.box);
        const local: FaceDetection = {
          ...det,
          box: { ...det.box, x: det.box.x - offsetX, y: det.box.y - offsetY },
          landmarks: det.landmarks?.map((pt) => ({ x: pt.x - offsetX, y: pt.y - offsetY })),
        };
        const embedding = await embedAligned(embedder, alignFace(raster, local));
        const match = matchFace(embedding, identities, { minAgreements: 1 });
        if (match) {
          regions.push({
            ...expandBox(det.box, w, h),
            confidence: match.score,
            identityId: match.identityId,
          });
        }
      }
    }
    return { result: { width: w, height: h, regions } };
  } finally {
    release();
  }
}

/* ---------- message entrypoint ---------- */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Only same-extension internal callers (background service worker, options
  // page). Reject content scripts (sender.tab set) and anything external.
  if (sender.id !== chrome.runtime.id || sender.tab) return false;
  if (!message || typeof message !== "object") return false;
  const m = message as {
    target?: string;
    type?: string;
    name?: string;
    url?: string;
    jpegBase64?: string;
    identities?: SavedIdentity[];
    faces?: EnrollPreviewFace[];
  };
  if (m.target !== "offscreen") return false;
  let work: Promise<
    | { identity: SavedIdentity }
    | { result: ImageResult }
    | { result: FrameResult }
    | { preview: EnrollPreview }
  >;
  if (m.type === "ENROLL") {
    work = enqueue(() => enroll(m.name));
  } else if (m.type === "ANALYZE") {
    work = enqueue(() => analyze(m.url, m.identities));
  } else if (m.type === "ANALYZE_FRAME") {
    work = enqueue(() => analyzeFrame(m.jpegBase64, m.identities));
  } else if (m.type === "RESOLVE_PREVIEW") {
    work = enqueue(() => resolvePreview(m.name));
  } else if (m.type === "CONFIRM_ENROLL") {
    work = enqueue(() => confirmEnroll(m.name, m.faces));
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
