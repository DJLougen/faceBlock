/**
 * FaceBlock local demo: enroll reference faces into named identities, then mask
 * matching faces in a test image. Everything runs in-browser; the only fetches
 * are the local model files under /models, /mediapipe-wasm and /ort.
 */

import { createFaceLandmarker, detectFaces } from "../src/cv/detector.ts";
import { createEmbedder, embedAligned, type Embedder } from "../src/cv/embedder.ts";
import { alignFace } from "../src/cv/align.ts";
import { imageToRaster } from "../src/cv/raster.ts";
import { renderCensors } from "../src/overlay/renderer.ts";
import { matchFace } from "../src/matching/matcher.ts";
import { cosineNormalized } from "../src/matching/cosine.ts";
import { MIN_GALLERY_AGREEMENTS, MIN_REFERENCE_IMAGES } from "../src/shared/config.ts";
import type {
  BlockedIdentity,
  CensorRegion,
  FaceDetection,
  Raster,
} from "../src/shared/types.ts";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";

/* ---------- limits ---------- */
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_DIMENSION = 4096;
const MAX_REF_FILES = 8;
const NEW_IDENTITY = "__new__";

/* ---------- dom ---------- */
function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

const el = {
  statusInd: $("status-ind"),
  statusText: $("status-text"),
  errors: $("errors"),
  identitySelect: $("identity-select") as HTMLSelectElement,
  identityName: $("identity-name") as HTMLInputElement,
  refInput: $("ref-input") as HTMLInputElement,
  refLabel: $("ref-label"),
  addSelected: $("add-selected") as HTMLButtonElement,
  clearRefs: $("clear-refs") as HTMLButtonElement,
  refNote: $("ref-note"),
  candidateList: $("candidate-list"),
  identityList: $("identity-list"),
  testInput: $("test-input") as HTMLInputElement,
  testLabel: $("test-label"),
  busyNote: $("busy-note"),
  threshold: $("threshold") as HTMLInputElement,
  thresholdValue: $("threshold-value") as HTMLOutputElement,
  testWrap: $("test-wrap"),
  testImg: $("test-image") as HTMLImageElement,
  testEmpty: $("test-empty"),
  testStats: $("test-stats"),
  scoreLog: $("score-log"),
};

/* ---------- state ---------- */
interface RefCandidate {
  fileName: string;
  img: HTMLImageElement;
  raster: Raster;
  detections: FaceDetection[];
  crops: string[]; // data URLs, parallel to detections
  el: HTMLElement;
}

/** In-memory blocked identity: display name + enrolled face embeddings/crops. */
interface Identity {
  id: string;
  name: string;
  embeddings: Float32Array[];
  crops: string[];
}

interface ProbeFace {
  detection: FaceDetection;
  embedding: Float32Array;
  cropUrl: string;
}

interface Probe {
  img: HTMLImageElement;
  source: { width: number; height: number };
  faces: ProbeFace[];
  ms: number;
  revoke: () => void;
}

let landmarker: FaceLandmarker | null = null;
let embedder: Embedder | null = null;
let busy = false;
let epoch = 0;
let nextIdentityNum = 1;
let candidates: RefCandidate[] = [];
let selected = new Set<string>();
let identities: Identity[] = [];
let probe: Probe | null = null;
let lastRegions: CensorRegion[] = [];

/* ---------- small helpers ---------- */
function fmtErr(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function setStatus(kind: "idle" | "loading" | "ready" | "error", text: string): void {
  el.statusInd.className = `ind ind-${kind}`;
  el.statusText.textContent = text;
}

function setErrors(msgs: string[]): void {
  el.errors.replaceChildren(
    ...msgs.map((m) => {
      const d = document.createElement("div");
      d.textContent = m;
      return d;
    }),
  );
  el.errors.hidden = msgs.length === 0;
}

function threshold(): number {
  return Number(el.threshold.value);
}

function updateControls(): void {
  const ready = landmarker !== null && embedder !== null;
  const off = busy || !ready;
  el.refInput.disabled = off;
  el.testInput.disabled = off;
  el.refLabel.setAttribute("aria-disabled", String(off));
  el.testLabel.setAttribute("aria-disabled", String(off));
  el.identitySelect.disabled = off;
  el.identityName.disabled = off || el.identitySelect.value !== NEW_IDENTITY;
  el.addSelected.disabled = off || selected.size === 0;
  el.clearRefs.disabled = busy || (identities.length === 0 && candidates.length === 0);
  el.threshold.disabled = busy;
  el.busyNote.hidden = !busy;
}

function setBusy(v: boolean): void {
  busy = v;
  updateControls();
}

function loadImage(url: string): Promise<HTMLImageElement> {
  const { promise, resolve, reject } = Promise.withResolvers<HTMLImageElement>();
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = () => reject(new Error("could not decode image"));
  img.src = url;
  return promise;
}

/**
 * Decode a File into an <img>, rejecting oversize input and downscaling
 * anything beyond MAX_DIMENSION so rasters stay bounded.
 */
async function loadBoundedImage(file: File): Promise<{ img: HTMLImageElement; revoke: () => void }> {
  if (!file.type.startsWith("image/")) {
    throw new Error(`"${file.name}" is not an image file`);
  }
  if (file.size > MAX_FILE_BYTES) {
    throw new Error(`"${file.name}" is ${(file.size / 1048576).toFixed(1)} MB (max 20 MB)`);
  }
  const urls: string[] = [];
  const revoke = () => urls.forEach((u) => URL.revokeObjectURL(u));
  try {
    const u1 = URL.createObjectURL(file);
    urls.push(u1);
    let img = await loadImage(u1);
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w === 0 || h === 0) throw new Error(`"${file.name}" decoded empty`);
    const scale = Math.min(1, MAX_DIMENSION / Math.max(w, h));
    if (scale < 1) {
      const cw = Math.round(w * scale);
      const ch = Math.round(h * scale);
      const c = document.createElement("canvas");
      c.width = cw;
      c.height = ch;
      const ctx = c.getContext("2d");
      if (!ctx) throw new Error("2d canvas unavailable");
      ctx.drawImage(img, 0, 0, cw, ch);
      const { promise, resolve, reject } = Promise.withResolvers<Blob>();
      c.toBlob((b) => (b ? resolve(b) : reject(new Error("downscale failed"))), "image/png");
      const u2 = URL.createObjectURL(await promise);
      urls.push(u2);
      img = await loadImage(u2);
    }
    return { img, revoke };
  } catch (e) {
    revoke();
    throw e;
  }
}

/** Expanded detection-box crop as a small data URL for display. */
function cropThumb(img: HTMLImageElement, det: FaceDetection, size = 112): string {
  const { box } = det;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const side = Math.max(box.width, box.height) * 1.25;
  const sx = Math.max(0, Math.min(img.naturalWidth - 1, cx - side / 2));
  const sy = Math.max(0, Math.min(img.naturalHeight - 1, cy - side / 2));
  const sw = Math.min(side, img.naturalWidth - sx);
  const sh = Math.min(side, img.naturalHeight - sy);
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);
  return c.toDataURL("image/png");
}

/* ---------- identities ---------- */
function identityById(id: string): Identity | undefined {
  return identities.find((i) => i.id === id);
}

function selectedIdentity(): Identity | null {
  const v = el.identitySelect.value;
  if (v !== NEW_IDENTITY) return identityById(v) ?? null;
  const name = el.identityName.value.trim() || `Person ${nextIdentityNum}`;
  const idn: Identity = {
    id: `id-${Date.now().toString(36)}-${nextIdentityNum}`,
    name,
    embeddings: [],
    crops: [],
  };
  nextIdentityNum++;
  identities.push(idn);
  const opt = document.createElement("option");
  opt.value = idn.id;
  opt.textContent = idn.name;
  el.identitySelect.appendChild(opt);
  el.identitySelect.value = idn.id;
  el.identityName.value = "";
  return idn;
}

function refreshIdentitySelect(): void {
  const cur = el.identitySelect.value;
  el.identitySelect.replaceChildren();
  const nw = document.createElement("option");
  nw.value = NEW_IDENTITY;
  nw.textContent = "New identity…";
  el.identitySelect.appendChild(nw);
  for (const idn of identities) {
    const opt = document.createElement("option");
    opt.value = idn.id;
    opt.textContent = `${idn.name} (${idn.embeddings.length})`;
    el.identitySelect.appendChild(opt);
  }
  el.identitySelect.value = identityById(cur) ? cur : NEW_IDENTITY;
}

function removeIdentity(id: string): void {
  epoch++;
  identities = identities.filter((i) => i.id !== id);
  refreshIdentitySelect();
  refreshRefUI();
  recomputeMatches();
}

/* ---------- boot ---------- */
async function boot(): Promise<void> {
  setStatus("loading", "Loading face models…");
  const [lm, em] = await Promise.all([
    createFaceLandmarker().then(
      (v) => ({ ok: true as const, v }),
      (e) => ({ ok: false as const, e }),
    ),
    createEmbedder().then(
      (v) => ({ ok: true as const, v }),
      (e) => ({ ok: false as const, e }),
    ),
  ]);
  const errs: string[] = [];
  if (lm.ok) landmarker = lm.v;
  else errs.push(`Face detector failed to load: ${fmtErr(lm.e)}`);
  if (em.ok) embedder = em.v;
  else errs.push(`Face embedder failed to load: ${fmtErr(em.e)}`);

  if (errs.length > 0) {
    setStatus("error", "Model load failed");
    setErrors([...errs, "Check that demo/public/models, /mediapipe-wasm and /ort are served, then reload."]);
  } else {
    setStatus("ready", "Models ready — enroll references to begin");
  }
  updateControls();
}

/* ---------- reference enrollment ---------- */
async function onRefFiles(files: FileList | null): Promise<void> {
  if (!files || files.length === 0 || !landmarker) return;
  const my = ++epoch;
  setBusy(true);
  const errs: string[] = [];
  try {
    const list = [...files].slice(0, MAX_REF_FILES);
    if (files.length > MAX_REF_FILES) {
      errs.push(`Only the first ${MAX_REF_FILES} files were used (batch limit).`);
    }
    for (const file of list) {
      if (epoch !== my) return;
      try {
        const { img, revoke } = await loadBoundedImage(file);
        try {
          const raster = imageToRaster(img);
          const dets = detectFaces(landmarker, img);
          const crops = dets.map((d) => cropThumb(img, d));
          const cand: RefCandidate = {
            fileName: file.name,
            img,
            raster,
            detections: dets,
            crops,
            el: document.createElement("div"),
          };
          candidates.push(cand);
          cand.el = renderCandidate(cand);
        } finally {
          revoke();
        }
      } catch (e) {
        errs.push(fmtErr(e));
      }
    }
    setErrors(errs);
    refreshRefUI();
  } finally {
    setBusy(false);
  }
}

function renderCandidate(cand: RefCandidate): HTMLElement {
  const card = document.createElement("div");
  card.className = "card";
  const head = document.createElement("div");
  head.className = "card-head";
  const name = document.createElement("span");
  name.textContent = cand.fileName;
  const count = document.createElement("span");
  count.textContent =
    cand.detections.length === 0
      ? "No faces detected"
      : `${cand.detections.length} face${cand.detections.length === 1 ? "" : "s"}`;
  head.append(name, count);
  card.appendChild(head);
  if (cand.detections.length === 0) return card;

  const grid = document.createElement("div");
  grid.className = "crops";
  cand.crops.forEach((url, i) => {
    const label = document.createElement("label");
    label.className = "crop";
    label.title = `Face ${i + 1} — tick to enroll`;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selected.has(candKey(cand, i));
    cb.addEventListener("change", () => {
      const key = candKey(cand, i);
      if (cb.checked) selected.add(key);
      else selected.delete(key);
      updateControls();
      el.addSelected.textContent = `Add selected faces (${selected.size})`;
    });
    const im = document.createElement("img");
    im.src = url;
    im.alt = `Detected face ${i + 1} in ${cand.fileName}`;
    label.append(cb, im);
    grid.appendChild(label);
  });
  card.appendChild(grid);
  return card;
}

function candKey(cand: RefCandidate, i: number): string {
  return `${candidates.indexOf(cand)}:${i}`;
}

async function onAddSelected(): Promise<void> {
  if (!embedder || selected.size === 0) return;
  const my = ++epoch;
  setBusy(true);
  const errs: string[] = [];
  try {
    const target = selectedIdentity();
    if (!target) {
      setErrors(["Pick an identity (or type a new name) before adding faces."]);
      return;
    }
    const picked: { cand: RefCandidate; i: number }[] = [];
    for (const key of selected) {
      const [ci, fi] = key.split(":").map(Number);
      const cand = candidates[ci!];
      if (cand && fi !== undefined && cand.detections[fi]) picked.push({ cand, i: fi });
    }
    const addedKeys = new Set<string>();
    for (const { cand, i } of picked) {
      if (epoch !== my) return;
      try {
        const aligned = alignFace(cand.raster, cand.detections[i]!);
        const embedding = await embedAligned(embedder, aligned);
        target.embeddings.push(embedding);
        target.crops.push(cand.crops[i]!);
        addedKeys.add(candKey(cand, i));
      } catch (e) {
        errs.push(`${cand.fileName} face ${i + 1}: ${fmtErr(e)}`);
      }
    }
    // Drop enrolled faces from candidates; refreshRefUI rebuilds the cards.
    const remaining: RefCandidate[] = [];
    for (const cand of candidates) {
      const keepIdx: number[] = [];
      cand.detections.forEach((_, i) => {
        if (!addedKeys.has(candKey(cand, i))) keepIdx.push(i);
      });
      if (keepIdx.length === cand.detections.length) {
        remaining.push(cand);
      } else if (keepIdx.length > 0) {
        cand.detections = keepIdx.map((i) => cand.detections[i]!);
        cand.crops = keepIdx.map((i) => cand.crops[i]!);
        remaining.push(cand);
      }
    }
    candidates = remaining;
    selected.clear();
    setErrors(errs);
    refreshIdentitySelect();
    refreshRefUI();
    recomputeMatches();
  } finally {
    setBusy(false);
  }
}

function onClear(): void {
  epoch++;
  identities = [];
  candidates = [];
  selected.clear();
  el.candidateList.replaceChildren();
  el.refInput.value = "";
  el.identityName.value = "";
  refreshIdentitySelect();
  setErrors([]);
  refreshRefUI();
  recomputeMatches();
}

function refreshRefUI(): void {
  if (candidates.length === 0) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No pending reference images.";
    el.candidateList.replaceChildren(p);
  } else {
    el.candidateList.replaceChildren(...candidates.map((c) => c.el));
  }
  if (identities.length === 0) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No identities yet — pick a name above, upload photos, select faces, add.";
    el.identityList.replaceChildren(p);
  } else {
    el.identityList.replaceChildren(...identities.map(identityCard));
  }
  const total = identities.reduce((n, i) => n + i.embeddings.length, 0);
  const thin = identities.filter((i) => i.embeddings.length > 0 && i.embeddings.length < MIN_REFERENCE_IMAGES);
  if (total === 0) {
    el.refNote.textContent = "";
    el.refNote.className = "note";
  } else if (thin.length > 0) {
    el.refNote.textContent =
      `${thin.map((i) => i.name).join(", ")}: under ${MIN_REFERENCE_IMAGES} references — fine ` +
      `for a smoke test, but ${MIN_REFERENCE_IMAGES}+ varied photos per person are recommended. ` +
      `A face is masked only when min(${MIN_GALLERY_AGREEMENTS}, that identity's ref count) of ` +
      `its embeddings clear the threshold, so a single reference must match on its own.`;
    el.refNote.className = "note warn";
  } else {
    el.refNote.textContent =
      `${identities.length} identit${identities.length === 1 ? "y" : "ies"}, ${total} ` +
      `references. Agreement rule per identity: min(${MIN_GALLERY_AGREEMENTS}, ref count) ` +
      `embeddings must clear the threshold to mask a face.`;
    el.refNote.className = "note";
  }
  el.addSelected.textContent = `Add selected faces (${selected.size})`;
  updateControls();
}

function identityCard(idn: Identity): HTMLElement {
  const card = document.createElement("div");
  card.className = "identity";
  const head = document.createElement("div");
  head.className = "identity-head";
  const name = document.createElement("strong");
  name.textContent = idn.name;
  const pill = document.createElement("span");
  pill.className = "pill";
  pill.textContent = `${idn.embeddings.length} ref${idn.embeddings.length === 1 ? "" : "s"}`;
  const need = document.createElement("span");
  need.className = "muted";
  need.textContent = `need ≥ ${Math.min(MIN_GALLERY_AGREEMENTS, idn.embeddings.length)} agree`;
  const rm = document.createElement("button");
  rm.type = "button";
  rm.className = "btn tiny danger";
  rm.textContent = "Remove";
  rm.addEventListener("click", () => removeIdentity(idn.id));
  head.append(name, pill, need, rm);
  const tray = document.createElement("div");
  tray.className = "tray";
  for (const [i, url] of idn.crops.entries()) {
    const im = document.createElement("img");
    im.src = url;
    im.alt = `${idn.name} reference ${i + 1}`;
    tray.appendChild(im);
  }
  card.append(head, tray);
  return card;
}

/* ---------- test image ---------- */
async function onTestFile(files: FileList | null): Promise<void> {
  const file = files?.[0];
  if (!file || !landmarker || !embedder) return;
  const my = ++epoch;
  setBusy(true);
  try {
    const { img, revoke } = await loadBoundedImage(file);
    if (epoch !== my) {
      revoke();
      return;
    }
    const t0 = performance.now();
    const raster = imageToRaster(img);
    const dets = detectFaces(landmarker, img);
    const faces: ProbeFace[] = [];
    for (const det of dets) {
      const aligned = alignFace(raster, det);
      const embedding = await embedAligned(embedder, aligned);
      if (epoch !== my) {
        revoke();
        return;
      }
      faces.push({ detection: det, embedding, cropUrl: cropThumb(img, det, 80) });
    }
    const ms = performance.now() - t0;

    probe?.revoke();
    probe = {
      img: el.testImg,
      source: { width: img.naturalWidth, height: img.naturalHeight },
      faces,
      ms,
      revoke,
    };
    el.testImg.src = img.src;
    await el.testImg.decode().catch(() => undefined);
    el.testWrap.hidden = false;
    el.testEmpty.hidden = true;
    recomputeMatches();
  } catch (e) {
    setErrors([fmtErr(e)]);
  } finally {
    setBusy(false);
  }
}

/** Score + mask decision for the current probe. No inference — slider-safe. */
function recomputeMatches(): void {
  if (!probe) {
    updateControls();
    return;
  }
  const t = threshold();
  const blocked: BlockedIdentity[] = identities
    .filter((i) => i.embeddings.length > 0)
    .map((i) => ({
      id: i.id,
      displayName: i.name,
      embeddings: i.embeddings,
      threshold: t,
      createdAt: 0,
    }));

  const regions: CensorRegion[] = [];
  const rows: HTMLElement[] = [];
  probe.faces.forEach((face, i) => {
    const res = blocked.length > 0 ? matchFace(face.embedding, blocked) : null;
    if (res) {
      regions.push({ ...face.detection.box, confidence: res.score, identityId: res.identityId });
    }
    rows.push(logRow(i, face, res, blocked, t));
  });
  if (probe.faces.length === 0) {
    const li = document.createElement("li");
    li.className = "note-row";
    li.textContent = "No faces detected in this image.";
    rows.push(li);
  }
  el.scoreLog.replaceChildren(...rows);

  lastRegions = regions;
  renderCensors(probe.img, probe.source, regions, "contain");

  el.testStats.textContent = "";
  const strong = document.createElement("strong");
  strong.textContent = `masked ${regions.length}`;
  el.testStats.append(
    `Detected ${probe.faces.length} face${probe.faces.length === 1 ? "" : "s"} · `,
    strong,
    ` · ${probe.ms.toFixed(0)} ms (detect + align + embed)`,
  );
  updateControls();
}

function logRow(
  i: number,
  face: ProbeFace,
  res: { identityId: string; score: number } | null,
  blocked: BlockedIdentity[],
  t: number,
): HTMLElement {
  const li = document.createElement("li");
  if (face.cropUrl) {
    const im = document.createElement("img");
    im.src = face.cropUrl;
    im.alt = "";
    li.appendChild(im);
  }
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = `Face ${i + 1}`;
  const detail = document.createElement("span");
  detail.className = "detail";
  const chip = document.createElement("span");

  if (blocked.length === 0) {
    detail.textContent = "no references enrolled — enroll faces on the left to match";
    chip.className = "chip na";
    chip.textContent = "n/a";
  } else {
    // Per-identity max cosine + agreement, best identity first.
    const stats = blocked
      .map((b) => {
        const cos = b.embeddings
          .map((e) => cosineNormalized(face.embedding, e))
          .sort((a, z) => z - a);
        const kNeed = Math.min(MIN_GALLERY_AGREEMENTS, b.embeddings.length);
        return {
          name: b.displayName ?? b.id,
          max: cos[0]!,
          agree: cos.filter((c) => c >= t).length,
          kNeed,
        };
      })
      .sort((a, b) => b.max - a.max);
    const top = stats[0]!;
    detail.textContent =
      `closest: ${top.name} · max cosine ${top.max.toFixed(3)} · ` +
      `agreement ${top.agree}/${top.kNeed} (need ≥ ${top.kNeed} refs ≥ ${t.toFixed(2)})` +
      (stats.length > 1
        ? ` · others: ${stats.slice(1).map((s) => `${s.name} ${s.max.toFixed(3)}`).join(", ")}`
        : "");
    if (res) {
      const name = identityById(res.identityId)?.name ?? res.identityId;
      chip.className = "chip masked";
      chip.textContent = `masked: ${name}`;
    } else {
      chip.className = "chip open";
      chip.textContent = "visible";
    }
  }
  li.append(who, detail, chip);
  return li;
}

/* ---------- wiring ---------- */
el.identitySelect.addEventListener("change", () => {
  if (el.identitySelect.value === NEW_IDENTITY) {
    el.identityName.value = "";
    el.identityName.focus();
  } else {
    el.identityName.value = identityById(el.identitySelect.value)?.name ?? "";
  }
  updateControls();
});
el.refInput.addEventListener("change", () => void onRefFiles(el.refInput.files));
el.testInput.addEventListener("change", () => void onTestFile(el.testInput.files));
el.addSelected.addEventListener("click", () => void onAddSelected());
el.clearRefs.addEventListener("click", onClear);
el.threshold.addEventListener("input", () => {
  el.thresholdValue.textContent = threshold().toFixed(2);
  recomputeMatches();
});

// Keep boxes glued to the image across layout changes.
new ResizeObserver(() => {
  if (probe && !el.testWrap.hidden) {
    renderCensors(probe.img, probe.source, lastRegions, "contain");
  }
}).observe(el.testImg);

window.addEventListener("unhandledrejection", (ev) => {
  setErrors([`Unexpected error: ${fmtErr(ev.reason)}`]);
});
window.addEventListener("error", (ev) => {
  setErrors([`Unexpected error: ${ev.message}`]);
});

refreshIdentitySelect();
refreshRefUI();
updateControls();
void boot();
