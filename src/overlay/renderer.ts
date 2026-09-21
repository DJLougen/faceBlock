import type { Box, CensorRegion, ObjectFit, Size } from "../shared/types.ts";
import { expandMaskOverlay, mapSourceBoxToRendered } from "./coordinates.ts";

const LAYER_ATTR = "data-faceblock-overlay";

export function ensureOverlay(img: HTMLImageElement): HTMLElement {
  const parent = img.parentElement;
  if (!parent) throw new Error("image has no parent for overlay");
  const style = getComputedStyle(parent);
  if (style.position === "static") parent.style.position = "relative";
  let layer = parent.querySelector<HTMLElement>(`:scope > [${LAYER_ATTR}]`);
  if (!layer) {
    layer = document.createElement("div");
    layer.setAttribute(LAYER_ATTR, "");
    layer.style.position = "absolute";
    layer.style.left = "0";
    layer.style.top = "0";
    layer.style.right = "0";
    layer.style.bottom = "0";
    layer.style.pointerEvents = "none";
    layer.style.overflow = "hidden";
    parent.appendChild(layer);
  }
  return layer;
}

export function renderCensors(
  img: HTMLImageElement,
  source: Size,
  regions: readonly CensorRegion[],
  objectFit: ObjectFit = "contain",
): void {
  const layer = ensureOverlay(img);
  layer.replaceChildren();
  const rendered: Size = { width: img.clientWidth, height: img.clientHeight };
  for (const region of regions) {
    const mapped = mapSourceBoxToRendered(region, source, rendered, objectFit);
    const box = expandMaskOverlay(mapped, rendered);
    layer.appendChild(censorEl(box, region.identityId, region.confidence));
  }
}

function censorEl(box: Box, identityId: string, confidence: number): HTMLElement {
  const el = document.createElement("div");
  el.style.position = "absolute";
  el.style.left = `${box.x}px`;
  el.style.top = `${box.y}px`;
  el.style.width = `${box.width}px`;
  el.style.height = `${box.height}px`;
  el.style.background = "#000";
  el.dataset.identityId = identityId;
  el.dataset.confidence = String(confidence);
  return el;
}
