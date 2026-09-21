/**
 * Face-block mask appearance: opaque black by default; generic sponsor fill when
 * Earn is on and the box is large enough to show attribution.
 *
 * Masks stay pointer-events:none — view-only sponsorship; links belong in extension UI.
 */

/** Minimum width and height (px) before a sponsor creative + label may appear. */
export const MIN_SPONSOR_MASK_EDGE_PX = 120;

/** Bundled placeholder creative (copied to dist by build-extension). */
export const SPONSOR_PLACEHOLDER_PATH = "sponsors/placeholder.svg";

export type MaskStyle = "black" | "sponsor";

export function qualifiesForSponsorMask(widthPx: number, heightPx: number): boolean {
  return widthPx >= MIN_SPONSOR_MASK_EDGE_PX && heightPx >= MIN_SPONSOR_MASK_EDGE_PX;
}

export function resolveMaskStyle(
  earnEnabled: boolean,
  widthPx: number,
  heightPx: number,
): MaskStyle {
  if (!earnEnabled) return "black";
  if (!qualifiesForSponsorMask(widthPx, heightPx)) return "black";
  return "sponsor";
}

const MASK_SHELL =
  "position:absolute;pointer-events:none;display:block;overflow:hidden;box-sizing:border-box;";

const AD_LABEL = "Ad · FaceBlock";

/** Empty positioned shell; caller sets left/top/width/height each layout/frame. */
export function createMaskShell(): HTMLDivElement {
  const mask = document.createElement("div");
  mask.dataset.fbMask = "1";
  mask.style.cssText = MASK_SHELL;
  return mask;
}

export function maskAppearanceMatches(
  mask: HTMLElement,
  style: MaskStyle,
  sponsorImageUrl: string,
): boolean {
  if (mask.dataset.fbMaskStyle !== style) return false;
  if (style === "sponsor" && mask.dataset.fbSponsorUrl !== sponsorImageUrl) return false;
  return true;
}

/**
 * Rebuild mask fill/children when appearance changes. Caller sets position/size first.
 * No-op when `dataset.fbMaskStyle` already matches (video path calls this every frame).
 */
export function applyMaskStyle(
  mask: HTMLElement,
  style: MaskStyle,
  sponsorImageUrl: string,
): void {
  if (maskAppearanceMatches(mask, style, sponsorImageUrl)) return;

  const left = mask.style.left;
  const top = mask.style.top;
  const width = mask.style.width;
  const height = mask.style.height;
  const display = mask.style.display;

  mask.style.cssText = MASK_SHELL;
  mask.style.left = left;
  mask.style.top = top;
  mask.style.width = width;
  mask.style.height = height;
  mask.style.display = display;
  mask.replaceChildren();

  if (style === "black") {
    mask.style.background = "#000";
    mask.dataset.fbMaskStyle = "black";
    delete mask.dataset.fbSponsorUrl;
    return;
  }

  mask.style.background = "#111";
  mask.dataset.fbMaskStyle = "sponsor";
  mask.dataset.fbSponsorUrl = sponsorImageUrl;

  const creative = document.createElement("div");
  creative.setAttribute("aria-hidden", "true");
  creative.style.cssText =
    "position:absolute;inset:0;background-size:cover;background-position:center;background-repeat:no-repeat;pointer-events:none;";
  creative.style.backgroundImage = `url("${sponsorImageUrl}")`;

  const label = document.createElement("span");
  label.textContent = AD_LABEL;
  label.style.cssText =
    "position:absolute;left:0;bottom:0;max-width:100%;padding:1px 4px;" +
    "font:600 9px/1.25 system-ui,sans-serif;color:#fff;background:rgba(0,0,0,.78);" +
    "letter-spacing:.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;";

  mask.append(creative, label);
}

/**
 * Resolve style and apply when appearance changes. Caller updates position/size each
 * frame; repeated calls with the same resolved style reuse existing DOM (video path).
 */
export function styleMaskForEarn(
  mask: HTMLElement,
  earnEnabled: boolean,
  box: { width: number; height: number },
  sponsorImageUrl: string,
): MaskStyle {
  const style = resolveMaskStyle(earnEnabled, box.width, box.height);
  applyMaskStyle(mask, style, sponsorImageUrl);
  return style;
}
