import { expandBox } from "../overlay/coordinates.ts";
import type { Box, Size } from "../shared/types.ts";

/** Pre–eef58d1 analyzer expansion: scale about box center, clamped to image. */
export const LEGACY_ANALYZER_SCALE = { scaleX: 1.2, scaleY: 1.25 } as const;

/** Pre–eef58d1 content.ts second pass (margin + scale on mapped box). */
export const LEGACY_RENDER_MARGIN = { marginX: 0.15, marginY: 0.2 } as const;
export const LEGACY_RENDER_SCALE = { scaleX: 1.2, scaleY: 1.25 } as const;

export function legacyAnalyzerMaskBox(box: Box, image: Size): Box {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const hw = (box.width * LEGACY_ANALYZER_SCALE.scaleX) / 2;
  const hh = (box.height * LEGACY_ANALYZER_SCALE.scaleY) / 2;
  const w = image.width;
  const h = image.height;
  const x0 = Math.max(0, Math.min(cx - hw, w));
  const y0 = Math.max(0, Math.min(cy - hh, h));
  const x1 = Math.max(0, Math.min(cx + hw, w));
  const y1 = Math.max(0, Math.min(cy + hh, h));
  return { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
}

/** Analyzer + content double expansion before render-path dedup (eef58d1). */
export function legacyFullPipelineMaskBox(box: Box, image: Size): Box {
  const analyzer = legacyAnalyzerMaskBox(box, image);
  return expandBox(
    analyzer,
    {
      marginX: LEGACY_RENDER_MARGIN.marginX,
      marginY: LEGACY_RENDER_MARGIN.marginY,
      scaleX: LEGACY_RENDER_SCALE.scaleX,
      scaleY: LEGACY_RENDER_SCALE.scaleY,
    },
    image,
  );
}
