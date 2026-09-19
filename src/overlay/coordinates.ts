import type { Box, ObjectFit, Size } from "../shared/types.ts";

/**
 * Map a box in source-image pixel space to rendered-element pixel space,
 * following CSS object-fit semantics.
 *
 * Returns a zero box when the source has no area (avoids NaN/Infinity).
 */
export function mapSourceBoxToRendered(
  sourceBox: Box,
  source: Size,
  rendered: Size,
  objectFit: ObjectFit
): Box {
  const sw = source.width;
  const sh = source.height;
  if (sw <= 0 || sh <= 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  const rw = rendered.width;
  const rh = rendered.height;

  let scaleX: number;
  let scaleY: number;
  let offsetX = 0;
  let offsetY = 0;

  switch (objectFit) {
    case "fill": {
      scaleX = rw / sw;
      scaleY = rh / sh;
      break;
    }
    case "contain": {
      const s = Math.min(rw / sw, rh / sh);
      scaleX = s;
      scaleY = s;
      offsetX = (rw - sw * s) / 2;
      offsetY = (rh - sh * s) / 2;
      break;
    }
    case "cover": {
      const s = Math.max(rw / sw, rh / sh);
      scaleX = s;
      scaleY = s;
      offsetX = (rw - sw * s) / 2;
      offsetY = (rh - sh * s) / 2;
      break;
    }
    case "none": {
      scaleX = 1;
      scaleY = 1;
      offsetX = (rw - sw) / 2;
      offsetY = (rh - sh) / 2;
      break;
    }
    case "scale-down": {
      // Behaves as `contain` when the image overflows the element, else `none`.
      if (sw > rw || sh > rh) {
        const s = Math.min(rw / sw, rh / sh);
        scaleX = s;
        scaleY = s;
        offsetX = (rw - sw * s) / 2;
        offsetY = (rh - sh * s) / 2;
      } else {
        scaleX = 1;
        scaleY = 1;
        offsetX = (rw - sw) / 2;
        offsetY = (rh - sh) / 2;
      }
      break;
    }
  }

  return {
    x: sourceBox.x * scaleX + offsetX,
    y: sourceBox.y * scaleY + offsetY,
    width: sourceBox.width * scaleX,
    height: sourceBox.height * scaleY,
  };
}

/**
 * Grow a box by a fractional margin then a multiplicative scale.
 * When `clamp` is given, intersect the result with
 * [0, 0, clamp.width, clamp.height]; width/height never go negative.
 */
export function expandBox(
  box: Box,
  opts: { marginX: number; marginY: number; scaleX: number; scaleY: number },
  clamp?: Size
): Box {
  let x = box.x - box.width * opts.marginX;
  let y = box.y - box.height * opts.marginY;
  let width = box.width * opts.scaleX;
  let height = box.height * opts.scaleY;

  if (clamp) {
    const x2 = Math.min(x + width, clamp.width);
    const y2 = Math.min(y + height, clamp.height);
    x = Math.max(x, 0);
    y = Math.max(y, 0);
    width = Math.max(x2 - x, 0);
    height = Math.max(y2 - y, 0);
  }

  return { x, y, width, height };
}
