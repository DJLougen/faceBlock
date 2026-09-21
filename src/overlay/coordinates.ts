import {
  BOX_PADDING_BOTTOM,
  BOX_PADDING_TOP,
  BOX_PADDING_X,
} from "../shared/config.ts";
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
    case "contain": {
      scaleX = scaleY = Math.min(rw / sw, rh / sh);
      const dw = sw * scaleX;
      const dh = sh * scaleY;
      offsetX = (rw - dw) / 2;
      offsetY = (rh - dh) / 2;
      break;
    }
    case "cover": {
      scaleX = scaleY = Math.max(rw / sw, rh / sh);
      const dw = sw * scaleX;
      const dh = sh * scaleY;
      offsetX = (rw - dw) / 2;
      offsetY = (rh - dh) / 2;
      break;
    }
    case "none": {
      scaleX = scaleY = 1;
      offsetX = (rw - sw) / 2;
      offsetY = (rh - sh) / 2;
      break;
    }
    case "scale-down": {
      if (sw > rw || sh > rh) {
        scaleX = scaleY = Math.min(rw / sw, rh / sh);
        const dw = sw * scaleX;
        const dh = sh * scaleY;
        offsetX = (rw - dw) / 2;
        offsetY = (rh - dh) / 2;
      } else {
        scaleX = scaleY = 1;
        offsetX = (rw - sw) / 2;
        offsetY = (rh - sh) / 2;
      }
      break;
    }
    default: {
      scaleX = rw / sw;
      scaleY = rh / sh;
      break;
    }
  }

  return {
    x: offsetX + sourceBox.x * scaleX,
    y: offsetY + sourceBox.y * scaleY,
    width: sourceBox.width * scaleX,
    height: sourceBox.height * scaleY,
  };
}

function clampBox(box: Box, clamp: Size): Box {
  const x2 = Math.min(box.x + box.width, clamp.width);
  const y2 = Math.min(box.y + box.height, clamp.height);
  const x = Math.max(box.x, 0);
  const y = Math.max(box.y, 0);
  return {
    x,
    y,
    width: Math.max(x2 - x, 0),
    height: Math.max(y2 - y, 0),
  };
}

/**
 * Grow YuNet's face box for censoring: extra room above for hair, below for
 * beard/jaw. Applied once in the analyzer; render paths only map coordinates.
 */
export function expandDetectionBox(box: Box, clamp?: Size): Box {
  let x = box.x - box.width * BOX_PADDING_X;
  let y = box.y - box.height * BOX_PADDING_TOP;
  const width = box.width * (1 + 2 * BOX_PADDING_X);
  const height = box.height * (1 + BOX_PADDING_TOP + BOX_PADDING_BOTTOM);
  const out = { x, y, width, height };
  return clamp ? clampBox(out, clamp) : out;
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
    return clampBox({ x, y, width, height }, clamp);
  }

  return { x, y, width, height };
}
