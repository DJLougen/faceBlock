import { readFile } from "node:fs/promises";
import jpeg from "jpeg-js";
import type { Raster } from "../shared/types.ts";

/** Decode a local JPEG into RGBA without canvas (Bun fixture path). */
export async function loadFixtureRaster(absPath: string): Promise<Raster> {
  const buf = await readFile(absPath);
  const decoded = jpeg.decode(buf, { useTArray: true });
  if (!decoded.width || !decoded.height) {
    throw new Error(`faceBlock: ${absPath} did not decode to a bitmap`);
  }
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8ClampedArray(decoded.data),
  };
}
