import type { Box } from "../src/shared/types.ts";

export interface SavedIdentity {
  id: string;
  name: string;
  embeddings: number[][];
  threshold: number;
  sources: string[];
  createdAt: number;
}
export interface BlockList {
  identities: SavedIdentity[];
  enabled: boolean;
  revision: number;
}
export interface ReferencePerson {
  id: string;
  name: string;
  aliases: string[];
  references: { path: string; source: string }[];
}
/**
 * Extension-page-only diagnostics for direct page -> offscreen ANALYZE /
 * ANALYZE_FRAME requests (verification tooling). Never attached to
 * background-forwarded production requests and never sent to content
 * scripts: numbers only — no embeddings, no pixels.
 */
export interface FaceDiagnostics {
  faceCount: number;
  /** The operating point the matcher applied. */
  threshold: number;
  /** Gallery agreements the matcher required. */
  minAgreements: number;
  /** Saved identities dropped whole for malformed embeddings before matching. */
  droppedIdentities: number;
  /** Saved identities that survived but lost at least one malformed vector. */
  degradedIdentities: number;
  faces: {
    box: Box;
    /** Unthresholded best cosine against the whole gallery; null when no gallery. */
    bestCosine: number | null;
    /** Identity behind bestCosine; null when no gallery. */
    bestIdentityId: string | null;
    /** True when the face actually matched (threshold + agreements). */
    matched: boolean;
    /** The matched identity, when matched. */
    matchedIdentityId: string | null;
  }[];
}
export interface ImageResult {
  width: number;
  height: number;
  faceCount: number;
  regions: { x: number; y: number; width: number; height: number; confidence: number; identityId: string }[];
  diagnostics?: FaceDiagnostics;
}
export interface FrameResult {
  width: number;   // decoded frame width in pixels
  height: number;  // decoded frame height in pixels
  regions: { x: number; y: number; width: number; height: number; confidence: number; identityId: string }[];
  diagnostics?: FaceDiagnostics;
}
export interface EnrollPreviewFace {
  url: string;
  thumbUrl?: string;
  filename: string;
  source: string;
  score: number;
  embedding: number[];   // stays on extension pages; persisted only on confirm
}
export interface EnrollPreview {
  name: string;
  /**
   * Set only when the preview targets an existing saved identity (a refresh):
   * confirming with this id replaces that identity's references. Absent for
   * a genuinely new person.
   */
  identityId?: string;
  candidatesTried: number;
  facesFound: number;
  kept: EnrollPreviewFace[];
  rejected: { url: string; reason: string }[];
}
// UI -> background: {target:'background',type:'GET_STATE'|'BLOCK_NAME'|'RESOLVE_PREVIEW'|'CONFIRM_ENROLL'|'REMOVE'|'SET_ENABLED', name?,id?,identityId?,enabled?,faces?}
// Content -> background: {target:'background',type:'PROCESS_IMAGE',url:string}
// Background -> offscreen: {target:'offscreen',type:'ANALYZE'|'ANALYZE_FRAME'|'RESOLVE_PREVIEW'|'CONFIRM_ENROLL',name?,url?,jpegBase64?,identities?:SavedIdentity[],faces?:EnrollPreviewFace[]}
// Extension page -> offscreen (diagnostics only): {target:'offscreen',type:'ANALYZE'|'ANALYZE_FRAME',url|jpegBase64,identities} — same handlers, but the
//   response carries `diagnostics` because the sender is an extension page.
// Every response: {ok:true, state?:BlockList,result?:ImageResult|FrameResult,identity?:SavedIdentity,preview?:EnrollPreview} OR {ok:false,error:string}.
// Enrollment flow: UI -> background {type:'RESOLVE_PREVIEW',name,identityId?} -> {ok,preview}; then
//   {type:'CONFIRM_ENROLL',name,faces,identityId?} -> {ok,identity,state}. Nothing persists before
//   CONFIRM_ENROLL — curated references.json hits also return a preview. BLOCK_NAME is the legacy
//   alias of RESOLVE_PREVIEW and likewise never persists.
// An explicit identityId must name an existing saved identity (refresh); without one the confirmed
//   id is the canonical references.json id for curated names, else a slug of the name.
// RESOLVE_PREVIEW/CONFIRM_ENROLL are extension-pages-only — EnrollPreviewFace carries raw embeddings.
// Background -> content: {target:'content',type:'STATE_CHANGED',revision:number,enabled:boolean}.
// Content -> background: {target:'background',type:'ANALYZE_FRAME',jpegBase64:string} — forwarded to the
// offscreen document as {target:'offscreen',type:'ANALYZE_FRAME',jpegBase64}; response {ok,result:FrameResult}.
// A second ANALYZE_FRAME while one is in flight gets {ok:false,error:'busy'} immediately — a stale video
// frame is worthless, so frames are dropped, never queued. Sampled frames are analysed on device and
// discarded; they are never cached, persisted, or uploaded.
// Do not expose embeddings outside extension-owned pages. Never send photos to remote inference.
