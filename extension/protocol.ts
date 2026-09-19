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
export interface ImageResult {
  width: number;
  height: number;
  faceCount: number;
  regions: { x: number; y: number; width: number; height: number; confidence: number; identityId: string }[];
}
export interface FrameResult {
  width: number;   // decoded frame width in pixels
  height: number;  // decoded frame height in pixels
  regions: { x: number; y: number; width: number; height: number; confidence: number; identityId: string }[];
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
  candidatesTried: number;
  facesFound: number;
  kept: EnrollPreviewFace[];
  rejected: { url: string; reason: string }[];
}
// UI -> background: {target:'background',type:'GET_STATE'|'BLOCK_NAME'|'REMOVE'|'SET_ENABLED', name?,id?,enabled?}
// Content -> background: {target:'background',type:'PROCESS_IMAGE',url:string}
// Background -> offscreen: {target:'offscreen',type:'ENROLL'|'ANALYZE'|'RESOLVE_PREVIEW'|'CONFIRM_ENROLL',name?,url?,identities?:SavedIdentity[],faces?:EnrollPreviewFace[]}
// Every response: {ok:true, state?:BlockList,result?:ImageResult,identity?:SavedIdentity,preview?:EnrollPreview} OR {ok:false,error:string}.
// Self-seed flow: UI -> background {type:'RESOLVE_PREVIEW',name} -> {ok,preview}; then {type:'CONFIRM_ENROLL',name,faces} -> {ok,identity}.
// RESOLVE_PREVIEW/CONFIRM_ENROLL are extension-pages-only — EnrollPreviewFace carries raw embeddings.
// Background -> content: {target:'content',type:'STATE_CHANGED',revision:number,enabled:boolean}.
// Content -> background: {target:'background',type:'ANALYZE_FRAME',jpeg:ArrayBuffer} — forwarded to the
// offscreen document as {target:'offscreen',type:'ANALYZE_FRAME',jpeg}; response {ok,result:FrameResult}.
// A second ANALYZE_FRAME while one is in flight gets {ok:false,error:'busy'} immediately — a stale video
// frame is worthless, so frames are dropped, never queued. Sampled frames are analysed on device and
// discarded; they are never cached, persisted, or uploaded.
// Do not expose embeddings outside extension-owned pages. Never send photos to remote inference.
