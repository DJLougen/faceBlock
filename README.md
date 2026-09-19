# FaceBlock

**Support this project:** [ko-fi.com/djlougen](https://ko-fi.com/djlougen)

A Chromium extension that blocks a **person** across the web — not just their account. Type a name once, confirm the faces it finds, and FaceBlock draws opaque black masks over matching faces in ordinary webpage images and videos. All matching runs on your device; nothing but the name you type is ever sent anywhere.

---

## Why this exists

Blocking someone on a social platform only removes *their account*. The same face still shows up in screenshots, reposts, memes, avatars, collages, thumbnails, embedded media, and video — posted by other people entirely. FaceBlock targets the actual thing you don't want to see: the face.

And it does it without shipping your browsing to a server. Face detection, embedding, and matching all run locally in the browser via ONNX Runtime Web. There is no backend.

---

## Screenshots

All four are real captures of the built extension running in Chrome, not mockups. Reproduction steps are in [Reproducing these screenshots](#reproducing-these-screenshots).

### 1. Manage blocked people

![FaceBlock options page showing two blocked identities with their reference-embedding counts and sources](docs/screenshots/01-options.png)

The options page after blocking two people. Each identity stores *embeddings*, not photographs — note "2 reference embeddings" and the recorded source URLs. **Remove** unblocks; the **Protection** toggle pauses all scanning.

### 2. Confirm the faces it found

![FaceBlock confirmation panel showing eight gathered reference faces for Donald Trump with per-face scores and a list of rejected photos](docs/screenshots/02-confirm.png)

Type a name and press **Block**. FaceBlock gathers candidate photos, detects and clusters the faces, and shows exactly what it kept: here "checked 47 photos · 25 had a single face", narrowed to **8** mutually-consistent faces spanning different angles and lighting. Each tile shows the source filename and its context score. Uncheck any tile that looks wrong, then **Confirm**. Nothing is enrolled until you confirm.

### 3. Blocked faces masked on an ordinary page

![A plain webpage where two faces are covered by opaque black boxes and an unrelated third portrait is untouched](docs/screenshots/03-masked.png)

A plain webpage with no face-recognition code of its own. Two blocked people are masked; the unrelated control portrait is untouched. The masks are opaque black rectangles grown slightly beyond the detected face box so hair and edges don't identify anyone.

### 4. Blocked face masked in video

![A playing video with the moving face covered by a black box, and an unrelated portrait beside it left visible](docs/screenshots/04-video-masked.png)

Video works too — with a deliberate design constraint. Recognising every frame is far too slow (measured on an M3 Max: **~59 ms** for one detect + align + embed), so FaceBlock recognises at a low adaptive rate (250–800 ms) and then **tracks** the box between recognitions via `requestVideoFrameCallback`. Masks follow the face at full video rate without running recognition per frame.

---

## Install

Requires Chrome or any Chromium-based browser (Edge, Brave, Arc…). The extension needs ~130 MB of local disk because the face models and WASM runtimes are bundled — there are no runtime downloads.

> **Not on the Chrome Web Store.** Install it as an unpacked extension, which requires Developer mode. Read the code first if you like — that's rather the point.

### Build from source

```bash
git clone https://github.com/DJLougen/faceBlock.git
cd faceBlock
bun install
bun run build:extension
```

Output goes to `dist-extension/`.

### Or use the packaged zip

```bash
bun run package
```

produces `faceBlock-<version>.zip` at the repo root with `manifest.json` at the archive root. Extract it anywhere.

### Load it into your browser

1. Open `chrome://extensions/`.
2. Turn on **Developer mode** (top-right toggle).
3. Click **Load unpacked** and pick the `dist-extension/` folder (or the extracted zip folder).
4. Pin FaceBlock to the toolbar and click it to open the options page.

---

## Use it

1. **Block someone.** Click the toolbar icon, type a name — e.g. *Ada Lovelace* — and press **Block**.
2. **Confirm the faces.** FaceBlock shows the reference faces it gathered. Uncheck anything that isn't them, then **Confirm**. This step exists because automatic photo gathering is imperfect: the plan and the code both refuse to enrol silently.
3. **Browse normally.** Visible images and videos on ordinary pages are scanned on your device and matching faces get masked.
4. **Manage the list** from the options page — each person has a **Remove** button, and the **Protection** toggle pauses everything.

### Choosing a name that will work

Discovery uses Wikipedia, Wikidata, and Wikimedia Commons, so it works for **public figures, historical figures, and notable people** — not private individuals.

| Name | Result |
| --- | --- |
| `Donald Trump` | 47 photos checked → 25 single-face → 8 kept |
| `Ada Lovelace` | 7 faces found → 2 kept (most depictions of her are paintings, which the face detector often can't read) |
| `Theo Browne` | **0 candidates** — those sources have no photos of him, so FaceBlock says so instead of guessing |

For anyone the public sources don't cover, use the bundled curated directory (`extension/references.json`) or your own reference photo. FaceBlock applies a relevance floor and will report *"nothing found"* rather than pad the list with unrelated faces.

---

## How it works

```
type a name
   ↓  Wikipedia / Wikidata / Wikimedia Commons  (name only, nothing else)
candidate photos
   ↓  score by file name + caption          ← a filename is real evidence
   ↓  download, detect faces, embed
   ↓  cluster: keep the dominant group, reject outliers
   ↓  farthest-point sampling → maximally different angles & lighting
you confirm
   ↓  store embeddings only (never the photos)
browse → images and video scanned on device → masks drawn
```

- **`extension/background.ts`** — MV3 service worker; owns the blocklist, routes all messages.
- **`extension/content.ts`** — content script; observes the DOM, queues images, samples video frames, draws masks into a closed shadow root the page cannot read or restyle.
- **`extension/offscreen.ts`** — offscreen document; the only place inference runs.
- **`extension/options.ts`** — options page: name-in enrolment, confirmation panel, blocklist.
- **`src/resolve/`** — reference discovery (Wikipedia/Wikidata/Commons), context scoring, embedding clustering and outlier rejection.
- **`src/tracking/`** — the video box tracker: IoU association with a centre-distance fallback, so a head that outruns overlap between recognitions doesn't spawn duplicate tracks.
- **`src/cv/`** — detection, alignment, embedding, rasterisation.
- **`src/matching/`** — cosine similarity, thresholds, prototype selection.
- **`src/overlay/`** — source→viewport coordinate mapping (`object-fit`/`object-position` aware).

### Models (both bundled, both local)

| Purpose | Model | Size |
| --- | --- | --- |
| Face detection | MediaPipe Face Landmarker | ~3.6 MB |
| Face embedding | InsightFace `w600k_mbf` | ~13 MB, **512-dim** |

---

## Privacy

- **The only thing that leaves your device is the name you type.** During enrolment it's sent to the Wikipedia / Wikidata / Wikimedia Commons APIs to find candidate photos — the same information you'd type into a search box.
- **Nothing else is transmitted.** No images, no face crops, no embeddings, no page media, no browsing history, no match results.
- **Reference photos are not kept.** After enrolment the downloads are discarded; only embeddings and minimal metadata (name, source URLs, timestamp) live in `chrome.storage.local`.
- **Video frames are analysed on device and discarded.** A sampled frame is reduced to a small JPEG, matched locally, and released.
- Inference runs in an offscreen document on the WASM SIMD backend, single-threaded, so no special cross-origin-isolation headers are needed.

---

## Accuracy — read this before trusting it

**This is a research preview, not a privacy guarantee.** The cosine-similarity threshold is an operating point, not a calibrated number.

- **False negatives are expected.** Small, obscured, profile, or low-quality faces get missed.
- **False positives are possible.** The wrong person can be masked.
- **Gathered reference photos can be the wrong person.** Name lookups are ambiguous (a search for a public figure surfaces impersonator photos, same-name relatives, plaques, and AI-generated images). FaceBlock suppresses those by text and clustering, then asks you to confirm — but it cannot be perfect.
- **Images and video only.** Canvas-rendered content and browser-protected pages (`chrome://`, the Web Store) aren't scanned.
- **Some video can't be read at all.** A cross-origin video without CORS taints the canvas, so its pixels are unreadable. It's detected once and skipped rather than retried forever.
- **Video masks can lag on abrupt motion.** A box is coasted between recognitions; a face that changes direction sharply can briefly show a stale box.
- **Video costs battery.** Work is bounded to 8 videos (largest first), one frame in flight per video, and pauses when the tab is hidden.
- **Local only.** Identities live in `chrome.storage.local` and don't sync across devices.
- **Enrolment depends on public sources.** See the table above.

---

## Reproducing these screenshots

Everything above is reproducible. `docs/screenshots/` was captured from a live Chrome session driving `dist-extension`, and the fixture pages ship in the repo.

```bash
bun install
bun run build:extension
bun run dev          # serves the fixture pages on http://127.0.0.1:5173
```

| Screenshot | How to reproduce |
| --- | --- |
| `01-options.png` | Load `dist-extension`, open the options page, and block `Theo Browne` and `Dwarkesh Patel` (both are in `extension/references.json`, so they enrol straight away). |
| `02-confirm.png` | On the options page, type `Donald Trump` and press **Block**. Wait for the gathering to finish — it downloads ~48 photos and runs inference on each — then screenshot the confirmation panel before confirming. |
| `03-masked.png` | Open `http://127.0.0.1:5173/extension-test.html` with both people above blocked. Two faces are masked; the control stays visible. |
| `04-video-masked.png` | Open `http://127.0.0.1:5173/video-test.html` while `Theo Browne` is blocked. The video is generated (see below); the moving face is masked. |

The video fixture is generated, not downloaded:

```bash
cd demo/public
ffmpeg -y -f lavfi -i "color=c=0x141414:s=780x660:d=6:r=25" -loop 1 -i samples/theo.jpg \
  -filter_complex "[1:v]format=rgba[face];[0:v][face]overlay=x='150+40*sin(t*2)':y=100:shortest=1,format=yuv420p" \
  -c:v libx264 -preset veryfast -movflags +faststart video-theo.mp4
```

---

## Development

```bash
bun install
bun test tests            # 112 tests
bun run typecheck         # tsc --noEmit
bun run build:extension   # → dist-extension/
bun run package           # → faceBlock-<version>.zip
bun run dev               # fixture pages on :5173
```

---

## Demo assets

`demo/public/samples/` contains photographs of public figures (Theo Browne, Dwarkesh Patel) and one unrelated public-domain portrait, used solely to demonstrate enrolment and control behaviour. Each file's origin is recorded in `demo/public/samples/sources.json`. If you fork this for anything beyond a demo, replace them with your own references.

---

## License

Research preview — not for production use. See `LICENSE` if present; otherwise treat as all-rights-reserved pending a licence decision.
