# FaceBlock

**Support this project:** [ko-fi.com/djlougen](https://ko-fi.com/djlougen)

A Chromium extension that blocks a **person** across the web — not just their account. Type a name once, confirm the faces it finds, and FaceBlock covers matching faces with opaque black boxes in ordinary webpage images and videos. Everything runs on your device; nothing but the name you type is ever sent anywhere.

---

## Why this exists

Blocking someone on a social platform only removes *their account*. Their face still shows up in screenshots, reposts, memes, avatars, collages, thumbnails, embedded media, and video — posted by other people entirely. FaceBlock targets the thing you actually don't want to see: the face itself.

It also does not ask you to trust a server. There is no backend. Once installed, it works offline.

---

## Screenshots

Real captures of the built extension running in Chrome. Reproduction steps are in [Reproducing these screenshots](#reproducing-these-screenshots).

### 1. Manage blocked people

![FaceBlock options page showing two blocked identities with their reference-embedding counts and sources](docs/screenshots/01-options.png)

After blocking two people. Each entry records *embeddings*, not photographs — note "2 reference embeddings" and the source URLs it learned from. **Remove** unblocks; the **Protection** toggle pauses all scanning.

### 2. Confirm the faces it found

![FaceBlock confirmation panel showing eight gathered reference faces for Donald Trump, each with a source filename and confidence score](docs/screenshots/02-confirm.png)

Type a name and press **Block**. FaceBlock goes and finds reference photos of that person itself, then shows you what it intends to remember: here "checked 47 photos · 25 had a single face", narrowed to **8** faces that agree with each other across different angles and lighting. Each tile shows the source file name and a confidence score. Untick anything that looks wrong, then **Confirm**. Nothing is saved until you confirm.

### 3. Blocked faces covered on an ordinary page

![A plain webpage where two faces are covered by opaque black boxes and an unrelated third portrait is untouched](docs/screenshots/03-masked.png)

A plain webpage with no face-recognition code of its own. Two blocked people are covered; the unrelated control portrait is untouched. Boxes are grown slightly beyond the detected face so hair and edges don't identify anyone.

### 4. Blocked face covered in video

![A playing video with the moving face covered by a black box, and an unrelated portrait beside it left visible](docs/screenshots/04-video-masked.png)

Video works too. Coverage is designed to stay on a *moving* face without processing every frame, which is what keeps it usable on a normal laptop.

---

## Install

Requires Chrome or any Chromium-based browser (Edge, Brave, Arc…). Budget around 130 MB of disk: the face models and runtimes are bundled so that nothing is downloaded at runtime.

> **Not on the Chrome Web Store.** It installs as an unpacked extension, which needs Developer mode. Reading the code before running it is encouraged — that's rather the point.

### Build from source

```bash
git clone https://github.com/DJLougen/faceBlock.git
cd faceBlock
bun install
bun run build:extension
```

Output lands in `dist-extension/`.

### Or use the packaged zip

```bash
bun run package
```

produces `faceBlock-<version>.zip` at the repo root, with `manifest.json` at the archive root. Extract it anywhere.

### Load it in your browser

1. Open `chrome://extensions/`.
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select the `dist-extension/` folder (or the extracted zip folder).
4. Pin FaceBlock to the toolbar and click it to open the options page.

---

## Use it

1. **Block someone.** Click the toolbar icon, type a name — e.g. *Ada Lovelace* — and press **Block**.
2. **Confirm the faces.** FaceBlock shows the reference faces it found. Untick anything that isn't them, then **Confirm**. This step is deliberate: automatically found photos are occasionally the wrong person, so nothing is ever saved silently.
3. **Browse normally.** Visible images and videos on ordinary pages are checked on your device, and matching faces get covered.
4. **Manage the list** from the options page — every person has a **Remove** button, and **Protection** pauses everything.

### Choosing a name that will work

FaceBlock finds reference photos from public sources, so it works for **public figures, historical figures, and notable people** — not private individuals.

| Name | What happens |
| --- | --- |
| `Donald Trump` | 47 photos checked → 25 with a single face → 8 kept |
| `Ada Lovelace` | 7 faces found → 2 kept (most depictions of her are paintings, which the detector often can't read) |
| `Theo Browne` | **0 candidates** — those sources have no photos of him, so FaceBlock says so instead of guessing |

If a name isn't covered, FaceBlock tells you plainly rather than filling the list with unrelated faces. You can also enrol someone from your own reference photo.

---

## Privacy

- **The only thing that leaves your device is the name you type.** During enrolment it's sent to public reference-photo APIs to find candidate images — the same information you'd type into a search box.
- **Nothing else is transmitted.** No images, no face crops, no embeddings, no page media, no browsing history, no match results.
- **Reference photos are not kept.** After enrolment the downloads are discarded; only what's needed to recognise the person later stays, plus minimal metadata (name, source URLs, timestamp) in `chrome.storage.local`.
- **Video frames are examined on your device and discarded.**
- Face detection and matching run locally in the browser. No server, no account, no telemetry.

---

## Accuracy — read this before trusting it

**This is a research preview, not a privacy guarantee.**

- **Misses are expected.** Small, obscured, profile, or low-quality faces get missed.
- **Wrong-person matches are possible.**
- **Automatically found reference photos can be the wrong person.** Name lookups are ambiguous — a search for a public figure surfaces impersonators, same-name relatives, commemorative plaques, and AI-generated images. FaceBlock filters these and then asks you to confirm, but it cannot be perfect.
- **Images and video only.** Canvas-rendered content and browser-protected pages (`chrome://`, the Web Store) are not covered.
- **Some video can't be read at all.** A cross-origin video that blocks canvas access can't be examined; it's detected once and skipped rather than retried forever.
- **Coverage can lag briefly on abrupt motion** — a face that changes direction sharply may show a stale box for a moment.
- **Video costs battery.** All video work pauses when the tab is hidden.
- **Local only.** Blocked people live in `chrome.storage.local` and don't sync across devices.
- **Enrolment depends on public sources.** See the table above.

---

## Reproducing these screenshots

Everything above is reproducible. The captures in `docs/screenshots/` were taken from a live Chrome session running the built extension, and the fixture pages ship in this repo.

```bash
bun install
bun run build:extension
bun run dev          # fixture pages on http://127.0.0.1:5173
```

| Screenshot | How to reproduce |
| --- | --- |
| `01-options.png` | Load `dist-extension`, open the options page, and block `Theo Browne` and `Dwarkesh Patel` (both ship in `extension/references.json`, so they enrol immediately). |
| `02-confirm.png` | On the options page, type `Donald Trump` and press **Block**. Wait for it to finish gathering — it downloads roughly 48 photos and examines each — then screenshot the confirmation panel before confirming. |
| `03-masked.png` | With both people above blocked, open `http://127.0.0.1:5173/extension-test.html`. Two faces are covered; the control stays visible. |
| `04-video-masked.png` | With `Theo Browne` blocked, open `http://127.0.0.1:5173/video-test.html`. The moving face is covered. |

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

Bundled third-party models are recorded, with their origins, in `demo/public/models/provenance.json`. Sample-photo origins are in `demo/public/samples/sources.json`.

---

## Demo assets

`demo/public/samples/` contains photographs of public figures (Theo Browne, Dwarkesh Patel) and one public-domain portrait, used solely to demonstrate enrolment and control behaviour. Their origins are recorded in `sources.json`. If you fork this for anything beyond a demo, replace them with your own references.

---

## License

FaceBlock is **source-available, not open source**. It is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE).

Copyright 2026 Daniel Lougen. See [NOTICE](NOTICE) for the required notice.

**Free** for personal use, hobby projects, study, research, education, charities,
public research and health organisations, and government institutions.

**Not free for commercial use.** You may not sell FaceBlock, bundle it into a
product you sell, or use it for any commercial purpose. If you want to do that,
you need a commercial license — get in touch: [ko-fi.com/djlougen](https://ko-fi.com/djlougen).

### ⚠️ Read this before commercial use

The licence above covers **this repository's code**. It does not relicense the
third-party model weights that ship in `demo/public/models/`:

| Asset | Terms | Commercially usable? |
| --- | --- | --- |
| MediaPipe Face Landmarker (face detection) | Apache-2.0 | Yes |
| InsightFace `w600k_mbf` (face recognition) | **Non-commercial research only** | **No** |

So even a commercial licence from the author is not enough while that second
model is bundled — the weights themselves forbid commercial use, and the model
author's MIT-licensed *code* does not license their pretrained *weights*.

A commercially sellable build therefore requires replacing that model. Asset
provenance and checksums: `demo/public/models/provenance.json`.
