/* FaceBlock options page: name-in enrollment, blocklist management, toggle. */

import type { BlockList, EnrollPreview, EnrollPreviewFace, SavedIdentity } from "./protocol.ts";

declare const chrome: {
  runtime: {
    sendMessage(message: unknown): Promise<unknown>;
    lastError?: { message?: string };
  };
  storage: {
    onChanged: {
      addListener(
        callback: (changes: Record<string, { newValue?: unknown }>, area: string) => void,
      ): void;
    };
  };
};

interface BgResponse {
  ok?: boolean;
  error?: string;
  state?: BlockList;
  identity?: SavedIdentity;
  preview?: EnrollPreview;
}

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
}

const toggle = $("enabled-toggle") as HTMLInputElement;
const enabledLabel = $("enabled-label");
const form = $("block-form") as HTMLFormElement;
const nameInput = $("name-input") as HTMLInputElement;
const blockBtn = $("block-btn") as HTMLButtonElement;
const progress = $("progress");
const errorBox = $("error");
const list = $("identity-list");
const emptyNote = $("empty-note");
const previewSection = $("preview");

let state: BlockList = { identities: [], enabled: true, revision: 0 };
let enrolling = false;

async function send(message: Record<string, unknown>): Promise<BgResponse> {
  const response = (await chrome.runtime.sendMessage({
    ...message,
    target: "background",
  })) as BgResponse | undefined;
  if (response == null) {
    throw new Error(chrome.runtime.lastError?.message ?? "No response from FaceBlock.");
  }
  if (!response.ok) throw new Error(response.error ?? "Request failed.");
  return response;
}

function showError(text: string | null): void {
  errorBox.hidden = text === null;
  errorBox.textContent = text ?? "";
}

/**
 * Opened from the toolbar (`?popup=1`) versus the full options page.
 * A popup is a small panel, so the tutorial and footer are hidden by CSS and
 * the page is not meant to scroll for a screen and a half.
 */
const isPopup = new URLSearchParams(location.search).has("popup");
if (isPopup) document.body.classList.add("compact");

/**
 * The three-step tutorial is for someone who has never used this. Once a person
 * is blocked it has done its job, and leaving it under the list buried the
 * thing the user actually came for.
 */
function updateTutorial(): void {
  const intro = document.getElementById("getting-started");
  if (intro) intro.hidden = state.identities.length > 0 || !previewSection.hidden;
}

function render(): void {
  updateTutorial();
  toggle.checked = state.enabled;
  enabledLabel.textContent = state.enabled ? "Protection on" : "Protection off";

  emptyNote.hidden = state.identities.length > 0;
  list.replaceChildren(
    ...state.identities.map((identity) => {
      const item = document.createElement("li");
      item.className = "identity";

      const info = document.createElement("div");
      info.className = "identity-info";

      const name = document.createElement("strong");
      name.textContent = identity.name;
      info.append(name);

      const meta = document.createElement("span");
      meta.className = "muted";
      // Plain language, not vocabulary from the implementation: a person using
      // this wants to know it worked and when, not how many vectors there are.
      const n = identity.embeddings.length;
      meta.textContent = `Learned from ${n} photo${n === 1 ? "" : "s"} · added ${new Date(
        identity.createdAt,
      ).toLocaleDateString()}`;
      info.append(meta);

      if (identity.sources.length > 0) {
        // Raw source URLs are set behind a collapsed disclosure. Printing them
        // inline produced a wall of upload.wikimedia.org links that made a
        // working install look broken.
        const details = document.createElement("details");
        details.className = "sources";
        const summary = document.createElement("summary");
        summary.textContent = `Where the photos came from (${identity.sources.length})`;
        details.append(summary);
        for (const src of identity.sources) {
          const line = document.createElement("span");
          line.className = "muted";
          // Trim the API tracking parameters; they are noise to a reader.
          let shown = src;
          try {
            const u = new URL(src);
            shown = `${u.hostname}${u.pathname}`;
          } catch {
            /* keep the original string when it will not parse */
          }
          line.textContent = shown;
          details.append(line);
        }
        info.append(details);
      }

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => void removeIdentity(identity));

      item.append(info, remove);
      return item;
    }),
  );
}

async function removeIdentity(identity: SavedIdentity): Promise<void> {
  showError(null);
  try {
    const response = await send({ type: "REMOVE", id: identity.id });
    if (response.state) {
      state = response.state;
      render();
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

function setFormEnabled(enabled: boolean): void {
  nameInput.disabled = !enabled;
  blockBtn.disabled = !enabled;
}

function endEnroll(): void {
  enrolling = false;
  setFormEnabled(true);
}

function clearPreview(): void {
  previewSection.hidden = true;
  previewSection.replaceChildren();
  updateTutorial();
}

interface PreviewEntry {
  face: EnrollPreviewFace;
  checkbox: HTMLInputElement;
}

// One tile per kept face. The tile itself is focusable so keyboard users can
// review each candidate; Space/Enter on the tile toggles its checkbox.
function buildFaceTile(face: EnrollPreviewFace): { tile: HTMLElement; entry: PreviewEntry } {
  const tile = document.createElement("div");
  tile.className = "face-tile";
  tile.tabIndex = 0;

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = true;
  checkbox.className = "face-check";

  const img = document.createElement("img");
  img.src = face.thumbUrl ?? face.url;
  img.alt = face.filename;
  img.referrerPolicy = "no-referrer";
  img.loading = "lazy";

  const caption = document.createElement("span");
  caption.className = "face-caption";
  caption.textContent = `${face.filename} · ${Math.round(face.score * 100)}%`;

  // A broken thumbnail must not leave an empty image box: collapse the tile to
  // a single text line so the user can still uncheck it.
  img.addEventListener("error", () => {
    img.remove();
    caption.remove();
    const fallback = document.createElement("span");
    fallback.className = "face-fallback";
    fallback.textContent = `${face.filename} — image unavailable`;
    tile.append(fallback);
  });

  tile.addEventListener("keydown", (event) => {
    if (event.target !== tile) return;
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      checkbox.checked = !checkbox.checked;
    }
  });

  tile.append(checkbox, img, caption);
  return { tile, entry: { face, checkbox } };
}

function renderPreview(preview: EnrollPreview): void {
  const heading = document.createElement("h2");
  heading.textContent = `Is this ${preview.name}?`;

  const subline = document.createElement("p");
  subline.className = "muted";
  // Say what happened in the reader's terms: how many photos were looked at,
  // and that these are the clearest ones. "reference faces" and "had a single
  // face" are implementation vocabulary.
  const n = preview.kept.length;
  subline.textContent =
    `Found ${n} photo${n === 1 ? "" : "s"} of them out of ${preview.candidatesTried} looked at. ` +
    `Untick any that aren't them, then press Confirm.`;

  const entries: PreviewEntry[] = [];
  const grid = document.createElement("div");
  grid.className = "face-grid";
  for (const face of preview.kept) {
    const { tile, entry } = buildFaceTile(face);
    entries.push(entry);
    grid.append(tile);
  }

  const actions = document.createElement("div");
  actions.className = "preview-actions";

  if (preview.kept.length > 0) {
    const confirm = document.createElement("button");
    confirm.type = "button";
    confirm.textContent = "Yes, block " + preview.name.split(" ")[0];
    confirm.addEventListener("click", () => {
      const faces = entries.filter((entry) => entry.checkbox.checked).map((entry) => entry.face);
      if (faces.length === 0) {
        showError("Keep at least one photo, or press Cancel.");
        return;
      }
      void confirmEnroll(preview.name, faces);
    });
    actions.append(confirm);
  }

  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => {
    clearPreview();
    endEnroll();
    progress.hidden = true;
  });
  actions.append(cancel);

  previewSection.replaceChildren(heading, subline);

  if (preview.kept.length === 0) {
    const guidance = document.createElement("p");
    guidance.className = "muted";
    guidance.textContent =
      "No photos found for that name. Try their full name, or a different " +
      "spelling. FaceBlock can only learn from public photos of someone — it " +
      "works best for public figures, and won't guess.";
    previewSection.append(guidance);
  } else {
    previewSection.append(grid);
    if (preview.kept.length < 3) {
      const warning = document.createElement("p");
      warning.className = "preview-warning";
      warning.textContent =
        "Only a few photos were usable, so this may miss them sometimes. You can block them again later to pick up more.";
      previewSection.append(warning);
    }
  }

  if (preview.rejected.length > 0) {
    const details = document.createElement("details");
    details.className = "rejects";
    const summary = document.createElement("summary");
    summary.textContent = `Why ${preview.rejected.length} other photo${
      preview.rejected.length === 1 ? " was" : "s were"
    } skipped`;
    const list = document.createElement("ul");
    for (const reject of preview.rejected) {
      const item = document.createElement("li");
      item.textContent = `${reject.reason}`;
      list.append(item);
    }
    details.append(summary, list);
    previewSection.append(details);
  }

  previewSection.append(actions);
  previewSection.hidden = false;
  updateTutorial(); // after unhiding: it reads previewSection.hidden
  progress.textContent = "Review the gathered faces, then confirm.";
}

async function confirmEnroll(name: string, faces: EnrollPreviewFace[]): Promise<void> {
  showError(null);
  try {
    const response = await send({ type: "CONFIRM_ENROLL", name, faces });
    clearPreview();
    nameInput.value = "";
    progress.hidden = false;
    progress.textContent = `Blocked ${name}.`;
    if (response.state) {
      state = response.state;
      render();
    } else {
      const refreshed = await send({ type: "GET_STATE" });
      if (refreshed.state) state = refreshed.state;
      render();
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  } finally {
    endEnroll();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (enrolling) return;
  const name = nameInput.value.trim();
  if (!name) {
    showError("Enter a name to block.");
    return;
  }
  enrolling = true;
  setFormEnabled(false);
  showError(null);
  clearPreview();
  progress.hidden = false;
  progress.textContent = `Searching for photos of “${name}”…`;
  void send({ type: "BLOCK_NAME", name })
    .then((response) => {
      // Curated reference directory hit: keep the original one-shot behaviour.
      if (response.state) state = response.state;
      nameInput.value = "";
      render();
      endEnroll();
      progress.hidden = true;
    })
    .catch(() => {
      // Unknown name: fall back to the self-seed resolver and let the user
      // confirm the gathered faces before anything is enrolled.
      void send({ type: "RESOLVE_PREVIEW", name })
        .then((response) => {
          if (!response.preview) throw new Error("No preview returned.");
          renderPreview(response.preview);
        })
        .catch((resolveError: unknown) => {
          showError(
            resolveError instanceof Error ? resolveError.message : String(resolveError),
          );
          endEnroll();
          progress.hidden = true;
        });
    });
});

toggle.addEventListener("change", () => {
  const enabled = toggle.checked;
  showError(null);
  void send({ type: "SET_ENABLED", enabled })
    .then((response) => {
      if (response.state) state = response.state;
      render();
    })
    .catch((error: unknown) => {
      showError(error instanceof Error ? error.message : String(error));
      render(); // restore the real state
    });
});

// Keep multiple open copies of this page in sync with the stored blocklist.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  const next = changes["faceblockState"]?.newValue as BlockList | undefined;
  if (next && Array.isArray(next.identities) && next.revision !== state.revision) {
    state = next;
    render();
  }
});

void send({ type: "GET_STATE" })
  .then((response) => {
    if (response.state) state = response.state;
    render();
  })
  .catch((error: unknown) => {
    showError(error instanceof Error ? error.message : String(error));
    render();
  });
