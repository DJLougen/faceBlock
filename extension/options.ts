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
const earnToggle = $("earn-toggle") as HTMLInputElement;
const earnLabel = $("earn-label");
const enabledLabel = $("enabled-label");
const form = $("block-form") as HTMLFormElement;
const nameInput = $("name-input") as HTMLInputElement;
const blockBtn = $("block-btn") as HTMLButtonElement;
const progress = $("progress");
const errorBox = $("error");
const list = $("identity-list");
const emptyNote = $("empty-note");
const previewSection = $("preview");

let state: BlockList = { identities: [], enabled: true, earnEnabled: false, revision: 0 };
let enrolling = false;
// Separate from `enrolling` so a CONFIRM_ENROLL already in flight can never be
// entered twice, even if a click slips past the disabled buttons.
let confirming = false;

// Buttons rendered inside the identity list. They are tracked so the whole
// page's mutating actions can be disabled while an enrollment is in flight —
// otherwise a second RESOLVE_PREVIEW could overlap the first.
const rowButtons: HTMLButtonElement[] = [];

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
  earnToggle.checked = state.earnEnabled;
  earnLabel.textContent = state.earnEnabled ? "Preview on" : "Preview off";

  emptyNote.hidden = state.identities.length > 0;
  // The list is rebuilt from scratch, so drop references to the old buttons.
  rowButtons.length = 0;
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

      const actions = document.createElement("div");
      actions.className = "identity-actions";

      const refresh = document.createElement("button");
      refresh.type = "button";
      refresh.className = "secondary";
      refresh.textContent = "Refresh reference photos";
      refresh.title = "Gather reference photos again and review a replacement set";
      refresh.disabled = enrolling || confirming;
      refresh.addEventListener("click", () => startEnroll(identity.name, identity.id));

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "remove";
      remove.textContent = "Remove";
      remove.disabled = enrolling || confirming;
      remove.addEventListener("click", () => void removeIdentity(identity));

      rowButtons.push(refresh, remove);
      actions.append(refresh, remove);
      item.append(info, actions);
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

function setControlsEnabled(enabled: boolean): void {
  nameInput.disabled = !enabled;
  blockBtn.disabled = !enabled;
  for (const button of rowButtons) button.disabled = !enabled;
}

function endEnroll(): void {
  enrolling = false;
  setControlsEnabled(true);
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

// One tile per kept face. The tile is a <label> wrapping its checkbox, so the
// caption acts as the checkbox's accessible name and clicking anywhere on the
// tile — image, caption, padding — toggles it exactly once: the browser
// forwards a label click to the control only when the click did not land on
// the control itself. Keyboard access is the checkbox's own native
// focus/Space behaviour, so there is no custom key handler to double-toggle.
function buildFaceTile(face: EnrollPreviewFace): { tile: HTMLElement; entry: PreviewEntry } {
  const tile = document.createElement("label");
  tile.className = "face-tile";

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

  tile.append(checkbox, img, caption);
  return { tile, entry: { face, checkbox } };
}

function renderPreview(preview: EnrollPreview, requestedIdentityId?: string): void {
  // The backend echoes the identity this preview would update: either the one
  // we asked to refresh, or one it matched by name. Fall back to the requested
  // id so a refresh never silently creates a second identity.
  const identityId = preview.identityId ?? requestedIdentityId;
  const isRefresh = identityId != null;

  const heading = document.createElement("h2");
  heading.textContent = isRefresh
    ? `Reference photos for ${preview.name}?`
    : `Is this ${preview.name}?`;

  const subline = document.createElement("p");
  subline.className = "muted";
  // Say what happened in the reader's terms: how many photos were looked at,
  // and that these are the clearest ones. "reference faces" and "had a single
  // face" are implementation vocabulary.
  const n = preview.kept.length;
  subline.textContent =
    `Found ${n} photo${n === 1 ? "" : "s"} of them out of ${preview.candidatesTried} looked at. ` +
    (isRefresh
      ? "Untick any that aren't them, then press Confirm — the photos you keep replace the current set."
      : "Untick any that aren't them, then press Confirm.");

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
    confirm.textContent = isRefresh
      ? "Yes, update " + preview.name.split(" ")[0]
      : "Yes, block " + preview.name.split(" ")[0];
    confirm.addEventListener("click", () => {
      const faces = entries.filter((entry) => entry.checkbox.checked).map((entry) => entry.face);
      if (faces.length === 0) {
        showError("Keep at least one photo, or press Cancel.");
        return;
      }
      // One persistence request at a time: a second click while CONFIRM_ENROLL
      // is in flight must not enqueue a duplicate write.
      confirm.disabled = true;
      cancel.disabled = true;
      void confirmEnroll(preview.name, faces, identityId).then((ok) => {
        if (!ok) {
          confirm.disabled = false;
          cancel.disabled = false;
        }
      });
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
    guidance.textContent = isRefresh
      ? "No usable photos found this time — the current set is unchanged. " +
        "You can try again later, or press Cancel."
      : "No photos found for that name. Try their full name, or a different " +
        "spelling. FaceBlock can only learn from public photos of someone — it " +
        "works best for public figures, and won't guess.";
    previewSection.append(guidance);
  } else {
    previewSection.append(grid);
    if (preview.kept.length < 3) {
      const warning = document.createElement("p");
      warning.className = "preview-warning";
      warning.textContent = isRefresh
        ? "Only a few photos were usable, so this may miss them sometimes. " +
          "You can refresh again later to try for more."
        : "Only a few photos were usable, so this may miss them sometimes. " +
          "Once they're blocked, use Refresh reference photos on their entry below to look for more.";
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

/**
 * Persist the confirmed faces. Returns true when the enrollment was saved —
 * the caller uses that to decide whether the preview's action buttons should
 * be re-enabled (on failure the preview stays open so the user can retry or
 * cancel).
 */
async function confirmEnroll(
  name: string,
  faces: EnrollPreviewFace[],
  identityId?: string,
): Promise<boolean> {
  if (confirming) return false;
  confirming = true;
  showError(null);
  try {
    const response = await send({
      type: "CONFIRM_ENROLL",
      name,
      faces,
      ...(identityId ? { identityId } : {}),
    });
    clearPreview();
    nameInput.value = "";
    progress.hidden = false;
    progress.textContent = identityId ? `Updated ${name}.` : `Blocked ${name}.`;
    if (response.state) {
      state = response.state;
      render();
    } else {
      const refreshed = await send({ type: "GET_STATE" });
      if (refreshed.state) state = refreshed.state;
      render();
    }
    return true;
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
    return false;
  } finally {
    confirming = false;
    endEnroll();
  }
}

/**
 * Every enrollment — typed name, curated directory hit, or refresh of a saved
 * identity — goes through RESOLVE_PREVIEW so nothing is persisted without the
 * user confirming the gathered faces. `identityId` marks a refresh: the
 * confirmed faces replace that identity's reference set.
 */
function startEnroll(name: string, identityId?: string): void {
  if (enrolling) return;
  enrolling = true;
  setControlsEnabled(false);
  showError(null);
  clearPreview();
  progress.hidden = false;
  progress.textContent = identityId
    ? `Gathering reference photos for “${name}”…`
    : `Searching for photos of “${name}”…`;
  void send({
    type: "RESOLVE_PREVIEW",
    name,
    ...(identityId ? { identityId } : {}),
  })
    .then((response) => {
      if (!response.preview) throw new Error("No preview returned.");
      renderPreview(response.preview, identityId);
    })
    .catch((resolveError: unknown) => {
      showError(
        resolveError instanceof Error ? resolveError.message : String(resolveError),
      );
      endEnroll();
      progress.hidden = true;
    });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (enrolling) return;
  const name = nameInput.value.trim();
  if (!name) {
    showError("Enter a name to block.");
    return;
  }
  // Re-typing someone already blocked used to silently re-enroll them. Route
  // it through the same refresh path as the Refresh photos button: gather new
  // candidates, preview them, and only replace the reference set on confirm.
  const existing = state.identities.find(
    (identity) => identity.name.toLowerCase() === name.toLowerCase(),
  );
  startEnroll(existing ? existing.name : name, existing?.id);
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

earnToggle.addEventListener("change", () => {
  const earnEnabled = earnToggle.checked;
  showError(null);
  void send({ type: "SET_EARN_ENABLED", earnEnabled })
    .then((response) => {
      if (response.state) state = response.state;
      render();
    })
    .catch((error: unknown) => {
      showError(error instanceof Error ? error.message : String(error));
      render();
    });
});

/** True when another tab/popup wrote storage this page should reflect. */
function storageStateOutOfSync(next: BlockList): boolean {
  if (!Array.isArray(next.identities)) return false;
  return (
    next.revision !== state.revision ||
    next.enabled !== state.enabled ||
    next.earnEnabled !== state.earnEnabled
  );
}

// Keep multiple open copies of this page in sync with the stored blocklist.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;
  const next = changes["faceblockState"]?.newValue as BlockList | undefined;
  if (next && storageStateOutOfSync(next)) {
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
