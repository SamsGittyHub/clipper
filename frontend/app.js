/* CLIPZ review UI.
 *
 * Plain ES modules-free JavaScript against the FastAPI backend: submit a job,
 * poll its status, then render the clips for review, editing and download.
 */

const POLL_MS = 2000;
const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
const SUBMIT_LABEL = '<span aria-hidden="true">✂</span> Generate clips';

const STAGE_ORDER = [
  ["QUEUED", "Queued"],
  ["DOWNLOADING", "Download"],
  ["EXTRACTING_AUDIO", "Audio"],
  ["TRANSCRIBING", "Transcribe"],
  ["ANALYZING", "Analyze"],
  ["SELECTING_CLIPS", "Select"],
  ["PROCESSING_VIDEO", "Process"],
  ["GENERATING_CAPTIONS", "Captions"],
  ["RENDERING", "Render"],
  ["COMPLETED", "Done"],
];

const SIGNAL_LABELS = {
  ai_score: "AI semantic score",
  hook: "Hook strength",
  emotion: "Emotion",
  speech_density: "Speech density",
  duration_fit: "Duration suitability",
  confidence: "Transcript confidence",
  novelty: "Novelty",
};

const el = (id) => document.getElementById(id);
const accountName = (account) => account.username || account.display_name || account.id;
const state = {
  jobId: null,
  timer: null,
  clipsRendered: false,
  accounts: [],
  accountNames: new Map(),
  readyClips: 0,
  publishTimer: null,
  publishRows: new Map(),   // clip_id -> [publishing job rows]
  autoPublish: false,
  publishAsDraft: false,
  historyTimer: null,
};

/* ------------------------------------------------------------------ api */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail.length) detail = body.detail[0].msg;
      else if (body.error_code) detail = `${body.error_code}: ${body.detail}`;
    } catch { /* non-JSON error body */ }
    if (response.status === 401) {
      // The server wants a token and this browser has none (or a stale one).
      openAuthDialog();
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* --------------------------------------------------------------- feedback */

let toastTimer = null;

function toast(message, kind = "") {
  let node = el("toast");
  if (!node) {
    node = document.createElement("div");
    node.id = "toast";
    node.setAttribute("role", "status");
    node.setAttribute("aria-live", "polite");
    document.body.appendChild(node);
  }
  node.className = `toast ${kind}`;
  node.textContent = message;
  // Restart the transition even when a toast is already on screen.
  requestAnimationFrame(() => node.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 4000);
}

/* An in-app replacement for window.confirm(): same promise-shaped question,
 * styled like the rest of the interface instead of a browser chrome dialog. */
function confirmDialog(message, { confirmLabel = "Confirm", danger = true } = {}) {
  const dialog = el("confirm-dialog");
  if (dialog.open) return Promise.resolve(false);
  el("confirm-message").textContent = message;
  const ok = el("confirm-ok");
  ok.textContent = confirmLabel;
  ok.className = `button ${danger ? "danger" : "primary"}`;
  return new Promise((resolve) => {
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "ok"),
      { once: true }
    );
    dialog.showModal();
  });
}

/* ------------------------------------------------------------------ auth */

function openAuthDialog() {
  const dialog = el("auth-dialog");
  if (dialog.open) return;
  el("auth-status").textContent = "";
  dialog.showModal();
  el("auth-token").focus();
}

el("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = el("auth-status");
  status.className = "settings-status busy";
  status.textContent = "Checking...";
  try {
    await api("/api/auth", {
      method: "POST",
      body: JSON.stringify({ token: el("auth-token").value }),
    });
    el("auth-token").value = "";
    el("auth-dialog").close();
    bootstrap();
  } catch (error) {
    status.className = "settings-status bad";
    status.textContent = error.message;
  }
});

async function checkAuth() {
  try {
    // Deliberately not via api(): /api/auth is exempt from the token gate, so
    // asking it whether a token is needed can never itself trip the dialog.
    const auth = await fetch("/api/auth").then((response) => response.json());
    if (auth.required && !auth.authenticated) {
      openAuthDialog();
      return false;
    }
  } catch { /* treat an unreachable server as the health chip's problem */ }
  return true;
}

/* --------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const health = await api("/api/health");
    const chips = [
      [health.ffmpeg ? "FFmpeg ready" : "FFmpeg missing", health.ffmpeg ? "ok" : "warn"],
      [`Encoder ${health.encoder}`, ""],
      [`Whisper ${health.whisper_model}`, ""],
      [`AI ${health.ai_model}`, health.ai_provider === "heuristic" ? "warn" : "ok"],
    ];
    if (!health.js_runtime) {
      chips.push(["No JS runtime — downloads may fail", "warn"]);
    }
    const free = health.storage?.free_bytes;
    if (typeof free === "number" && free < 6e9) {
      chips.push([`${formatBytes(free)} disk free`, free < 1.5e9 ? "warn" : ""]);
    }
    if (health.reframe_mode === "smart") {
      chips.push([
        health.face_tracking_available ? "Smart crop" : "Smart crop (unavailable — using centre crop)",
        health.face_tracking_available ? "ok" : "warn",
      ]);
    }
    // The desktop build has no terminal to stop the server from.
    el("quit-btn").hidden = !health.packaged;
    el("health").innerHTML = chips
      .map(([text, cls]) => `<span class="chip ${cls}">${escapeHtml(text)}</span>`)
      .join("");
  } catch {
    el("health").innerHTML = '<span class="chip warn">Backend unreachable</span>';
  }
}

async function loadConfig() {
  try {
    const config = await api("/api/config");
    const select = el("caption_style");
    select.innerHTML = config.caption_styles
      .map(
        (style) =>
          `<option value="${style}"${style === config.default_caption_style ? " selected" : ""}>` +
          `${style[0].toUpperCase()}${style.slice(1)}</option>`
      )
      .join("");
    el("max_duration").max = config.max_clip_duration;
    el("max_duration").min = config.min_clip_duration;
    el("form-hint").textContent =
      `YouTube URLs. ${config.recommended_duration[0]}-${config.recommended_duration[1]}s ` +
      `performs best; output is ${config.output_resolution}.`;
  } catch { /* defaults in the markup still work */ }
}

/* ------------------------------------------------------------ submitting */

el("job-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = el("submit-btn");
  const errorBox = el("form-error");
  const resultsBox = el("batch-results");
  errorBox.hidden = true;
  resultsBox.hidden = true;
  resultsBox.innerHTML = "";

  const urls = el("url").value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (urls.length === 0) return; // required attribute already guards this

  const commonPayload = {
    clip_count: Number(el("clip_count").value),
    max_duration: Number(el("max_duration").value),
    caption_style: el("caption_style").value,
  };

  button.disabled = true;
  button.textContent = urls.length > 1 ? "Queueing..." : "Starting...";

  try {
    if (urls.length === 1) {
      // The common case stays a single simple request, watched immediately.
      const job = await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify({ url: urls[0], ...commonPayload }),
      });
      watchJob(job.job_id);
    } else {
      const results = await api("/api/jobs/batch", {
        method: "POST",
        body: JSON.stringify({ urls, ...commonPayload }),
      });
      renderBatchResults(results);
      const firstQueued = results.find((r) => r.job);
      if (firstQueued) watchJob(firstQueued.job.job_id);
    }
    loadJobHistory();
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = SUBMIT_LABEL;
  }
});

function renderBatchResults(results) {
  const box = el("batch-results");
  const queued = results.filter((r) => r.job).length;
  box.hidden = false;
  box.innerHTML =
    `<li><strong>${queued} of ${results.length} queued</strong></li>` +
    results
      .map(
        (r) =>
          `<li><span class="url">${escapeHtml(r.url)}</span>` +
          (r.job
            ? '<span class="ok">Queued</span>'
            : `<span class="bad">${escapeHtml(r.error || "Failed")}</span>`) +
          `</li>`
      )
      .join("");
}

/* -------------------------------------------------------------- polling */

function watchJob(jobId) {
  state.jobId = jobId;
  state.clipsRendered = false;
  el("cancel-btn").disabled = false;
  el("cancel-btn").textContent = "Cancel";
  el("progress-panel").hidden = false;
  el("clips-panel").hidden = true;
  el("clip-grid").innerHTML = "";
  el("job-error").hidden = true;
  el("publish-summary").hidden = true;
  el("publish-summary").dataset.signature = "";
  state.readyClips = 0;
  state.publishRows = new Map();
  if (state.publishTimer) clearInterval(state.publishTimer);
  state.publishTimer = null;
  pollPublishing();
  if (state.timer) clearInterval(state.timer);
  pollJob();
  state.timer = setInterval(pollJob, POLL_MS);
}

async function pollJob() {
  if (!state.jobId) return;
  let job;
  try {
    job = await api(`/api/jobs/${state.jobId}`);
  } catch (error) {
    el("progress-message").textContent = `Lost contact with the backend: ${error.message}`;
    return;
  }
  renderProgress(job);

  if (TERMINAL.has(job.status)) {
    clearInterval(state.timer);
    state.timer = null;
    loadJobHistory();
  }
  // Clips appear as they finish, so refresh while rendering too.
  if (job.clip_ready_count > 0 || job.status === "COMPLETED") {
    loadClips(job);
  }
}

function renderProgress(job) {
  const badge = el("status-badge");
  const finished = TERMINAL.has(job.status);
  badge.textContent = job.status;
  badge.className =
    "badge" +
    (job.status === "COMPLETED"
      ? " done"
      : job.status === "FAILED"
        ? " failed"
        : job.status === "CANCELLED"
          ? " cancelled"
          : "");

  el("progress-title").textContent = job.video?.title || "Processing";
  el("progress-source").textContent = job.video?.channel
    ? `${job.video.channel} · ${job.source_url}`
    : job.source_url;
  el("bar-fill").style.width = `${job.progress}%`;
  el("bar-fill").parentElement.classList.toggle("settled", finished);
  el("progress-message").innerHTML =
    `<strong>${job.progress}%</strong> · ${escapeHtml(job.message)}`;
  el("progress-eta").textContent =
    !finished && job.eta_seconds ? `about ${formatDuration(job.eta_seconds)} left` : "";

  // Cancel while it runs; offer the same settings again once it has stopped.
  el("cancel-btn").hidden = finished || job.cancel_requested;
  el("retry-btn").hidden = !finished || job.status === "COMPLETED";

  const currentIndex = STAGE_ORDER.findIndex(([key]) => key === job.status);
  el("stages").innerHTML = STAGE_ORDER.slice(0, -1)
    .map(([key, label], index) => {
      let cls = "";
      if (job.status === "COMPLETED") cls = "done";
      else if (job.status === "CANCELLED") cls = index < currentIndex ? "done" : "cancelled";
      else if (index === currentIndex) cls = "active";
      else if (currentIndex > index) cls = "done";
      return `<li class="${cls}">${label}</li>`;
    })
    .join("");

  const errorBox = el("job-error");
  if (job.status === "FAILED") {
    errorBox.textContent = `${job.error_code || "ERROR"} — ${job.error || job.message}`;
    errorBox.hidden = false;
  } else {
    errorBox.hidden = true;
  }
}

/* ----------------------------------------------------- cancel / retry */

el("cancel-btn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  if (!state.jobId) return;
  if (!(await confirmDialog("Stop this job? Clips already rendered are kept.", { confirmLabel: "Stop job" }))) return;
  button.disabled = true;
  button.textContent = "Cancelling...";
  try {
    renderProgress(await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" }));
    toast("Cancelling the job.", "");
  } catch (error) {
    toast(error.message, "bad");
    button.disabled = false;
    button.textContent = "Cancel";
  }
});

el("retry-btn").addEventListener("click", async (event) => {
  if (!state.jobId) return;
  event.currentTarget.disabled = true;
  try {
    const job = await api(`/api/jobs/${state.jobId}/retry`, { method: "POST" });
    watchJob(job.job_id);
    loadJobHistory();
    toast("Queued again.", "ok");
  } catch (error) {
    toast(error.message, "bad");
  } finally {
    event.currentTarget.disabled = false;
  }
});

/* ---------------------------------------------------------------- clips */

async function loadClips(job) {
  const clips = await api(`/api/jobs/${job.job_id}/clips`);
  if (!clips.length) return;

  el("clips-panel").hidden = false;
  const ready = clips.filter((clip) => clip.has_output).length;
  el("clips-count").textContent =
    `${ready} of ${clips.length} rendered · ranked by final score`;
  state.readyClips = ready;
  updatePostButtons();

  const grid = el("clip-grid");
  const existing = new Set([...grid.children].map((node) => node.dataset.clipId));
  clips.forEach((clip) => {
    if (existing.has(clip.id)) return;
    if (!clip.has_output && clip.status !== "FAILED") return;
    grid.appendChild(buildClipCard(clip));
  });
}

function buildClipCard(clip) {
  const node = el("clip-template").content.firstElementChild.cloneNode(true);
  node.dataset.clipId = clip.id;

  const query = (selector) => node.querySelector(selector);
  query(".rank").textContent = `#${clip.rank}`;
  query(".clip-name").textContent = clip.title || "Untitled clip";
  query(".clip-duration").textContent = `${Math.round(clip.duration)}s`;

  // The ring's sweep is the score itself, tinted by band; the number sits in
  // its centre.
  const ring = query(".score-ring");
  const score = Math.max(0, Math.min(100, clip.final_score));
  ring.style.setProperty("--p", score.toFixed(1));
  ring.classList.add(score >= 75 ? "high" : score >= 50 ? "mid" : "low");
  ring.setAttribute("aria-label", `Final score ${clip.final_score.toFixed(1)} of 100`);
  query(".final-score").textContent = clip.final_score.toFixed(1);
  query(".timing").textContent =
    `${formatTime(clip.start_time)} → ${formatTime(clip.end_time)} · ${clip.duration.toFixed(1)}s`;

  const metaEdit = query(".meta-edit");
  if (clip.has_output) {
    query(".clip-video video").src = clip.preview_url;
    query(".download").href = clip.download_url;
  } else {
    node.classList.add("failed");
    const error = document.createElement("p");
    error.className = "clip-error";
    error.textContent = clip.error || "This clip failed to render.";
    query(".clip-body").prepend(error);
    query(".download").setAttribute("aria-disabled", "true");
    query(".download").removeAttribute("href");
    query(".post").hidden = true;
  }

  query(".clip-title").value = clip.title || "";
  query(".clip-hook").value = clip.hook || "";
  query(".clip-description").value = clip.description || "";
  query(".clip-hashtags").value = (clip.hashtags || []).join(" ");
  query(".transcript-text").textContent = clip.transcript_text || "No transcript captured.";

  // Unsaved edits show a dot on the collapsed summary, so closing the card
  // cannot silently hide them.
  for (const cls of ["clip-title", "clip-hook", "clip-description", "clip-hashtags"]) {
    query("." + cls).addEventListener("input", () => metaEdit.classList.add("dirty"));
  }

  renderBreakdown(query(".breakdown"), clip);
  query(".why-btn").addEventListener("click", (event) => {
    const panel = query(".breakdown");
    panel.hidden = !panel.hidden;
    event.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
  });

  query(".post").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    if (!state.accounts.length) return;
    const ok = await confirmDialog(
      state.publishAsDraft
        ? `Upload this clip to ${state.accounts.length} account(s) as a TikTok draft?`
        : `Post this clip to ${state.accounts.length} account(s)?`,
      { confirmLabel: state.publishAsDraft ? "Upload draft" : "Post", danger: false }
    );
    if (!ok) return;
    button.disabled = true;
    try {
      await api(`/api/accounts/publish/job/${state.jobId}`, {
        method: "POST",
        body: JSON.stringify({ clip_ids: [clip.id] }),
      });
      toast("Posting to TikTok.", "ok");
      pollPublishing();
    } catch (error) {
      toast(error.message, "bad");
    } finally {
      button.disabled = false;
    }
  });

  query(".save").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const stateLabel = query(".save-state");
    button.disabled = true;
    stateLabel.textContent = "";
    try {
      await api(`/api/clips/${clip.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          title: query(".clip-title").value,
          hook: query(".clip-hook").value,
          description: query(".clip-description").value,
          hashtags: query(".clip-hashtags").value.split(/[\s,]+/).filter(Boolean),
        }),
      });
      metaEdit.classList.remove("dirty");
      stateLabel.textContent = "Saved";
      setTimeout(() => (stateLabel.textContent = ""), 2500);
    } catch (error) {
      stateLabel.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  renderClipPublishState(clip.id, node);
  return node;
}

function renderBreakdown(list, clip) {
  const breakdown = clip.score_breakdown || {};
  const signals = breakdown.signals || {};
  const contributions = breakdown.contributions || {};
  const weights = breakdown.weights || {};

  const rows = Object.keys(SIGNAL_LABELS)
    .filter((key) => key in signals)
    .map((key) => {
      const percent = Math.round((weights[key] || 0) * 100);
      const width = Math.max(0, Math.min(100, (signals[key] || 0) * 100));
      return (
        `<dt>${SIGNAL_LABELS[key]} <span class="muted">${percent}%</span></dt>` +
        `<dd class="signal-bar"><span style="width:${width.toFixed(1)}%"></span></dd>` +
        `<dd>+${(contributions[key] || 0).toFixed(1)}</dd>`
      );
    })
    .join("");

  list.innerHTML =
    rows +
    `<dt><strong>Final</strong></dt>` +
    `<dd class="signal-bar"><span style="width:${Math.min(100, clip.final_score).toFixed(1)}%"></span></dd>` +
    `<dd class="final-value"><strong>${clip.final_score.toFixed(1)}</strong></dd>` +
    (clip.reason ? `<dd class="reason">${escapeHtml(clip.reason)}</dd>` : "");
}

/* ------------------------------------------------------------- settings */

const settingsDialog = el("settings-dialog");
let presets = [];

function setStatus(message, kind = "") {
  const node = el("settings-status");
  node.textContent = message;
  node.className = `settings-status ${kind}`;
}

async function loadPublishingSettings() {
  try {
    const config = await api("/api/settings/publishing");
    state.autoPublish = config.auto_publish;
    state.publishAsDraft = config.publish_as_draft;
    el("settings-auto-publish").checked = config.auto_publish;
    el("settings-privacy").value = config.privacy_level;
    el("settings-draft").checked = config.publish_as_draft;
    el("settings-privacy").disabled = config.publish_as_draft;
    el("draft-note").hidden = !config.publish_as_draft;
    const sources = [
      `Privacy default from ${config.source.privacy_level === "runtime" ? "this panel" : config.source.privacy_level === "env" ? "DEFAULT_PRIVACY_LEVEL in the environment" : "the built-in default"}.`,
      config.auto_publish ? "Auto-post is ON." : "Auto-post is off.",
      config.publish_as_draft
        ? "Drafts are ON: clips land in each account's TikTok app instead of posting."
        : "",
    ];
    el("publishing-state").textContent = sources.filter(Boolean).join(" ");
  } catch (error) {
    el("publishing-state").textContent = error.message;
  }
  updateAutoChip();
}

function updateAutoChip() {
  el("auto-publish-chip").hidden = !state.autoPublish;
}

async function savePublishingSettings() {
  const config = await api("/api/settings/publishing", {
    method: "PUT",
    body: JSON.stringify({
      auto_publish: el("settings-auto-publish").checked,
      privacy_level: el("settings-privacy").value,
      publish_as_draft: el("settings-draft").checked,
    }),
  });
  state.autoPublish = config.auto_publish;
  state.publishAsDraft = config.publish_as_draft;
  updateAutoChip();
}

el("settings-draft").addEventListener("change", () => {
  const draft = el("settings-draft").checked;
  el("settings-privacy").disabled = draft;
  el("draft-note").hidden = !draft;
});

async function openSettings() {
  setStatus("");
  el("settings-key").value = "";
  loadPublishingSettings();
  try {
    const config = await api("/api/settings/ai");
    presets = config.presets || [];
    el("settings-preset").innerHTML = presets
      .map((preset) => `<option value="${preset.id}">${escapeHtml(preset.label)}</option>`)
      .join("");
    // Select the preset whose base URL is already in use, else Custom.
    const match = presets.find((preset) => preset.base_url === config.base_url);
    el("settings-preset").value = match ? match.id : "custom";
    el("settings-base-url").value = config.base_url || "";
    el("settings-model").value = config.model || "";
    el("settings-heuristic").checked = config.provider === "heuristic";
    el("key-state").textContent = config.configured
      ? `A key is saved (${config.key_hint}, from ${config.source.api_key}). ` +
        "Leave this blank to keep it."
      : config.env_key_present
        ? "No key saved here; OX_API_KEY from the environment will be used."
        : "No key configured — highlight selection falls back to the offline heuristic.";
  } catch (error) {
    setStatus(error.message, "bad");
  }
  loadAccounts();
  settingsDialog.showModal();
}

function settingsPayload({ includeEmptyKey = false } = {}) {
  const key = el("settings-key").value.trim();
  const payload = {
    base_url: el("settings-base-url").value.trim(),
    model: el("settings-model").value.trim(),
    provider: el("settings-heuristic").checked ? "heuristic" : "openai_compatible",
  };
  // Sending an empty key would clear the stored one, so only include it when
  // the user actually typed something.
  if (key || includeEmptyKey) payload.api_key = key;
  return payload;
}

el("settings-btn").addEventListener("click", openSettings);

el("quit-btn").addEventListener("click", async () => {
  const running = state.timer !== null;
  const message = running
    ? "A job is still running and will be stopped. Quit CLIPZ?"
    : "Quit CLIPZ?";
  if (!(await confirmDialog(message, { confirmLabel: "Quit" }))) return;
  try {
    await api("/api/quit", { method: "POST" });
  } catch { /* the server closing the connection is the expected outcome */ }
  document.body.innerHTML =
    '<main><section class="panel"><h2>CLIPZ has stopped</h2>' +
    '<p class="muted">You can close this window.</p></section></main>';
});
el("settings-close").addEventListener("click", () => settingsDialog.close());

el("settings-preset").addEventListener("change", (event) => {
  const preset = presets.find((item) => item.id === event.target.value);
  if (!preset || preset.id === "custom") return;
  el("settings-base-url").value = preset.base_url;
  if (preset.model) el("settings-model").value = preset.model;
});

el("settings-save").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  setStatus("Saving...", "busy");
  try {
    const config = await api("/api/settings/ai", {
      method: "PUT",
      body: JSON.stringify(settingsPayload()),
    });
    await savePublishingSettings();
    el("settings-key").value = "";
    setStatus(
      config.effective_provider === "heuristic"
        ? "Saved. No model is active, so the offline heuristic will be used."
        : `Saved. Jobs will use ${config.model}.`,
      "ok"
    );
    loadHealth();
    loadPublishingSettings();
  } catch (error) {
    setStatus(error.message, "bad");
  } finally {
    event.currentTarget.disabled = false;
  }
});

el("settings-test").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  setStatus("Testing...", "busy");
  try {
    const result = await api("/api/settings/ai/test", {
      method: "POST",
      body: JSON.stringify({
        api_key: el("settings-key").value.trim() || null,
        base_url: el("settings-base-url").value.trim() || null,
        model: el("settings-model").value.trim() || null,
      }),
    });
    setStatus(result.message, result.ok ? "ok" : "bad");
  } catch (error) {
    setStatus(error.message, "bad");
  } finally {
    event.currentTarget.disabled = false;
  }
});

el("settings-clear").addEventListener("click", async (event) => {
  if (
    !(await confirmDialog(
      "Remove the saved model key and AI settings from this machine?\n\n" +
        "Publishing settings are kept.",
      { confirmLabel: "Clear" }
    ))
  ) {
    return;
  }
  event.currentTarget.disabled = true;
  try {
    await api("/api/settings/ai", { method: "DELETE" });
    await openSettings();
    setStatus("Cleared. Environment settings apply again.", "ok");
    loadHealth();
  } catch (error) {
    setStatus(error.message, "bad");
  } finally {
    event.currentTarget.disabled = false;
  }
});

/* ------------------------------------------------------------- accounts */

async function loadAccounts() {
  const list = el("account-list");
  try {
    const accounts = await api("/api/accounts");
    state.accounts = accounts;
    state.accountNames = new Map(accounts.map((a) => [a.id, accountName(a)]));
    if (!accounts.length) {
      list.innerHTML =
        '<li class="account-empty">No accounts connected yet.</li>';
    } else {
      list.innerHTML = accounts
        .map(
          (account) => `
        <li>
          ${account.avatar_url ? `<img src="${escapeHtml(account.avatar_url)}" alt="">` : "<img alt=''>"}
          <span class="who">
            <span class="handle">${escapeHtml(account.username || account.display_name || account.id)}</span>
            <span class="sub ${account.last_error ? "bad" : ""}">${
              escapeHtml(account.last_error || "Connected")
            }</span>
          </span>
          <button type="button" data-disconnect="${account.id}">Disconnect</button>
        </li>`
        )
        .join("");
      list.querySelectorAll("[data-disconnect]").forEach((button) => {
        button.addEventListener("click", async () => {
          if (
            !(await confirmDialog("Disconnect this account?", { confirmLabel: "Disconnect" }))
          ) {
            return;
          }
          await api(`/api/accounts/${button.dataset.disconnect}`, { method: "DELETE" });
          loadAccounts();
        });
      });
    }
    el("accounts-note").textContent = accounts.length
      ? `Clips post to all ${accounts.length} connected account${accounts.length === 1 ? "" : "s"} at once.`
      : "Connect the accounts you want clips posted to.";
    updatePostButtons();
    renderAccountStrip();
  } catch (error) {
    list.innerHTML = `<li class="account-empty">${escapeHtml(error.message)}</li>`;
  }
}

/* One connect flow, reachable from Settings and from the topbar strip: open
 * the OAuth popup, and the callback page refreshes this page when it is done.
 * TikTok connects whichever account is logged in at tiktok.com, so the loop
 * for several accounts is: + -> approve (log out there first if needed) -> +. */
async function connectTikTok(button) {
  if (!button || button.disabled) return;
  button.disabled = true;
  try {
    const { authorize_url } = await api("/api/accounts/tiktok/connect");
    window.open(authorize_url, "clipz-tiktok", "width=620,height=760");
    setStatus("Approve the connection in the TikTok window.", "");
    toast("Approve the connection in the TikTok window.", "ok");
  } catch (error) {
    setStatus(error.message, "bad");
    toast(error.message, "bad");
  } finally {
    button.disabled = false;
  }
}

function renderAccountStrip() {
  const strip = el("account-strip");
  if (!state.accounts.length) {
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  const avatars = state.accounts
    .slice(0, 4)
    .map((account) =>
      account.avatar_url
        ? `<img class="strip-avatar" src="${escapeHtml(account.avatar_url)}" alt="" title="${escapeHtml(accountName(account))}">`
        : `<span class="strip-avatar" title="${escapeHtml(accountName(account))}"></span>`
    )
    .join("");
  const more =
    state.accounts.length > 4
      ? `<span class="strip-more">+${state.accounts.length - 4}</span>`
      : "";
  strip.innerHTML =
    `${avatars}${more}` +
    `<button type="button" class="strip-add" id="strip-add" aria-label="Connect another TikTok account">+</button>`;
  el("strip-add").addEventListener("click", (event) => connectTikTok(event.currentTarget));
}

el("connect-tiktok").addEventListener("click", (event) => connectTikTok(event.currentTarget));

// The OAuth callback page posts back when it finishes.
window.addEventListener("message", (event) => {
  if (event.data && event.data.clipz === "accounts-changed") {
    loadAccounts();
    setStatus("Account connected.", "ok");
  }
});

/* ------------------------------------------------------------ publishing */

function updatePostButtons() {
  const button = el("post-all-btn");
  const clipsReady = state.readyClips > 0;
  button.hidden = !clipsReady;
  button.disabled = !state.accounts.length;
  button.title = state.accounts.length
    ? `Post ${state.readyClips} clip(s) to ${state.accounts.length} account(s)`
    : "Connect a TikTok account in Settings first";

  document.querySelectorAll("#clip-grid .post").forEach((post) => {
    post.disabled = !state.accounts.length;
    post.title = state.accounts.length
      ? `Post to ${state.accounts.length} account(s)`
      : "Connect a TikTok account in Settings first";
  });
}

function groupPublishRows(rows) {
  const byClip = new Map();
  for (const row of rows) {
    if (!byClip.has(row.clip_id)) byClip.set(row.clip_id, []);
    byClip.get(row.clip_id).push(row);
  }
  return byClip;
}

function renderClipPublishState(clipId, node) {
  node = node || el("clip-grid").querySelector(`[data-clip-id="${CSS.escape(clipId)}"]`);
  if (!node) return;
  const badges = node.querySelector(".pub-badges");
  const rows = state.publishRows.get(clipId) || [];
  const signature = publishSignature(rows);
  if (badges.dataset.signature === signature) return;
  badges.dataset.signature = signature;
  badges.innerHTML = rows
    .map((row) => {
      const who = state.accountNames.get(row.account_id) || "account";
      const cls =
        row.status === "PUBLISHED"
          ? "ok"
          : row.status === "FAILED"
            ? "bad"
            : row.status === "PENDING" || row.status === "SCHEDULED"
              ? ""
              : "busy";
      const title = row.error || row.status.toLowerCase();
      const chip = `<span class="pub-chip ${cls}" title="${escapeHtml(title)}">${escapeHtml(who)}</span>`;
      return row.published_url
        ? `<a class="pub-chip ${cls}" href="${escapeHtml(row.published_url)}" target="_blank" rel="noreferrer" title="Published — ${escapeHtml(title)}">${escapeHtml(who)}</a>`
        : chip;
    })
    .join("");
}

let retryInFlight = false;

async function retryFailedPublishes() {
  // Send exactly the failed (clip, account) pairs: an empty body would make
  // the backend re-fan-out every READY clip, posting clips the clip cap left
  // out and clips to accounts connected after the original run.
  const failedClips = new Set();
  const failedAccounts = new Set();
  for (const rows of state.publishRows.values()) {
    for (const row of rows) {
      if (row.status !== "FAILED") continue;
      failedClips.add(row.clip_id);
      if (row.account_id) failedAccounts.add(row.account_id);
    }
  }
  if (!failedClips.size || retryInFlight) return;
  retryInFlight = true;
  const retry = el("retry-publish-btn");
  if (retry) {
    retry.disabled = true;
    retry.textContent = "Retrying...";
  }
  try {
    await api(`/api/accounts/publish/job/${state.jobId}`, {
      method: "POST",
      body: JSON.stringify({
        clip_ids: [...failedClips],
        account_ids: [...failedAccounts],
      }),
    });
    pollPublishing();
  } catch (error) {
    toast(error.message, "bad");
  } finally {
    retryInFlight = false;
  }
}

function publishSignature(rows) {
  return rows.map((r) => `${r.id}:${r.status}:${r.published_url || ""}`).join("|");
}

function renderPublishSummary(rows) {
  const box = el("publish-summary");
  // Rebuilding the DOM on every poll tick would drop keyboard focus and hover
  // from the Retry button and links, so only write when something changed.
  const signature = publishSignature(rows);
  if (box.dataset.signature === signature) return;
  box.dataset.signature = signature;
  box.hidden = false;

  const done = rows.filter((r) => ["PUBLISHED", "FAILED"].includes(r.status)).length;
  const failed = rows.filter((r) => r.status === "FAILED").length;

  const header =
    `<div class="row"><span class="label"><strong>` +
    (state.publishAsDraft ? "Uploading to TikTok drafts" : "Posting to TikTok") +
    `</strong>` +
    (state.autoPublish ? `<span class="auto-note">auto-post on</span>` : "") +
    `</span><span>${done} of ${rows.length} finished</span></div>` +
    (failed
      ? `<div class="row"><span class="bad">${failed} failed</span>` +
        `<button type="button" id="retry-publish-btn">Retry failed</button></div>`
      : "");

  const list = rows
    .map((row) => {
      const who = state.accountNames.get(row.account_id) || "account";
      const cls = row.status === "PUBLISHED" ? "ok" : row.status === "FAILED" ? "bad" : "";
      const detail =
        row.draft && row.status === "PUBLISHED"
          ? `<span class="muted">saved to drafts</span>`
          : row.published_url
            ? `<a href="${escapeHtml(row.published_url)}" target="_blank" rel="noreferrer">view</a>`
            : escapeHtml(row.error || row.status);
      return `<div class="row"><span>${escapeHtml(who)}</span><span class="${cls}">${detail}</span></div>`;
    })
    .join("");

  box.innerHTML = header + list;

  const retry = el("retry-publish-btn");
  if (retry) retry.addEventListener("click", retryFailedPublishes);
}

el("post-all-btn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const count = state.readyClips * state.accounts.length;
  const names = state.accounts.map(accountName).join(", ");
  const ok = await confirmDialog(
    state.publishAsDraft
      ? `Upload ${state.readyClips} clip(s) to ${state.accounts.length} account(s) as TikTok drafts — ${names}?\n\n` +
        `Each account's owner finishes the post from the TikTok app.`
      : `Post ${state.readyClips} clip(s) to ${state.accounts.length} account(s) — ${names}?\n\n` +
        `That is ${count} post(s), sent at the same time.`,
    { confirmLabel: state.publishAsDraft ? "Upload drafts" : "Post all", danger: false }
  );
  if (!ok) return;

  button.disabled = true;
  button.textContent = state.publishAsDraft ? "Uploading..." : "Posting...";
  try {
    await api(`/api/accounts/publish/job/${state.jobId}`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    pollPublishing();
  } catch (error) {
    el("publish-summary").hidden = false;
    el("publish-summary").innerHTML = `<span class="bad">${escapeHtml(error.message)}</span>`;
  } finally {
    button.disabled = false;
    button.textContent = "Post all to TikTok";
  }
});

// A watched job with no publishing rows yet is polled briefly (an auto-post
// may be starting); beyond that, and after repeated fetch errors, the poll
// stops instead of outliving its usefulness.
const PUBLISH_IDLE_TICKS = 20;       // ~1 min of empty responses
const PUBLISH_MAX_FETCH_ERRORS = 5;  // job deleted, or backend gone

async function pollPublishing() {
  if (!state.jobId) return;
  if (state.publishTimer) clearInterval(state.publishTimer);

  let idleTicks = 0;
  let errors = 0;
  const stop = () => {
    if (state.publishTimer) clearInterval(state.publishTimer);
    state.publishTimer = null;
  };

  const render = async () => {
    if (!state.jobId) return stop();
    let rows;
    try {
      rows = await api(`/api/accounts/publish/job/${state.jobId}`);
    } catch {
      errors += 1;
      if (errors >= PUBLISH_MAX_FETCH_ERRORS) stop();
      return;
    }
    if (!rows.length) {
      idleTicks += 1;
      if (idleTicks >= PUBLISH_IDLE_TICKS) stop();
      return;
    }
    idleTicks = 0;
    errors = 0;

    state.publishRows = groupPublishRows(rows);
    renderPublishSummary(rows);
    for (const clipId of state.publishRows.keys()) renderClipPublishState(clipId);

    const done = rows.filter((r) => ["PUBLISHED", "FAILED"].includes(r.status)).length;
    if (done === rows.length) stop();
  };

  render();
  state.publishTimer = setInterval(render, 3000);
}

/* -------------------------------------------------------------- history */

async function loadJobHistory() {
  try {
    const jobs = await api("/api/jobs?limit=8");
    if (!jobs.length) return;
    el("history-panel").hidden = false;
    el("job-list").innerHTML = jobs
      .map((job) => {
        const dot =
          job.status === "COMPLETED"
            ? "done"
            : job.status === "FAILED"
              ? "failed"
              : TERMINAL.has(job.status)
                ? ""
                : "running";
        const finished = TERMINAL.has(job.status);
        const when = timeAgo(job.created_at);
        const thumb = job.video?.thumbnail_url
          ? `<img class="thumb" src="${escapeHtml(job.video.thumbnail_url)}" alt="" loading="lazy">`
          : '<span class="thumb" aria-hidden="true"></span>';
        return `
      <li class="${job.job_id === state.jobId ? "current" : ""}${finished ? " settled" : ""}">
        ${thumb}
        <span class="mid">
          <span class="meta">
            <span class="status-dot ${dot}" aria-hidden="true"></span>
            <span class="title">${escapeHtml(job.video?.title || job.source_url)}</span>
          </span>
          <span class="bar mini"><span class="bar-fill" style="width:${job.progress || 0}%"></span></span>
        </span>
        <span class="muted">${job.status} · ${job.clip_ready_count} clip${job.clip_ready_count === 1 ? "" : "s"}${when ? ` · ${when}` : ""}</span>
        <span class="job-actions">
          <button type="button" data-open="${job.job_id}">Open</button>
          ${finished ? `<button type="button" data-delete="${job.job_id}" title="Delete this job and its files">Delete</button>` : ""}
        </span>
      </li>`;
      })
      .join("");

    el("job-list").querySelectorAll("[data-open]").forEach((button) => {
      button.addEventListener("click", () => watchJob(button.dataset.open));
    });
    el("job-list").querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", async () => {
        if (!(await confirmDialog("Delete this job and every file it produced?", { confirmLabel: "Delete" }))) {
          return;
        }
        button.disabled = true;
        try {
          await api(`/api/jobs/${button.dataset.delete}`, { method: "DELETE" });
          if (state.jobId === button.dataset.delete) {
            state.jobId = null;
            clearInterval(state.timer);
            state.timer = null;
            if (state.publishTimer) clearInterval(state.publishTimer);
            state.publishTimer = null;
            el("progress-panel").hidden = true;
            el("clips-panel").hidden = true;
          }
          toast("Job deleted.", "ok");
          loadJobHistory();
        } catch (error) {
          toast(error.message, "bad");
          button.disabled = false;
        }
      });
    });

    // Running jobs keep their bars moving without a job being watched.
    const running = jobs.some((job) => !TERMINAL.has(job.status));
    if (running && !state.historyTimer) {
      state.historyTimer = setInterval(loadJobHistory, 5000);
    } else if (!running && state.historyTimer) {
      clearInterval(state.historyTimer);
      state.historyTimer = null;
    }
  } catch { /* history is optional */ }
}

/* --------------------------------------------------------------- utils */

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function timeAgo(iso) {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "";
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 90) return "just now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.floor(minutes)}m ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.floor(hours)}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.round(total / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])
  );
}

function bootstrap() {
  loadHealth();
  loadConfig();
  loadJobHistory();
  loadAccounts();
  loadPublishingSettings();
}

checkAuth().then((allowed) => {
  if (allowed) bootstrap();
});
setInterval(loadHealth, 30000);
