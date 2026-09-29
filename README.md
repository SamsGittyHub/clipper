# CLIPZ

Turns a long-form video URL into ranked, captioned, publish-ready 9:16 clips.

Built to the [product requirements](clipper_PRD.md), on its central principle:

> **AI makes semantic decisions; deterministic software performs media processing.**

The model reads a transcript and judges which moments are worth clipping. It
never touches a file path, a timestamp it can't justify, or an encoder setting
— conventional code downloads, transcribes, validates, cuts, captions and
renders. That split is why a bad model response degrades quality instead of
corrupting output.

## What works today

The full MVP (PRD §7) — paste a URL, get reviewable clips:

| Stage | Implementation |
|---|---|
| Intake + async jobs | FastAPI, 12 job states, progress, ETA and message on every poll |
| Batch intake | One job per URL from a single multi-line paste; bad lines don't block the rest |
| Download | `yt-dlp`, best ≤1080p rendition, parallel fragments, optional cookies |
| Transcription | `faster-whisper` with word-level timings, batched on CUDA |
| Highlight detection | Any OpenAI-compatible endpoint, or an offline heuristic provider |
| Long sources | Transcripts past the prompt budget are analysed in concurrent sections |
| Validation | Bounds checks, clamping, word-boundary snapping, speech checks |
| Dedup + ranking | Overlap ratio dedup, 7-signal weighted score with a full breakdown |
| Rendering | FFmpeg, smart face-tracked 9:16 reframing, clips encoded in parallel |
| Captions | ASS from word timings, 5 styles, burned in with libass |
| Review UI | Preview, score explanation, metadata editing, MP4 download |
| Job control | Cancel mid-run, re-run with the same settings, delete a job and its files |
| Durability | A restart re-queues what never started and fails what was interrupted |
| Publishing | TikTok Direct Post to every connected account at once |

Still post-MVP: scheduling, analytics. The old `POST /api/clips/{id}/publish`
stub still returns `501`; the working endpoint is
`POST /api/accounts/publish/job/{job_id}`.

## Install as a desktop app (AppImage)

A single portable file — no install, no root, no Python setup:

```bash
chmod +x CLIPZ-x86_64.AppImage
./CLIPZ-x86_64.AppImage
```

It picks a free port, starts the server, and opens the UI (a chromeless window
when a Chromium-family browser is present, otherwise a normal tab). A **Quit**
button appears in the top bar, since launching from a menu entry leaves no
terminal to interrupt.

FFmpeg is the one thing not bundled — the app checks at startup and shows a
dialog with the exact command if it is missing:

```bash
sudo apt install ffmpeg
```

Everything it writes lives outside the AppImage:

| What | Where |
|---|---|
| Clips, downloads, database | `~/.local/share/CLIPZ/` |
| Optional settings file | `~/.config/clipz/config.env` |
| API key from the settings panel | `~/.local/share/CLIPZ/runtime_config.json` |

### Upgrading from ClipForge AI

CLIPZ was previously called ClipForge AI. The first run adopts the old data
directory, database and config automatically, and repoints the absolute paths
stored against every clip so previews and downloads keep working:

```text
~/.local/share/ClipForgeAI/  ->  ~/.local/share/CLIPZ/
          clipforge.db       ->  clipz.db
~/.config/clipforge/         ->  ~/.config/clipz/
```

Nothing is overwritten: if a CLIPZ directory already exists, the old one is
left untouched and you can merge it by hand. This matters beyond convenience,
because connected TikTok accounts live in that database as encrypted refresh
tokens, and starting fresh would mean reconnecting each one.

Options: `--port N`, `--no-browser`, `--storage /path`. To get a menu entry and
icon, install [AppImageLauncher](https://github.com/TheAssassin/AppImageLauncher)
or run the AppImage once with it installed.

### Building it yourself

```bash
./packaging/build-appimage.sh      # writes dist/CLIPZ-x86_64.AppImage
```

The build downloads a relocatable CPython, installs the runtime dependencies
into it, and packs everything with `appimagetool`. Downloads are cached in
`build/cache/`, so rebuilds are quick. Note it packs the fuse3-capable AppImage
runtime deliberately: current distros (Ubuntu 24.04 included) ship `libfuse3`
but not the `libfuse2` the classic runtime needs.

## Running from source

## Requirements

- Python 3.11+
- FFmpeg and ffprobe on `PATH` (with libass — Ubuntu's `ffmpeg` package has it)
- ~2 GB disk for the Whisper model on first run
- Optional: an NVIDIA GPU for faster transcription and encoding

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # optional: add OX_API_KEY for AI selection
./.venv/bin/uvicorn app.main:app --app-dir backend --port 8000
```

Open <http://localhost:8000>. `GET /api/health` reports which encoder, Whisper
model and highlight provider this install resolved to; `/docs` has the full
OpenAPI surface.

### Configuring the model from the browser

Click **Settings** in the top bar. The panel takes an API key, a base URL and a
model name for any endpoint that speaks the OpenAI chat-completions protocol,
with presets for OpenAI, OpenRouter, Groq, Together, Ollama and Ox as starting
points — the model field stays editable, so anything the endpoint serves works.
**Test connection** makes one tiny live call so a wrong key or model name
surfaces immediately instead of mid-job.

What is entered there wins over `OX_API_KEY`; `Clear` removes it and the
environment applies again. The panel also has a toggle to force the offline
heuristic without deleting the key.

The key is written to `storage/runtime_config.json` with `0600` permissions and
is **never returned to the browser** — the panel shows only a masked hint
(`sk-abc...wxyz`), and it is scrubbed from connection-test error messages. Note
that this is a deliberate relaxation of the PRD's "keys in environment
variables only": the app is single-user and local, and a key typed into the
panel is readable by anything that can read that file or reach this API. On a
shared or exposed host, prefer `OX_API_KEY` and leave the panel empty.

### Without an API key

With no `OX_API_KEY` set, the heuristic provider runs instead: it scores
transcript windows on speech density, hook phrasing, emotional language,
questions and sentence completeness. The pipeline is fully functional and
selection is noticeably blunter — it is the fallback, not the destination. It
also catches a failing model at runtime, so an outage degrades the job rather
than killing it.

## Using it

```bash
curl -X POST localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{
  "url": "https://www.youtube.com/watch?v=...",
  "clip_count": 5,
  "max_duration": 60,
  "caption_style": "dynamic"
}'

curl localhost:8000/api/jobs/job_abc123          # status, progress, ETA
curl localhost:8000/api/jobs/job_abc123/clips    # ranked clips
curl localhost:8000/api/jobs/job_abc123/logs     # per-stage timings
curl -O localhost:8000/api/clips/clip_xyz/download

# Job control
curl -X POST localhost:8000/api/jobs/job_abc123/cancel   # stop it now
curl -X POST localhost:8000/api/jobs/job_abc123/retry    # same settings, new job
curl -X DELETE localhost:8000/api/jobs/job_abc123        # remove it and its files

# Queue several sources at once -- one job per URL, a bad line does not
# block the rest (the UI does this too: paste one URL per line)
curl -X POST localhost:8000/api/jobs/batch -H 'Content-Type: application/json' -d '{
  "urls": ["https://www.youtube.com/watch?v=aaa", "https://www.youtube.com/watch?v=bbb"],
  "clip_count": 5,
  "max_duration": 60
}

# AI configuration (the settings panel uses these)
curl localhost:8000/api/settings/ai
curl -X PUT localhost:8000/api/settings/ai -H 'Content-Type: application/json' \
  -d '{"api_key":"sk-...","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini"}'
curl -X POST localhost:8000/api/settings/ai/test -H 'Content-Type: application/json' -d '{}'
curl -X DELETE localhost:8000/api/settings/ai
```

Caption styles: `classic`, `highlight`, `dynamic`, `chunks`, `karaoke`.

## Cancelling, re-running and cleaning up

A job can be stopped at any point with the **Cancel** button, or
`POST /api/jobs/{id}/cancel`. The worker checks the flag at every stage
boundary and on every progress callback, and any FFmpeg process the job has
running is terminated outright — so cancelling during a long encode takes
effect immediately rather than whenever that encode would have finished.
Clips that already rendered are kept, and the job ends as `CANCELLED` rather
than `FAILED`, because a decision is not a fault.

**Run again** queues a fresh job with the same URL and settings. **Delete**
removes a finished job, its clips, its thumbnails, its captions and its
transcript from disk.

Nothing survives the process exiting, so on startup the server reconciles what
it finds: jobs that never reached a worker are queued again, and jobs that were
mid-flight are failed with `ERROR_INTERRUPTED`. Without that they would sit at
`TRANSCRIBING` forever with the UI polling something that will never move.

## Long sources

The transcript of a three-hour video does not fit in one prompt. Rather than
truncating it — which silently made everything after roughly the first hour
unclippable — it is split into overlapping sections, each analysed by its own
concurrent model call, and the results merged with near-duplicates from the
seams removed. Sections overlap slightly so a moment lying across a boundary
is still visible whole to at least one call, and one section failing costs its
own clips rather than the job.

`AI_MAX_PROMPT_CHARS` (default 120,000) is the budget per call; lower it for a
model with a small context window.

## Running it on a network

The server binds `127.0.0.1` and the API is open, which is right for one user
on their own machine. Before exposing it anywhere else, set `API_TOKEN`:

```bash
API_TOKEN=$(openssl rand -hex 32)
```

Every `/api` call then needs it, as `Authorization: Bearer …`, `X-API-Key: …`,
or the cookie the UI sets after you enter it once. This matters more than it
looks: the settings panel stores a model API key, and `/api/quit` stops the
server.

Two paths stay open by necessity — `GET /api/auth`, so the page can ask whether
a token is needed, and the TikTok OAuth callback, which TikTok's redirect
cannot attach a header to and which carries its own one-time `state` instead.

## How ranking works

Every clip carries the arithmetic that produced its score, and the review UI
shows it behind "Why this clip?":

| Signal | Weight | Source |
|---|---:|---|
| AI semantic score | 50% | Model's `viral_score` |
| Hook strength | 15% | Model's `hook_score` |
| Emotion | 10% | Model's `emotion_score` |
| Speech density | 10% | Measured speech seconds ÷ clip length |
| Duration suitability | 5% | Distance from the 25–60s band |
| Transcript confidence | 5% | Mean word probability |
| Novelty | 5% | Vocabulary overlap with higher-ranked clips |

Weights are environment variables and are normalised at use, so they can be
retuned without touching code. Candidates overlapping more than 70% of the
shorter clip are dropped in favour of the higher-ranked one.

## Smart vertical reframing

The default is no longer a fixed centre crop. Every clip is scanned with a
small on-device face detector (YuNet, ~230KB, CPU-only, no network call), and
the 9:16 crop pans horizontally to follow the largest detected face instead of
sitting still. On a wide source where the speaker is not centred, a fixed
crop can miss them entirely; smart cropping keeps them in frame.

```
naive centre crop:  |███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| <- mostly black padding
smart crop:         |░░░░░░░░███████░░░░░░░░░░░░░░░░░░░░░░░░| <- follows the face
```

How it works, end to end:

1. Sample the source every `FACE_SAMPLE_INTERVAL_S` (default 0.5s) across the
   clip and run face detection on each downscaled frame.
2. With more than one face, the **largest** (closest to camera) is treated as
   the subject -- there is no active-speaker detection.
3. The horizontal track is smoothed (EMA + a dead zone + a max pan speed) so
   the crop drifts calmly instead of chasing every head movement. Vertical
   framing is a single fixed anchor per clip, not animated -- speakers bob and
   gesture far more than they change seats.
4. The result becomes a short list of `(time, x-offset)` keyframes, fed to
   FFmpeg's `crop` filter at render time via `sendcmd`, in the same pass as
   scaling, cropping and caption burn-in -- no extra encode.

This can only make output better or identical to a fixed crop: with no face
found, OpenCV missing, or any error mid-detection, it falls back to the exact
same centred crop v1 always produced, silently, without failing the render.
Set `REFRAME_MODE=crop` to skip detection and always use a fixed centre crop.

## Layout

```text
backend/app/
├── main.py              # app wiring, logging, token gate, error handling
├── config.py            # every tunable, one place
├── db.py                # engine, sessions, additive schema migration
├── models.py            # Video, Job, Clip, PublishingJob, StageLog
├── api/                 # jobs, clips, videos, system, settings, auth, streaming
├── services/
│   ├── downloader.py    # yt-dlp
│   ├── transcript.py    # transcript structures + normalisation
│   ├── transcription.py # faster-whisper, batched on CUDA
│   ├── ai_highlights.py # providers, prompt, sectioning, untrusted-output parsing
│   ├── selection.py     # validation, dedup, ranking
│   ├── metadata.py      # title/hook/description/hashtag completion
│   ├── runtime_config.py # AI settings entered in the UI
│   ├── captions.py      # ASS generation
│   ├── face_tracking.py # face-aware crop planning
│   ├── clipper.py       # FFmpeg render
│   └── media.py         # ffmpeg wrappers, encoder + session management
└── workers/             # job queue, restart recovery, pipeline
frontend/                # HTML/CSS/JS review UI
storage/                 # downloads, clips, transcripts, logs, SQLite
```

Adding a column to a model does not break an existing install: `init_db()`
adds any column the database is missing, which is the only kind of schema
change this project makes.

## Tests

```bash
./.venv/bin/python -m pytest                # the whole suite
./.venv/bin/python -m pytest -m "not slow"  # skip real FFmpeg encoding
```

The pipeline tests stub only the network-bound stages (download,
transcription); everything after runs for real, including FFmpeg producing
1080×1920 MP4s that are probed to confirm dimensions, duration and audio.
Publishing runs against a local server speaking TikTok's protocol, and the
model provider against one speaking the OpenAI protocol — real clients, real
chunking, real parsing, only the far end replaced.

## Where the time goes

Three things dominate a job, and each is parallel where it can be:

| Stage | Approach |
|---|---|
| Download | `yt-dlp` fetches 4 fragments at once (`YTDLP_CONCURRENT_FRAGMENTS`) |
| Transcription | Batched inference on CUDA, several VAD chunks per forward pass |
| Precision | Falls through float16 → int8_float16 → float32 rather than leaving the GPU |
| Face tracking | One FFmpeg decode of the clip's span, frames piped out pre-scaled |
| Rendering | Clips encode concurrently, bounded by what the encoder actually grants |

Measured on this machine (GTX 1080 Ti, 32 cores, 1080p source):

| Change | Before | After |
|---|---:|---:|
| Face sampling, 60s window | 3.03s | 0.81s |
| Rendering 5 × 30s clips | 16.55s | 10.62s |

Face tracking used to seek `VideoCapture` to each sample time, which decodes
from the preceding keyframe every time — on a long-GOP source that is tens of
thousands of decoded frames per clip. It now decodes the span once.

**On sharing a GPU.** Two failure modes here are silent rather than loud, and
both cost minutes per job:

- *Precision.* Pascal cards (a GTX 1080 Ti, say) have no efficient float16
  path, and CTranslate2 refuses the request rather than downgrading. Treating
  that one refusal as "no GPU here" moved the whole job to the CPU, when
  float32 runs on that card perfectly well. Whisper now tries float16,
  int8_float16 and float32 on the GPU before considering the CPU at all.
- *Memory.* Batched inference is the biggest transcription win and the first
  thing to exhaust VRAM on a card shared with anything else. A CUDA
  out-of-memory does not fail cleanly — it wedges the context, so every
  smaller retry fails too with an unrelated-looking error. The batch size is
  therefore chosen from the memory actually free, and batching is skipped
  below about 1.5 GB rather than attempted and lost.

**On the encoder session limit.** A hardware encoder opens a limited number of
simultaneous sessions, bounded both by the driver (consumer NVIDIA cards were
long capped at two or three) and by free VRAM, since each session allocates
buffers sized by the output resolution. Asking for one more fails that encode,
and the renderer would quietly finish it in software — slower than waiting, and
it makes quality depend on scheduling luck. So the limit is measured at
startup by opening sessions at the real output resolution until one is refused,
then enforced with a semaphore. Probing at a token frame size would report a
limit the actual renders cannot sustain: on this machine, with most of the card
held by another process, 256×256 suggested four sessions where 1080×1920
allowed two.

`RENDER_CONCURRENCY` overrides the desired count; the measured limit still caps it.

## Swapping the pieces

- **Model** — use the Settings panel, or set `OX_BASE_URL`/`OX_MODEL` for any
  OpenAI-compatible endpoint, or add a class with a `find_highlights` method in
  `ai_highlights.py`.
- **Queue** — `workers/queue.py` is the only module that knows about job
  dispatch; a Celery/Dramatiq/RQ swap is confined to it.
- **Database** — set `DATABASE_URL` to PostgreSQL; no SQLite-specific SQL is
  used elsewhere.
- **Reframing** — `REFRAME_MODE=blur_pad` keeps the full frame over a blurred
  fill instead of cropping it away. `REFRAME_MODE=crop` is the old
  always-centred behaviour, kept for anyone who prefers a fixed, predictable
  frame over a moving one.

## Posting to every TikTok account at once

Connect as many accounts as you like in **Settings → TikTok accounts**, then
press **Post all to TikTok** on a finished job. Every rendered clip is sent to
every connected account in parallel — 5 clips across 3 accounts is 15 posts,
all started at the same time.

### Setup

1. Create an app at [developers.tiktok.com](https://developers.tiktok.com), add
   the **Content Posting API** product, and enable **Direct Post**.
2. Register a redirect URI. **TikTok will not accept `localhost` or
   `127.0.0.1`**, under either scheme, so a purely local callback cannot
   work. Verified against the live authorize endpoint:

   | Redirect URI | Result |
   |---|---|
   | `http://localhost:8000/...` | rejected |
   | `https://localhost:8000/...` | rejected |
   | `http://127.0.0.1:8000/...` | rejected |
   | `https://127.0.0.1:8000/...` | rejected |
   | `https://your-domain.example/...` | accepted |

   The failure surfaces as an error naming `client_key`, which sends you
   looking at your credentials instead of the URI.

   So the callback needs a public HTTPS hostname that forwards to this
   machine, from a tunnel (`cloudflared`, `ngrok`) or a reverse proxy. Use a
   **stable** hostname: TikTok compares the URI character for character, and
   a throwaway tunnel URL has to be re-registered every restart. Register the
   result, ending in `/api/accounts/tiktok/callback`.
3. Put the credentials in `~/.config/clipz/config.env`, mode `600`:

```bash
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...
TIKTOK_REDIRECT_URI=http://localhost:8000/api/accounts/tiktok/callback
```

   That file is read by both the installed app and a source checkout, and it
   wins over the repository's `.env`. Keep the client secret out of `.env` and
   out of the AppImage: the bundle is portable, and anyone holding a copy of
   your secret can post as you.

4. Restart, open Settings, and connect each account. OAuth runs per account, so
   you sign in to each one once.

**Exposing CLIPZ through a tunnel makes the API reachable from the internet.**
Set `API_TOKEN` before you do, or anyone who learns the URL can drive the
pipeline, read your clips and post to your connected accounts.

**A local redirect URI pins the port.** TikTok compares the redirect URI
against the registered one character for character, and sends the browser back
to whatever is listening on that port. If CLIPZ quietly moved to a spare port,
connecting an account would hand a live authorization code to some unrelated
program. So a redirect URI pointing at this machine pins the listening port,
and a clash is reported rather than worked around (a tunnelled URI pins
nothing, since its port belongs to the tunnel):

```text
Port 8000 is already in use.
... requires CLIPZ to listen on port 8000.
Stop the other program, or start CLIPZ with --port to run without publishing.
```

`--port` still overrides it, which starts CLIPZ normally with publishing
unavailable.

### What the limits mean for you

| Limit | Effect |
|---|---|
| Unaudited app | Everything posts **private (SELF_ONLY)** and only **5 accounts per 24h** can post. Passing TikTok's audit lifts both. |
| ~15 posts/day per account | Shared across every app touching that account, not just this one. |
| 6 post requests/min per account | Enforced client-side, so a burst queues instead of getting 429s. |

Accounts post concurrently (`PUBLISH_CONCURRENCY`, default 4), each with its
own rate-limit window. A failure on one account never stops the others, and
each `(clip, account)` pair is unique in the database — re-pressing the button
retries only what failed and cannot double-post.

### Automatic posting

**Settings → Publishing → "Post finished jobs to TikTok automatically"**
removes the button press: the moment a job completes, every rendered clip is
fanned out to every connected account. `AUTO_PUBLISH=true` in the environment
sets the same toggle; the panel wins. `AUTO_PUBLISH_MAX_CLIPS` caps how many
clips of one job post (0 = all), because a batch paste times every account is
a burst against per-account daily limits. The toggle only affects jobs that
finish while it is on — nothing is back-filled — and a posting failure never
changes a completed job's state; the posts carry their own status, with a
**Retry failed** button in the publishing summary.

### Draft mode (posting before your app is audited)

Until TikTok audits your developer app, direct posts are forced to
`SELF_ONLY`. **Settings → Publishing → "Upload to TikTok drafts instead of
posting directly"** (or `PUBLISH_AS_DRAFT=true`) switches the flow to draft
uploads: clips land in each account's TikTok drafts with no audit needed, and
the account's owner finishes each post from the TikTok app — two taps, any
privacy. Draft rows show as `saved to drafts` in the publishing summary.
Multiple accounts: each **Connect** pass links whichever TikTok user is
logged in at tiktok.com (the topbar strip's **+** opens the same flow), so
log out between accounts or connect from an incognito window.

Tokens are encrypted at rest with a key in `storage/token.key` (0600) and are
never returned by the API. Access tokens are refreshed automatically before
they expire.

### Worth knowing

TikTok's guidelines expect clients to "facilitate authentic creators to post
original content". Posting the same clip to accounts you genuinely operate is
what this is for; blasting identical content across many throwaway accounts is
what their spam and unoriginal-content enforcement targets, and no amount of
API plumbing protects against that.

## Troubleshooting downloads

**`ERROR_DOWNLOAD_FAILED` — "Requested format is not available"**

Usually transient, and usually caused by a missing JavaScript runtime. yt-dlp
needs one to decipher YouTube's stream URLs; without it, it falls back to a
limited player client that sometimes exposes no usable video/audio pairing at
all — so the same URL fails on one attempt and succeeds on the next.

The app already retries three times (bypassing the player cache on the second
attempt and dropping every format constraint on the third), but the real fix is
to install a runtime. `/api/health` reports `js_runtime`, and the UI shows a
warning chip when there is none:

```bash
curl -fsSL https://deno.land/install.sh | sh   # deno — what yt-dlp prefers
# or Node 20+ (Node 18 and older are rejected by yt-dlp), or bun, or quickjs
```

Anything found on `PATH` is enabled automatically on the next job; nothing to
configure. Keeping yt-dlp current matters too — `pip install -U yt-dlp` is the
first thing to try when downloads start failing across the board.

Failures that no retry can fix (private, members-only, age-restricted, removed,
region-blocked, or a livestream that has not started) are reported as such
immediately instead of being retried.

## Known limits

- YouTube only. `SUPPORTED_HOSTS` in `downloader.py` is the gate.
- Smart cropping tracks the single largest detected face -- there is no
  active-speaker detection, so a wide two-shot where neither person
  dominates the frame will follow whichever face is biggest, not whoever is
  currently talking. It still degrades safely to a centred crop if no face
  is found at all.
- Whisper `base` is the default for speed. `WHISPER_MODEL=small` or `medium`
  transcribes names and jargon noticeably better, at a time cost.
- One job runs at a time by default (`WORKER_COUNT`), since transcription and
  encoding already saturate the GPU. Clips *within* a job render in parallel.
- Age-restricted and members-only videos need a signed-in session: set
  `YTDLP_COOKIES_FILE` to a Netscape-format cookies file, or
  `YTDLP_COOKIES_FROM_BROWSER=firefox` to read them from a local browser.
- A cancelled or interrupted job cannot resume from where it stopped; its
  intermediate files are cleaned up, so re-running starts from the download.
