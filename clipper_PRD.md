# CLIPZ — Product Requirements Document

| Field | Value |
|---|---|
| Status | Draft v1.0 |
| Owner | Sam |
| Product type | AI-powered short-form video clipping and publishing platform |
| Initial delivery | Web application with a local/self-hosted backend |
| Primary outputs | TikTok, YouTube Shorts, and Instagram Reels-compatible video |

## 1. Executive summary

CLIPZ turns long-form video into publish-ready short-form social content. A user submits a source video URL, selects a few settings, and receives ranked vertical clips with synchronized captions and editable publishing metadata.

The product is deliberately modular:

> **AI makes semantic decisions; deterministic software performs media processing.**

AI models interpret the transcript, identify compelling moments, generate metadata, and score candidates. Conventional services download media, transcribe audio, validate timestamps, crop and render video, manage jobs, and publish content. This separation improves reliability, reproducibility, cost control, and model portability.

## 2. Problem

Producing short-form content from a podcast, interview, livestream, or long YouTube video is a repetitive, multi-hour process. Creators must identify moments, set edit points, reframe video, add captions, write metadata, export clips, and upload or schedule every post.

Current clipping tools often provide weak highlight selection, generic metadata, limited control over caption styling or framing, opaque scoring, expensive subscriptions, and little support for self-hosted processing or interchangeable AI providers.

CLIPZ will provide an inspectable pipeline whose stages can be independently improved or replaced.

## 3. Product vision

The long-term product is an autonomous content-production pipeline:

```text
Long-form video → semantic analysis → highlight selection → clip rendering
→ vertical reframing → captions → metadata → ranking → scheduling → publishing
```

The first release focuses on a dependable creator workflow: produce high-quality clips for human review and download. Automated publishing and performance-learning loops follow once that core pipeline is stable.

## 4. Goals and non-goals

### Goals

1. **Identify strong highlights automatically.** Detect hooks, insights, emotional moments, stories, humor, surprising facts, and strong opinions from a timestamped transcript.
2. **Produce mobile-ready vertical video.** Render 9:16 clips suitable for major short-form platforms.
3. **Create accurate, expressive captions.** Use word-level timing to enable modern caption styles.
4. **Generate editable metadata.** Produce a title, hook, description, and hashtags for each clip.
5. **Rank transparently.** Combine AI and measurable signals into an explainable final score.
6. **Support progressive automation.** Enable manual download first, then assisted posting, scheduling, and optional autonomous publishing.
7. **Keep AI portable.** Allow the highlight model to change without altering the broader media pipeline.

### Non-goals for the initial product

- A professional non-linear video editor or advanced visual-effects suite.
- A guarantee of viral performance.
- Generation of entirely new video footage.
- Replacement of professional editors.
- Perfect understanding of every content format.
- Multi-tenant billing, team permissions, enterprise authentication, or creator-community features.
- Immediate support for every source or social platform.

## 5. Users

### Primary user

An individual creator operating a personal pipeline for podcasts, interviews, education, commentary, gaming, livestreams, or YouTube videos.

### Future users

Social agencies, podcast studios, marketing teams, influencers, businesses, and content-automation agencies.

### Initial product assumption

Version 1 is optimized for one user. This intentionally avoids early complexity from multi-tenancy, billing, role management, and enterprise identity systems.

## 6. Core user journey

1. The user pastes a YouTube URL.
2. They select clip count, maximum duration, and caption style.
3. They select **Generate clips**.
4. CLIPZ downloads and analyzes the source, then identifies, validates, ranks, and renders clip candidates.
5. The user reviews previews, edits metadata if needed, and downloads the finished MP4s.
6. In later releases, the user can publish immediately or schedule posts.

## 7. MVP scope

The MVP is complete when a user can:

1. Paste a YouTube URL and request a number of clips.
2. Download the source and create a timestamped transcript.
3. Send the transcript to a configured AI model and receive structured highlight recommendations.
4. Validate, deduplicate, and rank recommendations.
5. Cut and render 9:16 MP4 clips with FFmpeg.
6. Generate and burn in synchronized captions.
7. Preview clips, edit their metadata, and download them.

Publishing, scheduling, face-aware reframing, and analytics are explicitly post-MVP.

## 8. Functional requirements

### 8.1 Video intake

The MVP accepts YouTube URLs. The interface includes:

- Video URL
- Desired clip count (default: 5)
- Maximum clip duration (default: 60 seconds)
- Caption style (default: Dynamic)
- **Generate clips** action

Future sources may include direct video URLs, TikTok, Instagram, Twitch, file uploads, and cloud storage.

For each source, retain title, duration, channel, source URL, thumbnail URL, and upload date.

### 8.2 Asynchronous jobs

Video processing may take several minutes, so every request runs as an asynchronous job. Jobs must report a state, numeric progress, and human-readable message.

| State | Meaning |
|---|---|
| `QUEUED` | Accepted and awaiting a worker |
| `DOWNLOADING` | Retrieving the source media |
| `EXTRACTING_AUDIO` | Preparing audio for transcription |
| `TRANSCRIBING` | Producing timestamped speech data |
| `ANALYZING` | AI is evaluating the transcript |
| `SELECTING_CLIPS` | Validating, deduplicating, and ranking candidates |
| `PROCESSING_VIDEO` | Cutting and reframing clips |
| `GENERATING_CAPTIONS` | Building caption assets |
| `RENDERING` | Encoding final output |
| `COMPLETED` | Outputs are ready |
| `FAILED` | The job stopped with a recoverable error record |

Example status response:

```json
{
  "job_id": "job_abc123",
  "status": "TRANSCRIBING",
  "progress": 42,
  "message": "Generating timestamped transcript"
}
```

### 8.3 Downloading and media preparation

Use `yt-dlp` to retrieve the best available source quality, extract source metadata, and download audio when needed. The service must report progress, handle failed downloads, retain diagnostic information, and clean temporary files after terminal job completion.

### 8.4 Transcription and preprocessing

Use `faster-whisper` as the initial transcription engine. It must produce:

- Full transcript text
- Segment and word-level timestamps
- Detected language
- Confidence data where available
- Speaker changes and topic boundaries where feasible

Before AI analysis, normalize the transcript into timestamp-preserving sentences and paragraphs. Remove nonessential silence markers while retaining source timing.

Word-level timing is required for animated captions.

### 8.5 AI highlight detection

The AI model operates only on semantic decisions. It should recognize hooks, emotional or funny moments, controversy, surprising facts, educational value, stories, opinions, and unexpected conclusions. It must not control video-editing operations.

The highlight-model interface is provider-agnostic. Ox Alpha is the initial model; future implementations may use OpenAI, Claude, Gemini, Qwen, local models, or any OpenAI-compatible API.

The request includes source title, duration, target clip count, maximum duration, and normalized transcript. The response must conform to a structured JSON schema such as:

```json
{
  "clips": [{
    "start": 124.2,
    "end": 174.5,
    "viral_score": 92,
    "hook_score": 95,
    "emotion_score": 82,
    "reason": "Strong controversial opening followed by a surprising conclusion.",
    "title": "Nobody Talks About This",
    "hashtags": ["#podcast", "#mindset", "#success"]
  }]
}
```

### 8.6 Candidate validation, deduplication, and ranking

Never trust AI output without validation. For each candidate, the backend must verify that timestamps exist, `start >= 0`, `end > start`, and duration falls within configured limits. Defaults are a 15-second minimum, 90-second maximum, and 25–60 second recommended range.

Candidates must be deduplicated. Calculate overlap as `overlapping duration / shorter clip duration`; if it exceeds the configurable threshold (default: 70%), retain the higher-ranked clip.

Final ranking is explainable and combines AI with deterministic signals:

| Signal | Weight |
|---|---:|
| AI semantic score | 50% |
| Hook strength | 15% |
| Emotion | 10% |
| Speech density | 10% |
| Duration suitability | 5% |
| Transcript confidence | 5% |
| Novelty | 5% |

Each clip stores its component scores, final score, and selection rationale.

### 8.7 Video clipping and vertical reframing

Use FFmpeg to cut, encode, and synchronize video/audio. Output quality and encoder must be configurable, with hardware acceleration used where available. Encoding failures should be retried and logged.

Version 1 uses a center crop to 9:16. A later release adds face-aware cropping with detection, tracking, primary-subject selection, and smooth crop movement. Smoothing may use an exponential moving average, Kalman filter, interpolation, or motion prediction. Optional scene detection via PySceneDetect may snap edit points to nearby scene boundaries.

### 8.8 Captions

Generate ASS subtitle files from word-level timestamps and render them with FFmpeg. Initial styles:

- Classic
- Highlight word
- Dynamic pop
- TikTok-style chunks of three to five words
- Karaoke/current-word highlight

### 8.9 Metadata and clip review

For every clip, generate an editable title, description, hook, and hashtags. The clip-review interface shows video preview, score, metadata, and actions to download, regenerate, post, or schedule. Post and schedule actions may be disabled until publishing is released.

### 8.10 Publishing (post-MVP)

Support these modes in sequence:

1. **Manual:** download the file.
2. **Assisted:** user selects **Post now**.
3. **Scheduled:** user selects date, time, and platform.
4. **Automatic:** the system schedules the best clips according to configured rules.

Planned integrations: TikTok Content Posting API, YouTube Data API, and Meta Graph API where supported. Publishing must use OAuth, prevent duplicate posts on retry, and retain platform status, external post ID, and published URL.

## 9. System design

```text
Frontend (initially HTML/JavaScript; later React)
  → FastAPI API
  → Job queue / worker
  → yt-dlp + faster-whisper + highlight-model adapter
  → clip engine (FFmpeg/OpenCV)
  → caption engine (ASS + FFmpeg)
  → storage and output delivery
```

Recommended backend organization:

```text
backend/app/
├── main.py
├── api/          # jobs, clips, videos, publishing
├── services/     # downloader, transcription, AI, validation, ranking, rendering
├── workers/      # pipeline worker
├── models/       # video, job, clip, publishing job
└── storage/
```

## 10. API and data model

### Core endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Create a clip-generation job |
| `GET` | `/api/jobs/{job_id}` | Retrieve job status and progress |
| `GET` | `/api/jobs/{job_id}/clips` | List generated clips |
| `GET` | `/api/clips/{clip_id}/download` | Download a rendered clip |
| `POST` | `/api/clips/{clip_id}/publish` | Publish or enqueue a clip (post-MVP) |

Example job request:

```json
{
  "url": "https://youtube.com/watch?v=xxxxx",
  "clip_count": 5,
  "max_duration": 60,
  "caption_style": "dynamic"
}
```

### Primary records

| Entity | Core fields |
|---|---|
| Video | id, source_url, title, duration, local_path, created_at |
| Job | id, video_id, status, progress, error, created_at, completed_at |
| Clip | id, job_id, start_time, end_time, duration, AI score, final score, title, hashtags, output_path, status |
| Publishing job | id, clip_id, platform, scheduled_time, status, external_post_id, published_url |

## 11. Technology choices

| Area | Initial choice | Evolution |
|---|---|---|
| Backend | Python, FastAPI, Pydantic | — |
| Downloading | yt-dlp | Additional source adapters |
| Video | FFmpeg, OpenCV | GPU encoders and smart crop |
| Transcription | faster-whisper | Alternate providers as needed |
| AI | OpenAI-compatible client; Ox Alpha | Swappable model adapters |
| Queue | FastAPI `BackgroundTasks` | Redis with Celery, Dramatiq, or RQ |
| Database | SQLite | PostgreSQL |
| Frontend | HTML/JavaScript | React with Vite |

## 12. Non-functional requirements

### Performance

For a typical one-hour source video on suitable hardware, target an end-to-end processing time of 10–20 minutes. AI analysis should complete in under two minutes; clip generation and caption rendering should each target under five minutes. Actual download and transcription time depends on network speed and hardware.

GPU acceleration should be used when available for transcription, face detection, and video encoding. The clipping workflow must not require a large local LLM.

### Reliability and observability

Target pipeline success rate: >95%. Target transcription success rate: >98%. Target clip-rendering success rate: >98%.

Log timestamp, job ID, stage, status, duration, and error for every stage. Define actionable errors including `ERROR_DOWNLOAD_FAILED`, `ERROR_UNSUPPORTED_SOURCE`, and `ERROR_TRANSCRIPTION_FAILED`. AI calls should use configurable retries; invalid structured output should trigger a repair attempt. Preserve FFmpeg logs. Publishing retries must be idempotent.

### Security

The initial application is local. Store API keys in environment variables, never expose them to the frontend, and encrypt OAuth tokens where feasible. Example variables include `OX_API_KEY`, `TIKTOK_CLIENT_KEY`, and `TIKTOK_CLIENT_SECRET`.

## 13. Delivery roadmap

| Phase | Outcome |
|---|---|
| 1. End-to-end MVP | YouTube URL → download → transcript → AI-selected timestamp → one FFmpeg clip → MP4 |
| 2. Multiple clips | Candidate selection, validation, overlap detection, ranking, and metadata |
| 3. Captions | Word timing, ASS generation, and rendered caption styles |
| 4. Smart reframing | Face detection/tracking, crop smoothing, and multi-person support |
| 5. Web UI | URL intake, progress, previews, downloads, and metadata editing |
| 6. Publishing | OAuth, platform integration, queue, scheduling, and posting status |
| 7. Advanced automation | Automatic scheduling, analytics, feedback loops, and improved ranking |

### Recommended implementation order

1. `yt-dlp` and FFmpeg
2. `faster-whisper`
3. AI highlight detection
4. Candidate validation
5. Multiple-clip generation
6. Captions
7. Basic web UI
8. Smart reframing
9. Publishing
10. Scheduling and analytics

Do not build the complete product concurrently. The first reliable milestone is a source URL producing one correctly selected, rendered MP4. Subsequent capabilities should build on that proven path.

## 14. Success measures

### Product and technical measures

- Pipeline, transcription, and rendering success rates.
- Processing time per hour of source video.
- Number of usable clips generated per source video.
- User acceptance rate and manual edits required per clip.

### Model-quality measures

- Agreement between AI-suggested clips and clips selected by the user.
- Calibration of predicted clip score against actual published performance.
- Over time, views, watch time, completion rate, and engagement can feed back into highlight selection and ranking.

## 15. Future state

The mature product may monitor a creator's channel for new uploads, generate and rank clips automatically, caption and package them, schedule and publish selected outputs, then learn from performance analytics. At that point, CLIPZ becomes an autonomous short-form content-production system—not merely an AI video clipper.
