## Context

The pipeline runs 3 levels deep: `IFTASegmentation.py` (Girder CLI) → `segmentation_school.py` (subprocess) → `IterativePredict_1X.py` (in-process) → `Deeplab_network/main.py` (subprocess). Progress is printed only inside the innermost `model.py` predict loop, and only at `step % 100 == 0` — so for jobs with fewer than 100 batches it prints exactly once ("step 0").

Girder workers expose the job ID via the `GIRDER_JOB_ID` environment variable. The Girder REST API accepts `PATCH /job/{id}` with a `progress` dict (`{current, total, message}`) to update the DSA progress bar. The top-level CLI already has a `GirderClient` (`gc`) and the API URL / token.

## Goals / Non-Goals

**Goals:**
- Print a clean, flushed line per batch step in `model.py` so Girder log tailing reflects real-time progress.
- Optionally update the Girder job progress bar (DSA UI) via REST if credentials and job ID are available.
- Fix `base_dir` auto-derive to work from `input_path` (which is always provided by the Girder CLI).

**Non-Goals:**
- Progress reporting during training (only prediction is addressed).
- Changing the subprocess chain structure.
- Making Girder progress reporting mandatory — it degrades gracefully when args are absent.

## Decisions

**D1 — Pass Girder credentials as CLI args, not env vars, into DeepLab**

`GIRDER_JOB_ID` is read at the top level (`IFTASegmentation.py`) and passed explicitly as `--girder_job_id` through the command chain. `girderApiUrl` and `girderToken` are already CLI args in `segmentation_school.py` and can be forwarded the same way.

Alternative considered: read env vars inside `model.py` directly. Rejected because it creates an implicit dependency on the environment instead of explicit configuration, making the behavior harder to test or override.

**D2 — Girder progress update is best-effort (silent fail)**

If the `girder_client` import fails or the API call raises, the exception is caught and ignored. Prediction must never abort due to a failed progress update.

**D3 — `base_dir` fallback uses `input_path`, not `input_files`**

`input_path` is the full local path to the downloaded WSI (e.g. `/mnt/girder_worker/.../image.svs`). Its `os.path.dirname` is the natural working directory. `input_files` was an additional arg introduced for a different use case and is never passed by the Girder CLI.

## Risks / Trade-offs

- [Girder API rate limit] Patching on every batch step (up to ~hundreds of calls per job) could be throttled → Mitigation: the call is wrapped in try/except; if throttled, progress just stops updating without breaking prediction.
- [Subprocess stdout buffering] Even with `flush=True`, the parent process (segmentation_school.py called via `shell=True`) may buffer output before Girder sees it → Mitigation: `PYTHONUNBUFFERED=1` can be set in the subprocess env, or the shell=True call can be replaced with list-form (separate improvement).
