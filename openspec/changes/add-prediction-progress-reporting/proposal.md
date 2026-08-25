## Why

The DeepLab prediction loop only prints "step 0" for an 88-batch job, giving no visibility into progress in Girder job logs or the DSA UI. Users cannot tell if the job is running, stuck, or nearly done. Additionally, `--base_dir` fails as mandatory even when `--input_path` is provided (its parent directory is the natural base).

## What Changes

- Replace the `step % 100 == 0` print in `model.py` with a clean per-step line (`Step X/N — Y/Z images (P%)`) flushed immediately to stdout so Girder log tailing shows real-time progress.
- Add optional `--girder_api_url`, `--girder_token`, and `--girder_job_id` args to `main.py`; when all three are present, patch the Girder job progress endpoint at each step to drive the DSA progress bar.
- Forward those three args from `IterativePredict_1X.py` into the DeepLab subprocess command, reading the job ID from the `GIRDER_JOB_ID` environment variable (set by the Girder worker).
- In `IFTASegmentation.py`, read `GIRDER_JOB_ID` from the environment and pass `--girder_job_id` down to `segmentation_school.py` / DeepLab.
- Fix `segmentation_school.py` auto-derive logic: when `--base_dir` is omitted, fall back to `os.path.dirname(os.path.abspath(args.input_path))` instead of requiring `--input_files` (which is never passed by the Girder CLI).

## Capabilities

### New Capabilities
- `prediction-progress`: Real-time per-step progress output during DeepLab prediction, with optional Girder job progress bar integration via REST API.
- `auto-base-dir`: Automatic derivation of `base_dir` from `input_path` when `--base_dir` is not explicitly provided.

### Modified Capabilities

## Impact

- `Codes/Deeplab_network/model.py` — `predict()` loop
- `Codes/Deeplab_network/main.py` — argument parser
- `Codes/IterativePredict_1X.py` — `predict_xml()` DeepLab command construction
- `ifta/cli/IFTASegmentation/IFTASegmentation.py` — job ID forwarding
- `ifta/ifta_code/segmentation_school.py` — `base_dir` auto-derive logic
