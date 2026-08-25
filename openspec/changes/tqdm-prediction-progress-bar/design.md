## Context

The predict loop in `model.py` currently prints one plain line per batch step. We want to replace this with a `tqdm` progress bar that tracks at the image level (finer granularity), shows a green colored block bar, and includes elapsed/ETA/throughput — matching the exact format the user demonstrated.

The Girder job log captures stdout line-by-line. `tqdm` by default writes to stderr and uses `\r` to overwrite the current line in a terminal. In a log-capture context like Girder, `\r` lines are generally discarded or appear as a single final value. To make each update visible as a log line, `tqdm` must be configured to write new lines instead of overwriting.

## Goals / Non-Goals

**Goals:**
- Match the exact `tqdm` output format: `Total WSI progress:  19%|█▉        | 87/462 [00:23<05:16,  1.18image/s]`
- Track at image granularity (increment once per image saved, not once per batch)
- Green bar color via `colour="green"`
- Each update appears as a new line in Girder logs

**Non-Goals:**
- In-place terminal overwrite (not useful in log-capture environments)
- Progress bar in any other part of the pipeline (chopping, reconstruction)

## Decisions

**D1 — Use `tqdm` with `file=sys.stdout` and new-line output**

Set `tqdm(file=sys.stdout, dynamic_ncols=False, ncols=80)` so output goes to stdout (captured by Girder). To emit a new line per update rather than `\r`-overwrite, set `bar_format` to not include `\r` or use `tqdm.write()` pattern. The simplest approach: use standard `tqdm` and call `pbar.update(batch_size)` per step — tqdm naturally emits a new line each time `miniters` is exceeded when writing to a non-TTY stream.

Actually, when stdout is not a TTY (as in Girder's captured subprocess), `tqdm` automatically disables the `\r` overwrite and emits new lines for each update. This means the default `tqdm` behavior is already correct for Girder logs — no special configuration needed beyond `file=sys.stdout`.

**D2 — Track images, not steps**

Create `tqdm(total=total_images, ...)` and call `pbar.update(batch_size)` after each batch (capped at remaining images for the final batch). This gives image-level throughput (`image/s`) matching the desired format.

**D3 — `tqdm` availability**

`tqdm` is a near-universal Python package already present in most ML environments. Verify it is installed in the Dockerfile; add `pip install tqdm` if absent.

## Risks / Trade-offs

- [tqdm not installed] If tqdm is missing the predict loop will crash → Mitigation: add to Dockerfile and wrap import in try/except with plain-text fallback.
- [Non-TTY new-line behavior] tqdm's non-TTY mode emits a line only when progress changes enough (`miniters`). With 176 images and batch_size=2 every update is significant, so all 88 lines will appear. For very large WSIs this is still fine.
