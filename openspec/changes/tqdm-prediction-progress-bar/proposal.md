## Why

The current per-step progress output (`Step X/N — Y/Z images (P%)`) is plain text with no timing or throughput information. Users want a rich tqdm-style bar showing percentage, a green block bar, current/total image count, elapsed time, ETA, and processing speed — matching the format:
`Total WSI progress:  19%|█▉        | 87/462 [00:23<05:16,  1.18image/s]`

## What Changes

- Replace the plain-text `Step X/N — Y/Z images (P%)` print in `model.py` with a `tqdm` progress bar that:
  - Is labeled `Total WSI progress`
  - Tracks individual images (not batches) — updated `batch_size` times per step
  - Uses green color (`colour="green"`)
  - Shows elapsed time, ETA, and throughput in `image/s`
  - Writes each update as a new line (suitable for Girder log capture — `tqdm` with `file=sys.stdout` and `dynamic_ncols=False`)
- Add `tqdm` as a dependency if not already installed in the Docker image.

## Capabilities

### New Capabilities
- `tqdm-wsi-progress-bar`: Rich tqdm progress bar for DeepLab prediction showing image-level throughput, ETA, and a green block bar.

### Modified Capabilities

## Impact

- `Codes/Deeplab_network/model.py` — `predict()` loop progress output
- `Dockerfile` — ensure `tqdm` is installed
