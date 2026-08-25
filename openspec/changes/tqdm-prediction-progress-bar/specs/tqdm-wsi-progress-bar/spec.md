## ADDED Requirements

### Requirement: tqdm progress bar replaces plain-text step output
The `predict()` method in `model.py` SHALL use a `tqdm` progress bar instead of the plain `Step X/N — Y/Z images (P%)` print. The bar SHALL be configured as:
- `desc="Total WSI progress"`
- `total=total_images` (image count, not batch count)
- `unit="image"`
- `colour="green"`
- `file=sys.stdout`
- `dynamic_ncols=False`, `ncols=80`

#### Scenario: Progress bar updates per batch
- **WHEN** a batch of images is processed
- **THEN** `pbar.update(n)` is called where `n` is the number of images in that batch (capped so total does not exceed `total_images`)

#### Scenario: Progress bar closes after prediction
- **WHEN** the prediction loop finishes
- **THEN** `pbar.close()` is called before printing the "output files saved" message

#### Scenario: tqdm not installed — fallback to plain text
- **WHEN** `import tqdm` raises `ImportError`
- **THEN** the predict loop falls back to the existing `Step X/N — Y/Z images (P%)` plain-text output with `sys.stdout.flush()`

### Requirement: tqdm installed in Docker image
The `Dockerfile` SHALL include `tqdm` in its pip install step so the package is available at runtime.

#### Scenario: tqdm importable in container
- **WHEN** the Docker image is built
- **THEN** `python -c "import tqdm"` exits with code 0
