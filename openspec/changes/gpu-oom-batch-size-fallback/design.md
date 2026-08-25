## Context

The DeepLab prediction loop in `model.py:_predict_impl()` runs inference batch-by-batch using a pre-built TF dataset iterator (`self.predict_dataset_iter`). When `batch_size=8` and input tiles are 750×750 pixels, a BatchNormalization layer fails with a GPU memory allocation error on the very first batch, causing the process to exit with code 1. Because the dataset iterator and model are rebuilt each call, recovery must happen at the `_predict_impl` level.

Current flow:
1. `prediction_setup()` → builds dataset with fixed batch size
2. Model loaded from checkpoint
3. Loop over batches → OOM crash → non-zero exit

## Goals / Non-Goals

**Goals:**
- Catch `tf.errors.ResourceExhaustedError` (GPU OOM) during inference and automatically retry with `batch_size // 2`
- Halve batch size up to a minimum of 1 before giving up
- Log a clear warning each time batch size is reduced
- Require no changes to CLI interface or caller code

**Non-Goals:**
- Dynamically adjusting tile size or overlap
- Recovering mid-WSI (retry always starts from the first image of that WSI call)
- Handling CPU OOM or non-OOM TF errors

## Decisions

**Decision: Wrap the entire `_predict_impl` in a retry loop, not individual batch steps**

Rationale: The TF dataset iterator (`self.predict_dataset_iter`) is created inside `prediction_setup()` with the batch size baked in. After an OOM, the iterator state is corrupt and the model graph may have partial state. The cleanest recovery is to call `prediction_setup()` again with the reduced batch size and restart inference from image 0.

Alternatives considered:
- Catching OOM per-batch and skipping: leaves partial/missing outputs for the failed batch
- Reducing per-image rather than per-batch: not possible without rebuilding the dataset

**Decision: Use `except (tf.errors.ResourceExhaustedError, Exception)` with OOM string check**

Rationale: TF2 OOM can surface as `tf.errors.ResourceExhaustedError` or as a generic Python exception wrapping it. Checking both the exception type and `"failed to allocate memory"` in the message string covers all observed cases from the logs.

**Decision: Minimum batch size = 1**

Rationale: If even batch_size=1 causes OOM, there is no further fallback; re-raise so the caller gets a meaningful error rather than an infinite loop.

## Risks / Trade-offs

- [Risk] Retry doubles total startup time (checkpoint reload) for each halving step → Mitigation: at most log₂(batch_size) retries (e.g., 3 for batch_size=8); each reload is ~19s based on logs, so worst case adds ~57s
- [Risk] Partial output files from the failed attempt may exist on disk → Mitigation: `_predict_impl` checks for and overwrites existing files; no cleanup needed
- [Risk] OOM that occurs on step >0 requires re-running all already-completed steps → Mitigation: acceptable trade-off given that mid-stream recovery would require saving state; document in warning log

## Migration Plan

Single-file change to `model.py`. No migration needed; batch_size parameter semantics shift from "exact" to "maximum starting value" which is a safe behavioral relaxation.
