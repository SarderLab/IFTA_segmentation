## Context

The IFTA prediction pipeline (`IterativePredict_1X.py`) locates a TF1 checkpoint by globbing for `*.ckpt*` files and extracting the highest step number. When no checkpoint files are present `get_test_step()` returns `0`, causing the DeepLab `eval.py` subprocess to try loading `model.ckpt-0`, which fails with a `ValueError`. Because `call()` does not propagate the subprocess exit code, the outer loop reaches `un_suey()` and crashes on missing prediction PNGs.

`get_choppable_regions.py` passes `selem=` to `skimage.morphology.binary_dilation`; that argument was renamed `footprint=` in scikit-image 0.19 and will be removed in 1.0.

## Goals / Non-Goals

**Goals:**
- Surface a clear, actionable error when no checkpoint files exist in the model directory.
- Prevent `un_suey()` from running when the DeepLab subprocess fails.
- Eliminate the `FutureWarning` for `selem` deprecation.

**Non-Goals:**
- Migrating the TF1 DeepLab code to TF2 or Keras.
- Changing the checkpoint naming convention or model directory layout.
- Modifying the SLURM/Singularity runner configuration.

## Decisions

**`get_test_step()` — raise instead of returning 0**
Returning `0` is undetectable by the caller and leads to a confusing downstream `ValueError`. Raising `FileNotFoundError` with the model directory path immediately identifies the root cause.

**`predict_xml()` — use `subprocess.run()` with `check=True`**
`subprocess.call()` discards the exit code. Switching to `subprocess.run(..., check=True)` raises `subprocess.CalledProcessError` on non-zero exit, which is then caught and re-raised with a human-readable message. This prevents `un_suey()` from running when prediction failed.

**`get_choppable_regions.py` — rename kwarg `selem` → `footprint`**
One-line change; no behavioral difference.

## Risks / Trade-offs

- Raising in `get_test_step()` changes existing behavior for any caller that relied on the silent `0` return (e.g., new-project bootstrapping). The only caller found is `predict_xml()`, which will now surface the error rather than silently failing — that is the intended outcome.
- `subprocess.run(..., check=True)` was introduced in Python 3.5 and is universally available in the target Python 3.7+ environment.
