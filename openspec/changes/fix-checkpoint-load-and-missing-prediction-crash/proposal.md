## Why

The IFTA segmentation pipeline crashes with two cascading errors during prediction:

1. **Silent checkpoint miss**: `get_test_step()` returns `0` when no `.ckpt*` files are found in the model directory, instead of raising an error. The DeepLab subprocess then attempts to load `model.ckpt-0` which does not exist, causing a `ValueError`.

2. **Cascading FileNotFoundError**: Because the checkpoint load failure happens inside a `call()` subprocess with no return-code check, the pipeline continues to `un_suey()`, which tries to read prediction mask PNGs that were never generated, crashing with `FileNotFoundError`.

A third issue is a `FutureWarning` about the deprecated `selem` argument in `binary_dilation` (should be `footprint`) in `get_choppable_regions.py`.

A fourth issue is a UX friction point: when `--input_files` (the WSI path) is provided to the CLI, the caller also has to explicitly provide `--basedir` (the output directory), even though it can be derived directly from the image file's parent directory. This forces users/plugins to set the same path information twice.

## What Changes

- **`get_test_step()`** in `IterativePredict_1X.py`: raise a clear `FileNotFoundError` (with the model directory path) when no checkpoint files are found, instead of silently returning `0`.
- **`predict_xml()`** in `IterativePredict_1X.py`: replace `call()` with `subprocess.run()` and check the return code; raise an exception before `un_suey()` if the DeepLab subprocess failed.
- **`get_choppable_regions.py`**: replace deprecated `selem=diamond(20)` with `footprint=diamond(20)` in the `binary_dilation` call.
- **`segmentation_school.py`**: when `--input_files` is provided and `--basedir` is not explicitly set (still at its default), auto-derive `--basedir` as `os.path.dirname(args.input_files)` so the directory never needs to be specified separately.

## Capabilities

### New Capabilities
- `checkpoint-validation`: Early validation that model checkpoint files exist before prediction starts, with a descriptive error message pointing to the missing directory.
- `subprocess-exit-check`: Prediction subprocess exit-code checking that aborts the pipeline with an actionable message instead of attempting mask reconstruction on empty output.
- `auto-basedir-from-input`: When an input image file is provided, automatically resolve the base/output directory from its parent path.

### Modified Capabilities

## Impact

- `Codes/IterativePredict_1X.py` — `get_test_step()` and `predict_xml()` functions
- `Codes/get_choppable_regions.py` — `binary_dilation` call
- `segmentation_school.py` — argument parsing / `main()` pre-processing
