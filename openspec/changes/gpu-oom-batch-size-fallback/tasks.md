## 1. Core OOM Fallback Logic

- [x] 1.1 In `model.py`, refactor `_predict_impl` to accept a `batch_size` parameter (defaulting to `self.conf.batch_size`) so it can be called recursively/iteratively with a reduced value
- [x] 1.2 Wrap the inference loop body in a try/except that catches `tf.errors.ResourceExhaustedError` and generic exceptions containing "failed to allocate memory" or "ResourceExhausted"
- [x] 1.3 On OOM catch: if `batch_size // 2 >= 1`, print a warning with old and new batch size, then call `prediction_setup()` with the halved batch size and restart inference; otherwise re-raise
- [x] 1.4 Update `prediction_setup()` to accept and use the passed `batch_size` instead of always reading from `self.conf.batch_size`, so dataset rebuild uses the reduced size

## 2. Default Batch Size Change

- [x] 2.1 Change the default `batch_size` in `main.py` argparse from `15` to `8`

## 3. Validation

- [ ] 3.1 Verify the warning message prints old and new batch size when OOM is simulated
- [ ] 3.2 Verify that at batch_size=1 an OOM re-raises instead of looping
- [ ] 3.3 Run a smoke test on a small WSI to confirm normal (non-OOM) inference still works end-to-end
