## 1. model.py — replace plain-text progress with tqdm bar

- [x] 1.1 In `Codes/Deeplab_network/model.py` `predict()`, add a try/except tqdm import at the top of the method (or module level): `try: from tqdm import tqdm as _tqdm; HAS_TQDM = True` / `except ImportError: HAS_TQDM = False`
- [x] 1.2 Before the prediction loop, create the progress bar when tqdm is available: `pbar = _tqdm(total=total_images, desc="Total WSI progress", unit="image", colour="green", file=sys.stdout, dynamic_ncols=False, ncols=80)` — otherwise set `pbar = None`
- [x] 1.3 Inside the per-image loop, after saving each mask, call `pbar.update(1)` when `pbar` is not None (incrementing one image at a time for accurate throughput tracking)
- [x] 1.4 Remove the existing `Step X/N — Y/Z images (P%)` print and `sys.stdout.flush()` call; replace with the plain-text fallback `print(f'Step {step+1}/{total_steps} ...')` only when `pbar is None`
- [x] 1.5 After the prediction loop ends, call `pbar.close()` when `pbar` is not None, before the "output files saved" print
