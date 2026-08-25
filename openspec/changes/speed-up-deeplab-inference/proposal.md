## Why

DeepLab prediction is the dominant runtime cost in the IFTA pipeline. The current default batch size of 2 leaves the L4 GPU (24GB VRAM) heavily underutilized — each forward pass processes only 2 tiles while the GPU idles waiting for the next load. Additionally, FP16 (half-precision) inference is not enabled, leaving L4 tensor-core throughput on the table. Together these two changes are expected to deliver ~4–6× speedup with no accuracy impact.

## What Changes

- **Default batch size 2 → 8** in `segmentation_school.py`. The L4 GPU has ample VRAM to handle 8 × 3000×3000px tiles per forward pass. Batch inference with DeepLabV2 at eval mode is mathematically identical per-tile regardless of batch size.
- **FP16 mixed precision** enabled in `model.py` `predict()` only, using `tf.keras.mixed_precision.set_global_policy('mixed_float16')` before inference and restoring `'float32'` afterward. Model weights remain FP32; only activations are cast to FP16. L4 tensor cores accelerate FP16 by ~1.5–2×.
- **Configurable batch size in the DSA job UI**: expose `batch_size` as an integer parameter in `IFTASegmentation.xml` (default=8, max=16) and forward it through `IFTASegmentation.py` to `segmentation_school.py`. Users can reduce it to 4 on smaller GPUs or increase up to 16 on the L4.
- **Suppress spurious CUDA log noise**: set `TF_CPP_MIN_LOG_LEVEL=3` before any TF import in `main.py` to silence the cuDNN/cuFFT/cuBLAS "already registered" C++ warnings.
- **Fix GPU memory-growth configuration error**: parse `--gpu` from `sys.argv` before TF imports in `main.py` and set `CUDA_VISIBLE_DEVICES` immediately, so `set_memory_growth` is called before TF initializes devices.

## Capabilities

### New Capabilities
- `configurable-batch-size`: User-adjustable batch size parameter in the DSA job submission UI, clamped to [1, 16].

### Modified Capabilities
- `prediction-progress`: Batch size change affects images-per-step count displayed in the tqdm bar — no spec change needed, the bar already tracks per-image.

## Impact

- `ifta/ifta_code/segmentation_school.py` — `--batch_size` default
- `ifta/ifta_code/Codes/Deeplab_network/model.py` — `predict()` FP16 policy
- `ifta/cli/IFTASegmentation/IFTASegmentation.xml` — new `batch_size` integer parameter
- `ifta/cli/IFTASegmentation/IFTASegmentation.py` — forward `args.batch_size` in command string
- `ifta/ifta_code/Codes/Deeplab_network/main.py` — early `CUDA_VISIBLE_DEVICES` + `TF_CPP_MIN_LOG_LEVEL` before TF imports
