## Context

The pipeline runs DeepLab prediction on 3000×3000px tiles. Each tile is ~108MB uncompressed FP32. At batch=2 the GPU receives ~216MB input per forward pass against a 24GB VRAM budget — roughly 1% utilization. The L4 GPU has dedicated FP16/BF16 tensor cores that are idle in the current FP32-only inference path.

The batch size flows through: `IFTASegmentation.py` → `segmentation_school.py` (via `--batch_size`) → `IterativePredict_1X.py` (`args.batch_size`) → `Deeplab_network/main.py` (via `--batch_size`) → `model.py` (`self.conf.batch_size`). The entire chain already exists; we only need to change the default and expose the top of the chain to the UI.

## Goals / Non-Goals

**Goals:**
- ~4× speedup from batch size increase (2→8)
- ~1.5× additional speedup from FP16 inference on L4 tensor cores
- User can override batch size at job submission time (1–16)
- Zero change to output masks or annotation quality

**Non-Goals:**
- FP16 training (inference only)
- Changing tile size or overlap
- Supporting batch sizes above 16 (memory safety margin)

## Decisions

**D1 — FP16 policy scoped to predict() only, restored afterward**

Set `tf.keras.mixed_precision.set_global_policy('mixed_float16')` at the start of `predict()` and restore `'float32'` in a `finally` block. This avoids any risk of FP16 leaking into training or other model operations if the model object is reused.

Alternative considered: set policy at process startup in `main.py`. Rejected — too broad, could affect checkpoint loading or other code paths.

**D2 — Clamp batch_size to max 16 in IFTASegmentation.py, not XML**

XML `<integer>` elements have no built-in max enforcement in all DSA versions. Enforce the clamp in Python: `batch_size = min(int(args.batch_size), 16)` before building the command string. The XML description states the max so users are informed.

**D3 — Default 8, not higher**

batch=8 uses ~864MB input VRAM + model weights (~500MB) + activations — comfortably within 24GB. batch=16 is likely fine too but 8 is the conservative recommended default. Users with confidence in their GPU can increase to 16 via the UI.

## Risks / Trade-offs

- [FP16 numerical precision] Rare edge case: very small activation values could underflow to zero in FP16. For semantic segmentation inference this is negligible in practice. → Mitigation: policy is scoped only to `predict()`, trivially reversible.
- [Batch size > VRAM] If a user runs on a GPU smaller than L4, batch=8 at 3000px tiles might OOM. → Mitigation: max cap at 16 in code, description in XML guides users to reduce for smaller GPUs.
- [FP16 + checkpoint compatibility] The checkpoint was saved in FP32. Mixed precision inference loads FP32 weights and casts activations — this is the standard TF2 pattern and is fully compatible.
