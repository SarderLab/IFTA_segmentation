## Why

When the DeepLab inference batch size (default 8) is too large for the available GPU memory, the model crashes with an unrecoverable OOM error on the first batch, causing the entire WSI segmentation job to fail. Users must manually retry with a smaller batch size, and there is no automatic recovery.

## What Changes

- Add automatic batch size reduction logic in the DeepLab prediction loop: if a GPU OOM error occurs during inference, catch it, reduce the batch size by half, reload the model, and retry from the beginning of that WSI's image list.
- Continue halving the batch size down to a minimum of 1; if OOM still occurs at batch_size=1, raise the error to the caller.
- Log a warning each time the batch size is reduced so operators can see what happened.

## Capabilities

### New Capabilities
- `gpu-oom-recovery`: Automatic GPU OOM detection and batch size fallback during DeepLab inference, ensuring jobs complete on constrained-memory GPUs without manual intervention.

### Modified Capabilities

## Impact

- `ifta/ifta_code/Codes/Deeplab_network/main.py`: prediction loop needs OOM catch + retry logic
- No API or CLI interface changes; batch_size parameter remains the starting value, not a guaranteed value
- No changes to checkpoint loading, model architecture, or output format
