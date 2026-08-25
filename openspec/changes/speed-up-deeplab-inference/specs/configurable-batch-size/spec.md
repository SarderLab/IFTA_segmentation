## ADDED Requirements

### Requirement: Default batch size is 8
`segmentation_school.py` SHALL default `--batch_size` to `8` instead of `2`.

#### Scenario: No batch_size provided
- **WHEN** the pipeline is invoked without an explicit `--batch_size` argument
- **THEN** DeepLab runs with batch size 8

### Requirement: Batch size exposed in DSA job UI
`IFTASegmentation.xml` SHALL include a `batch_size` integer parameter with default 8. `IFTASegmentation.py` SHALL clamp the value to a maximum of 16 before passing it to `segmentation_school.py`.

#### Scenario: User sets batch_size in DSA UI
- **WHEN** a user submits a job with `batch_size=4`
- **THEN** `segmentation_school.py` receives `--batch_size 4` and DeepLab runs with batch size 4

#### Scenario: batch_size clamped at 16
- **WHEN** a user submits a job with `batch_size=32`
- **THEN** `IFTASegmentation.py` clamps it to 16 before passing it down

#### Scenario: batch_size forwarded in command string
- **WHEN** `IFTASegmentation.py` builds the segmentation_school command
- **THEN** `--batch_size {batch_size}` appears in the command string

### Requirement: FP16 mixed precision during prediction
`model.py` `predict()` SHALL set `tf.keras.mixed_precision.set_global_policy('mixed_float16')` before the prediction loop and restore `'float32'` in a `finally` block after the loop completes or raises.

#### Scenario: FP16 active during prediction
- **WHEN** `predict()` is called
- **THEN** the global mixed precision policy is `mixed_float16` for the duration of the forward passes

#### Scenario: FP16 policy restored after prediction
- **WHEN** `predict()` completes or raises an exception
- **THEN** the global mixed precision policy is restored to `float32`

#### Scenario: Output masks unchanged
- **WHEN** prediction runs with FP16 enabled
- **THEN** mask PNG files are saved as uint8 (unchanged from current behavior)
