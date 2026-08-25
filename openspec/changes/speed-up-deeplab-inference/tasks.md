## 1. segmentation_school.py — update default batch size

- [x] 1.1 In `segmentation_school.py`, change `--batch_size` argument default from `2` to `8`

## 2. model.py — FP16 mixed precision for prediction

- [x] 2.1 In `model.py` `predict()`, set `tf.keras.mixed_precision.set_global_policy('mixed_float16')` before the prediction loop, wrapped in a try/finally that restores `tf.keras.mixed_precision.set_global_policy('float32')` after the loop

## 3. IFTASegmentation.xml — expose batch_size parameter

- [x] 3.1 In `IFTASegmentation.xml`, add a `batch_size` `<integer>` parameter with `<default>8</default>` after the `output_annotation_name` parameter (already has index 6 from a previous partial edit — verify and set correctly)

## 4. IFTASegmentation.py — clamp and forward batch_size

- [x] 4.1 In `IFTASegmentation.py` `main()`, read `batch_size = min(int(args.batch_size), 16)` after existing arg handling
- [x] 4.2 Add `--batch_size {batch_size}` to the command string passed to `segmentation_school.py`

## 5. main.py — fix GPU init order and suppress CUDA log noise

- [x] 5.1 At the very top of `main.py`, before `import tensorflow as tf` and `from model import Model`, add: parse `--gpu` value from `sys.argv` (default `'0'`), then set `os.environ['CUDA_VISIBLE_DEVICES']` and `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'`
- [x] 5.2 In `setup_gpu()`, remove the `os.environ['CUDA_VISIBLE_DEVICES']` line (already set before import); keep only the `set_memory_growth` call and its try/except
