## 1. Fix get_test_step — raise on missing checkpoints

- [x] 1.1 In `Codes/IterativePredict_1X.py`, update `get_test_step()`: after the loop over `pretrains`, if `maxmodel == 0` and `len(pretrains) == 0`, raise `FileNotFoundError(f"No checkpoint files found in model directory: {modeldir}")`
- [x] 1.2 Verify the same function in `Codes/IterativePredict.py` and apply the same fix if it has the same silent-zero issue

## 2. Fix predict_xml — check subprocess return code

- [x] 2.1 In `Codes/IterativePredict_1X.py`, replace the `call([...])` invocation in `predict_xml()` with `result = subprocess.run([...])` and add `import subprocess` if not already imported
- [x] 2.2 After the `subprocess.run()` call, check `result.returncode != 0` and raise `RuntimeError(f"DeepLab prediction failed for {wsi} (exit code {result.returncode}). Aborting reconstruction.")` before reaching `un_suey()`

## 3. Fix deprecated selem argument in get_choppable_regions

- [x] 3.1 In `Codes/get_choppable_regions.py`, change `binary_dilation(binary, selem=diamond(20))` to `binary_dilation(binary, footprint=diamond(20))`

## 4. Auto-derive basedir from input_files

- [x] 4.1 In `segmentation_school.py`, add `--input_files` and `--basedir` arguments to the parser if not already present (with `--basedir` default `None`)
- [x] 4.2 After `args = parser.parse_args()`, add logic: if `args.input_files` is set and `args.basedir` is `None` (not explicitly provided), set `args.base_dir = os.path.dirname(os.path.abspath(args.input_files))`
- [x] 4.3 Ensure existing `--base_dir` behavior is preserved when `--input_files` is absent or `--basedir` is explicitly passed
