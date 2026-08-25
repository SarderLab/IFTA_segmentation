## 1. DeepLab main.py — add Girder progress args

- [x] 1.1 In `Codes/Deeplab_network/main.py`, add `--girder_api_url`, `--girder_token`, and `--girder_job_id` arguments to the argument parser (all optional, default `None`)

## 2. model.py — per-step progress output and Girder REST updates

- [x] 2.1 In `Codes/Deeplab_network/model.py` `predict()`, compute `total_images = len(image_list)` and `total_steps` before the loop
- [x] 2.2 Replace `if step % 100 == 0: print('step {:d}'.format(step))` with a per-step print: `Step X/N — Y/Z images (P%)` followed by `sys.stdout.flush()`
- [x] 2.3 Before the loop, initialize a `girder_gc` client (using `girder_client.GirderClient`) when `self.conf.girder_job_id`, `self.conf.girder_api_url`, and `self.conf.girder_token` are all set; wrap in try/except and set `girder_gc = None` on failure
- [x] 2.4 After each step's print, call `girder_gc.patch(f'/job/{girder_job_id}', data={'progress': {...}})` when `girder_gc` is not None; wrap in try/except to silently ignore failures

## 3. IterativePredict_1X.py — forward Girder args to DeepLab subprocess

- [x] 3.1 In `predict_xml()`, check if `args` has `girder_api_url`, `girder_token`, and `girder_job_id` attributes and they are not None; if so, append `--girder_api_url`, `--girder_token`, `--girder_job_id` to `deeplab_cmd`

## 4. segmentation_school.py — add Girder progress args and fix base_dir

- [x] 4.1 Add `--girder_api_url`, `--girder_token`, and `--girder_job_id` arguments to the parser in `segmentation_school.py` (all optional, default `None`) so they can be forwarded to `IterativePredict_1X.py` via `args`
- [x] 4.2 Fix the `base_dir` auto-derive block: priority order is `--basedir` > `--base_dir` > `os.path.dirname(args.input_path)` > `os.path.dirname(args.input_files)` > raise `ValueError`

## 5. IFTASegmentation.xml — make base_dir optional

- [x] 5.1 In `ifta/cli/IFTASegmentation/IFTASegmentation.xml`, add `<default></default>` inside the `base_dir` `<directory>` element so Girder treats it as optional in the DSA UI

## 6. IFTASegmentation.py — omit base_dir when empty, read job ID and pass down

- [x] 6.1 In `IFTASegmentation.py` `main()`, only include `--base_dir` in the command string when `args.base_dir` is non-empty; when it is empty or absent, omit it so `segmentation_school.py` auto-derives it from `input_path`
- [x] 6.2 Read `job_id = os.environ.get('GIRDER_JOB_ID')` after the `gc` setup
- [x] 6.3 Add `--girder_job_id {job_id}` to the command string when `job_id` is not None; also pass `--girder_api_url` and `--girder_token` (already available as `args.girderApiUrl` and `args.girderToken`)
