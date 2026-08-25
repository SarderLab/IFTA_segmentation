## ADDED Requirements

### Requirement: base_dir auto-derived from input_path
When `--base_dir` is not provided, `segmentation_school.py` SHALL derive `base_dir` as `os.path.dirname(os.path.abspath(args.input_path))` when `--input_path` is set. If `--input_path` is also absent, it SHALL fall back to `os.path.dirname(os.path.abspath(args.input_files))` when `--input_files` is set. If none of the three are provided, it SHALL raise a `ValueError` with a descriptive message.

#### Scenario: base_dir derived from input_path
- **WHEN** `--base_dir` is not passed and `--input_path` is `/mnt/worker/abc/image.svs`
- **THEN** `args.base_dir` is set to `/mnt/worker/abc`

#### Scenario: explicit base_dir takes precedence
- **WHEN** both `--base_dir /some/path` and `--input_path /other/image.svs` are passed
- **THEN** `args.base_dir` is `/some/path` (explicit value is not overridden)

#### Scenario: no base_dir and no input_path raises error
- **WHEN** neither `--base_dir`, `--basedir`, `--input_path` nor `--input_files` can supply a base directory
- **THEN** a `ValueError` is raised before `main()` is called
