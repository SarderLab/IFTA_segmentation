## ADDED Requirements

### Requirement: basedir auto-derived from input_files
When `--input_files` is provided and `--basedir` is not explicitly set (is still the default value), the CLI must automatically set `basedir` to the parent directory of the input file path.

#### Scenario: input_files provided, basedir not set
- **WHEN** the CLI is called with `--input_files /some/path/image.svs` and no `--basedir` argument
- **THEN** `args.basedir` is set to `/some/path` (i.e., `os.path.dirname(args.input_files)`) before `main()` executes

#### Scenario: both input_files and basedir provided
- **WHEN** the CLI is called with both `--input_files` and an explicit `--basedir`
- **THEN** the explicit `--basedir` value is used unchanged (explicit always wins)

#### Scenario: input_files not provided
- **WHEN** the CLI is called without `--input_files`
- **THEN** existing `--basedir` / `--base_dir` behavior is unchanged
