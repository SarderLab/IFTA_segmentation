## ADDED Requirements

### Requirement: get_test_step raises on missing checkpoints
`get_test_step(modeldir)` must raise `FileNotFoundError` when no `.ckpt*` files are found in `modeldir`, instead of returning `0`.

#### Scenario: no checkpoint files in model directory
- **WHEN** `get_test_step` is called with a directory that contains no `*.ckpt*` files
- **THEN** a `FileNotFoundError` is raised with a message that includes the `modeldir` path

#### Scenario: checkpoint files are present
- **WHEN** `get_test_step` is called with a directory containing one or more `*.ckpt*` files
- **THEN** it returns the integer step number of the highest-numbered checkpoint (unchanged behavior)

### Requirement: selem deprecation resolved
`get_choppable_regions.py` must not trigger a `FutureWarning` about `selem` when calling `binary_dilation`.

#### Scenario: binary_dilation is called
- **WHEN** `get_choppable_regions` processes a tissue image
- **THEN** `binary_dilation` is called with `footprint=diamond(20)` (not `selem=diamond(20)`)
