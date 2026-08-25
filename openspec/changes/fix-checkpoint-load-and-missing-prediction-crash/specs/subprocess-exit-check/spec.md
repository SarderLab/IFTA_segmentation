## ADDED Requirements

### Requirement: DeepLab subprocess failure aborts prediction
`predict_xml()` must detect when the DeepLab eval subprocess exits with a non-zero code and raise an exception before calling `un_suey()`.

#### Scenario: DeepLab subprocess fails
- **WHEN** the DeepLab `eval.py` subprocess exits with a non-zero return code
- **THEN** `predict_xml()` raises an exception with a message that identifies the WSI file and the subprocess exit code
- **THEN** `un_suey()` is NOT called

#### Scenario: DeepLab subprocess succeeds
- **WHEN** the DeepLab `eval.py` subprocess exits with return code 0
- **THEN** `predict_xml()` continues to `un_suey()` as before (unchanged behavior)
