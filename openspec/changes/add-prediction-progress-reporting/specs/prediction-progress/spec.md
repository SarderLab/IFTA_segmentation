## ADDED Requirements

### Requirement: Per-step stdout progress
The prediction loop in `model.py` SHALL print one line per batch step to stdout, immediately flushed, in the format `Step X/N — Y/Z images (P%)` where X is the current step (1-based), N is total steps, Y is images processed so far, Z is total images, and P is percentage complete.

#### Scenario: Progress line printed each step
- **WHEN** the DeepLab predict loop processes a batch
- **THEN** a line `Step X/N — Y/Z images (P%)` is printed and flushed to stdout before the next batch begins

#### Scenario: Final step reaches 100%
- **WHEN** the last batch is processed
- **THEN** the progress line shows 100% and the image count equals total images

### Requirement: Girder job progress bar integration
When `--girder_api_url`, `--girder_token`, and `--girder_job_id` are all provided to `main.py`, the system SHALL call `PATCH /job/{id}` on the Girder REST API at each batch step with `progress: {current, total, message}`.

#### Scenario: Progress bar updates during prediction
- **WHEN** all three Girder args are present and a batch step completes
- **THEN** the Girder job's progress is updated via REST with current image count and percentage message

#### Scenario: Missing Girder args — no crash
- **WHEN** one or more of the Girder args are absent or empty
- **THEN** prediction proceeds normally with no Girder REST calls and no errors

#### Scenario: Girder API call fails — no crash
- **WHEN** the Girder REST call raises any exception
- **THEN** the exception is silently caught and prediction continues uninterrupted

### Requirement: Girder job ID forwarded through subprocess chain
`IFTASegmentation.py` SHALL read `GIRDER_JOB_ID` from the environment and pass it as `--girder_job_id` to `segmentation_school.py`. `IterativePredict_1X.py` SHALL forward `girder_api_url`, `girder_token`, and `girder_job_id` from `args` into the DeepLab subprocess command when they are present.

#### Scenario: Job ID propagates to DeepLab
- **WHEN** `GIRDER_JOB_ID` is set in the Girder worker environment
- **THEN** the value reaches `model.py` as `self.conf.girder_job_id` and is used for REST progress updates

#### Scenario: No job ID set — silent skip
- **WHEN** `GIRDER_JOB_ID` is not in the environment
- **THEN** no `--girder_job_id` arg is added to the command and no Girder REST calls are made
