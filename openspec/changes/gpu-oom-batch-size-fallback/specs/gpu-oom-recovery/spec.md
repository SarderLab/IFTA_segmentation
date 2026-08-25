## ADDED Requirements

### Requirement: Automatic batch size fallback on GPU OOM
When a GPU out-of-memory error occurs during DeepLab inference, the system SHALL automatically reduce the batch size by half and restart inference from the beginning of the current WSI's image list. This SHALL repeat until either inference succeeds or the batch size would fall below 1, at which point the error SHALL be re-raised.

#### Scenario: OOM on first batch with batch_size=8
- **WHEN** inference begins with batch_size=8 and a GPU OOM error occurs on the first batch
- **THEN** the system logs a warning containing the original and new batch size, rebuilds the dataset with batch_size=4, reloads the checkpoint, and restarts inference from image 0

#### Scenario: OOM at batch_size=1
- **WHEN** inference is attempted with batch_size=1 and a GPU OOM error still occurs
- **THEN** the system re-raises the exception without further retry, resulting in a non-zero exit code

#### Scenario: OOM mid-WSI
- **WHEN** inference processes some batches successfully then encounters a GPU OOM
- **THEN** the system halves the batch size, restarts from image 0, and overwrites any already-saved prediction files

### Requirement: OOM warning logging
When the system reduces batch size due to GPU OOM, it SHALL print a warning to stdout that includes the error message, the original batch size, and the new reduced batch size before restarting inference.

#### Scenario: Warning message content
- **WHEN** batch size is reduced from N to N/2 due to OOM
- **THEN** a line is printed to stdout containing both the old and new batch size values prior to restarting
