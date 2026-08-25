#!/usr/bin/env bash

# retire-girder-dependency: no more slicer_cli_web.cli_list_entrypoint (CLI-XML driven dispatch) —
# the job dispatcher (kidease_app/api/services/{k8s,container}_dispatch.py) passes plain env vars
# (ITEM_ID, STORAGE_API_URL, JOB_AUTH_TOKEN, MODEL_ID, BOX_SIZE_HR, OVERLAP_PERCENT_HR, BATCH_SIZE,
# OUTPUT_ANNOTATION_NAME) instead of CLI flags, so this runs the entrypoint script directly.
exec ${PYTHON_BIN:-/usr/bin/python} IFTASegmentation/IFTASegmentation.py
