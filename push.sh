#!/bin/bash
# Per-project wrapper for the e156 push pipeline.
# Edit only the slug below — actual logic lives in ../_shared/e156-push.sh
exec bash "$(dirname "$0")/../_shared/e156-push.sh" "trialatlas" "$@"
