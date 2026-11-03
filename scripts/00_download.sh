#!/usr/bin/env bash
# Download the local–global session of OpenNeuro ds005429 (Rutiku et al., 2024; CC0).
# Pulls top-level metadata, every subject's ses-LGeffect folder and the sourcedata
# (stimulation sequences + paradigm code). Add the other sessions with
# INCLUDE_ALL=1 bash scripts/00_download.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/data/bids"
DS=ds005429
TAG=1.0.0
mkdir -p "$TARGET"

if command -v openneuro-py >/dev/null 2>&1; then
    if [[ "${INCLUDE_ALL:-0}" == "1" ]]; then
        openneuro-py download --dataset "$DS" --tag "$TAG" --target-dir "$TARGET"
    else
        args=(--include dataset_description.json --include participants.tsv
              --include README --include CHANGES
              --include sourcedata)
        for s in $(seq -w 1 15); do
            args+=(--include "sub-$s/sub-${s}_sessions.tsv" --include "sub-$s/ses-LGeffect")
        done
        openneuro-py download --dataset "$DS" --tag "$TAG" --target-dir "$TARGET" "${args[@]}"
    fi
elif command -v aws >/dev/null 2>&1; then
    # Fallback: public S3 bucket, no credentials needed
    if [[ "${INCLUDE_ALL:-0}" == "1" ]]; then
        aws s3 sync --no-sign-request "s3://openneuro.org/$DS" "$TARGET"
    else
        aws s3 sync --no-sign-request "s3://openneuro.org/$DS" "$TARGET" \
            --exclude "sub-*/ses-MMN/*" --exclude "sub-*/ses-P3b/*"
    fi
else
    echo "Install openneuro-py (pip install openneuro-py) or the AWS CLI first." >&2
    exit 1
fi

echo "Done -> $TARGET"
find "$TARGET" -name "*LocalGlobal*_eeg.vhdr" | sort
