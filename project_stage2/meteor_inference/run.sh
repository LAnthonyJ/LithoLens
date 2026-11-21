#!/bin/bash
# =============================================
# Run batch meteorite classification with Meteor-Mamba
# =============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Meteor-Mamba batch meteorite classification"
echo "=========================================="
echo "Script dir:  $SCRIPT_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Test images: $PROJECT_DIR/test_images/test_images"
echo ""

# Add the Meteor repository to PYTHONPATH
METEOR_PATH="/tmp/Meteor"
if [ -d "$METEOR_PATH" ]; then
    export PYTHONPATH="$METEOR_PATH:$PYTHONPATH"
    echo "Meteor found at: $METEOR_PATH"
fi

# Run inference
cd "$SCRIPT_DIR"
python meteor_batch_infer.py

echo ""
echo "Done! Check output/submission_meteor.csv"
