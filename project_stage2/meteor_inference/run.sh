#!/bin/bash
# =============================================
# 运行 Meteor-Mamba 陨石批量识别
# =============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Meteor-Mamba 陨石批量识别"
echo "=========================================="
echo "Script dir:  $SCRIPT_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Test images: $PROJECT_DIR/test_images/test_images"
echo ""

# 确保 Meteor 仓库在 PYTHONPATH 中
METEOR_PATH="/tmp/Meteor"
if [ -d "$METEOR_PATH" ]; then
    export PYTHONPATH="$METEOR_PATH:$PYTHONPATH"
    echo "Meteor found at: $METEOR_PATH"
fi

# 运行推理
cd "$SCRIPT_DIR"
python meteor_batch_infer.py

echo ""
echo "Done! Check output/submission_meteor.csv"
