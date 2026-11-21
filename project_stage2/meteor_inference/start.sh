#!/bin/bash
cd /data/数据科学实践project/meteor_output
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/tmp/Meteor:$PYTHONPATH
nohup python run_on_server.py > meteor_inference.log 2>&1 &
echo "PID: $!"
