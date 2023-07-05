#! /bin/bash

# metadata key (google cloud): shutdown-script or shutdown-script-url

train_process_pid=$(ps -e -o pid,command | grep python | grep train | head -n1 | awk '{print $1}')

echo "kill $train_process_pid"

kill -SIGINT $train_process_pid