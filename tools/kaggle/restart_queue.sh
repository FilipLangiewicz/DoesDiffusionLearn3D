#!/bin/bash
# Stop and start the Kaggle job queue, detached from the terminal.
# The job states are in marker files, so a restart loses nothing.
# Do not type "pkill -f kqueue.py" in a shell: the pattern also matches that shell itself.
cd "$(dirname "$0")/../.."
for pid in $(pgrep -f "python -u tools/kaggle/kqueue.py"); do
  kill "$pid"
done
sleep 2
setsid nohup .venv/bin/python -u tools/kaggle/kqueue.py runs/kaggle/queue.txt >> runs/kaggle/logs/queue.log 2>&1 < /dev/null &
disown
sleep 1
pgrep -af "python -u tools/kaggle/kqueue.py"
