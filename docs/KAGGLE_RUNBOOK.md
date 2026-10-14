# Kaggle runbook

This file tells how the project runs work on Kaggle. Read it before you start or change a Kaggle job.

## Rules

- Do not download feature files (about 10 GB for each job) to the laptop. Drive C: is small, and the WSL disk file does not shrink by itself. Only logs and small result files come to the laptop.
- Do not run `scripts/experiments/auto_process.sh`. It fits the probes on the laptop and needs the feature files on the local disk. Run the analysis as a Kaggle job instead.
- Do not push a job again if it has a `.done` marker. A new version replaces the old output on Kaggle.
- Push the code to GitHub before you write a job. Each job clones the repository at a given commit. `make_jobs.py` refuses a commit that is not on `origin`.
- Do not stop the queue with `pkill -f kqueue.py` in a shell: the pattern also matches the shell itself. Use `tools/kaggle/restart_queue.sh`.

## Account and limits

- Account `jankombajn`. The API token is in `~/.kaggle/`; never print it. Use `/home/flang/tools/kag` for all Kaggle CLI calls (log in `~/kaggle.log`).
- At most 5 CPU jobs and 2 GPU jobs at the same time. GPU quota: 30 hours each week.
- Accelerator `NvidiaTeslaT4` = one T4. Accelerator `NvidiaTeslaT4Highmem` = two T4 and 31 GB RAM.
- A job stops after 12 hours. Its output folder can have at most 20 GB and at most 500 files; pack many files into tar files.

## Tools (`tools/kaggle/`)

| Tool | Use |
|---|---|
| `make_jobs.py render/extract/intervene` | Writes job folders (run.py + kernel-metadata.json) from `templates/`. `extract --gpus 2` uses both GPUs of a T4x2 session. |
| `kqueue.py` | The queue. Pushes the jobs of `runs/kaggle/queue.txt` when Kaggle has a free session, checks their state and downloads the output with `kfetch.py`. For extraction jobs and render packs, it downloads only `results.txt` and the log. After each round it writes `docs/KAGGLE_DATA.md`. |
| `restart_queue.sh` | Stops and starts the queue, detached from the terminal. |
| `kfetch.py` | Downloads job output files to disk with resume (`--pattern` selects files). The Kaggle CLI reads each file into RAM, so do not use `kaggle kernels output` for large files. |
| `data_list.py` | Writes `docs/KAGGLE_DATA.md` (data sets and job outputs with their state). Add new data sets to its list. |
| `collect.py` | Unpacks downloaded tar files and merges feature chunks (local use only). |

## Queue file

`runs/kaggle/queue.txt` has one job on each line, in the order of priority:

```
<job folder> <CPU|GPU> [after=<slug prefix>] [acc=NvidiaTeslaT4Highmem]
```

The queue reads the file in each round (every 2 minutes), so a new line needs no restart. Marker files in each job folder keep the state: `.pushed`, `.complete`, `.done`, `.failed`. To run a job again, delete its markers (and see the rules above).

## Start a new job

1. Commit and push the code.
2. `python tools/kaggle/make_jobs.py extract --name <set> --tasks 6 --gpus 2 --runs sana,sd15 --kernel jankombajn/<source> ... --ref <commit> --out-root runs/kaggle/extract`
3. Add the job folders to `runs/kaggle/queue.txt`.
4. If the queue does not run (`pgrep -af kqueue.py` shows nothing): `tools/kaggle/restart_queue.sh`.

## Logs

- Queue: `tail -f runs/kaggle/logs/queue.log`; only events: `grep -v "state {" runs/kaggle/logs/queue.log | tail`.
- Job: `runs/kaggle/<kind>/<slug>/output/results.txt` (short) and `<slug>.log` (full log, JSON).
- The API gives no log while a job runs; open `https://www.kaggle.com/code/jankombajn/<slug>`.

## Job inputs and outputs

- Inputs of a job: data sets (`dataset_sources`) and outputs of other jobs (`kernel_sources`). They appear under `/kaggle/input/datasets/...` and `/kaggle/input/notebooks/jankombajn/<slug>/`. Kaggle unpacks tar and zip files of a data set, but not of a job output.
- Each extraction task writes one tar with `<run>/block_<bb>/t_<ttt>/chunk_<task>.npy`, `chunk_<task>_ids.txt`, `meta.json` and `stats/`. With `--gpus 2`, task T of N writes chunks 2T and 2T+1 of 2N.

## Data sets

The Kaggle API makes a data set only from local files. The user makes data sets from job outputs in the browser (job page, Output, New Dataset; one job for each data set). The list is in `docs/KAGGLE_DATA.md`.

## Known problems

- Hugging Face downloads without a token can fail with HTTP 429; the jobs try again.
- The T4 has no bfloat16. SANA runs in float16 (same features as float32). The old noise rule overflows in float16 and runs in float32.
- Blender 2.78c cannot use the Kaggle GPUs; renders run on the CPU (about 7 s for each image).
- A restart of WSL stops the queue; start it again with `restart_queue.sh`.
