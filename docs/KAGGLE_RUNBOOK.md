# Kaggle runbook

This file tells how the project runs work on Kaggle. Read it before you start or change a Kaggle job.

## Rules

- Do not download feature files (about 10 GB for each job) to the laptop. Drive C: is small, and the WSL disk file does not shrink by itself. Only logs and small result files come to the laptop.
- Do not unpack renders or features on the laptop. The analysis runs as a Kaggle job (see Analysis jobs). The laptop keeps only labels, splits, small results and the examples in `data/examples/`.
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
| `make_jobs.py render/extract/intervene/analyze` | Writes job folders (run.py + kernel-metadata.json) from `templates/`. `extract --gpus 2` uses both GPUs of a T4x2 session. `intervene --spec` and `analyze --spec` read a JSON file in `configs/kaggle/`. |
| `analysis_specs.py` | Writes the specs of all analysis jobs to `configs/kaggle/` (scan, deep, transfer, plans, causal, patch). |
| `sync_reports.py` | Copies the small results of finished analysis jobs from `runs/kaggle/analyze/` to `reports/experiments/` (in git). |
| `kqueue.py` | The queue. Pushes the jobs of `runs/kaggle/queue.txt` when Kaggle has a free session, checks their state and downloads the output with `kfetch.py`. For extraction jobs, render jobs and render packs, it downloads only `results.txt` and the log. After each round it writes `docs/KAGGLE_DATA.md`. |
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

Two more markers control the download of a finished job:

- `.small`: the queue downloads only `results.txt` and the log. Use it for GPU jobs with many output images. An analysis job reads these images on Kaggle.
- `.pattern`: a regular expression of the files to download, for example `results|[.]log$|[.]json$`.

## Start a new job

1. Commit and push the code.
2. `python tools/kaggle/make_jobs.py extract --name <set> --tasks 6 --gpus 2 --runs sana,sd15 --kernel jankombajn/<source> ... --ref <commit> --out-root runs/kaggle/extract`
3. Add the job folders to `runs/kaggle/queue.txt`.
4. If the queue does not run (`pgrep -af kqueue.py` shows nothing): `tools/kaggle/restart_queue.sh`.

## Analysis jobs (CPU)

The template `templates/analyze.py` runs scripts of the repository on Kaggle, next to the stored features. The spec (`configs/kaggle/<name>.json`) gives the render sets, the feature sources, the inputs and the commands. The job does these steps:

1. It clones the repository at the given commit.
2. It unpacks the render tars of each set and makes the splits. The tars come from the job `dd3d-renders` or from render jobs.
3. It stops if the image ids of a split differ from the local split (md5 check).
4. It links the feature sources of each set into `{FEAT}/<set>`. `dd3d.store` reads the chunks in place from data sets and tar files.
5. It links the outputs of earlier jobs and the data set `dd3d-plans` into `{PREV}/<slug>`.
6. It runs the commands. The results go to `{OUT}`, and the queue downloads them to the laptop.

The stages: `scan-<set>` (all blocks and timesteps), then `deep-<set>`, `transfer-pos` and `transfer-cue`. The second stage reads `scan.csv` from `{PREV}`. `plans-v2` writes the plans of the causal test. `causal-v2` and `patch-v2` analyse the images of the GPU jobs on Kaggle.

Test a new template or spec with a light job first (`configs/kaggle/smoke2.json`: one block, one timestep).

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
- Do not attach the data set `dd3d-renders-tar` (18,000 PNG files). A job with it waited 47 minutes for its inputs. Use the job output `dd3d-renders` (tar files).
- A job that reads large tar files of other jobs can wait some minutes at the start, because Kaggle copies the inputs.
- The queue downloads all of `{OUT}` of an analysis job. Do not write large files there, for example plan files of 150 MB. If you must, add a `.pattern` marker.
