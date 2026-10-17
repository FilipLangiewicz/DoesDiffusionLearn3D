# How to reproduce the results

This file shows the path from the data to each table and figure of the article. Each step names the job spec, the script and the result file.

## 1. Where the work runs

| Place | What runs there | Data |
|---|---|---|
| Laptop (CPU) | Unit tests, lint, the label audit, the tables and figures from `reports/`, the article PDF | Labels, splits, `reports/`, `data/examples/` |
| Kaggle CPU job | Renders (Blender 2.78c), probes, plans of the causal tests, analysis of output images | Renders and features of earlier jobs |
| Kaggle GPU job (T4) | Feature extraction, causal tests and patch tests | Renders, plans |

The renders and the features are on Kaggle only (`docs/KAGGLE_DATA.md`). Do not copy them to the laptop. The data sets and job outputs are private to the project account. To repeat the work with another account, copy them first.

## 2. Prepare the environment once

1. Make the environment: `make venv install`.
2. Run the checks: `make test` and `make lint`. All tests must pass.
3. Put the Kaggle API token in `~/.kaggle/`.
4. Optional: set `KAGGLE_CLI` to a wrapper of the Kaggle CLI, for example a wrapper that logs each call. Without it, the tools use the CLI of the virtual environment.

## 3. Run a Kaggle job

A job clones this repository at a commit. Thus, push the commit first.

1. Write or change the spec in `configs/kaggle/`. The analysis specs come from `python tools/kaggle/analysis_specs.py`.
2. Write the job folder:
   - analysis: `python tools/kaggle/make_jobs.py analyze --spec configs/kaggle/<spec>.json --ref <commit>`;
   - extraction: `python tools/kaggle/make_jobs.py extract --name <name> --tasks 1 --gpus 2 --runs <runs> --kernel <render job> --ref <commit> --out-root runs/kaggle/extract`;
   - causal or patch test: `python tools/kaggle/make_jobs.py intervene --name <name> --spec configs/kaggle/intervene_<name>.json --ref <commit>`, then move the folder to `runs/kaggle/intervene/`.
3. Add one line for the job to `runs/kaggle/queue.txt` (format in `docs/KAGGLE_RUNBOOK.md`). Use `after=<slug prefix>` for a job that needs the output of another job.
4. Start the queue: `tools/kaggle/restart_queue.sh`. It pushes the jobs, waits for them and downloads the small outputs.
5. Copy the small results into git: `python tools/kaggle/sync_reports.py`.

## 4. From the jobs to the article

| Article item | Jobs (spec in `configs/kaggle/`) | Script | Result file |
|---|---|---|---|
| Table 1: original probe and pixels (v1) | ZZSN results; `scan_v1`, `deep_v1` | `feature_scan.py`, `best_probes.py` | `probe_exp/pca256_job_2628839/`, `reports/experiments/v1/best/predictions.csv` |
| Table 2: noise rule | GPU test of the noise rule | `tools/diffusers_upstream/functional/repro_sana.py` | `tools/diffusers_upstream/functional/kaggle_sana1600m_result.txt`; unit test `tests/unit/test_noise.py` |
| Table 3: pixel probes on image regions | local, with the v1 images | `scripts/experiments/pixel_cues.py` (`make cues`) | `reports/experiments/pixel_cues/results.md` |
| Label and arrow audit | local | `scripts/audit/arrow_audit.py` (`make audit`) | `reports/audit/`, `docs/audit/labels_and_arrows.md` |
| Table 4: probes on all data sets | `scan_<set>`, `deep_<set>`, then `vis_<set>` (DINOv2, CLIP) | `feature_scan.py`, `best_probes.py`, then local `rq1_table.py` | `reports/experiments/<set>/best/`, `reports/experiments/rq1.md`, `docs/article/tables/rq1.tex` |
| Figure 4: probe scan | as Table 4 | local `result_figures.py` | `docs/article/figures/scan.png` |
| Position transfer (v2 to v2r) | `transfer_pos`, `transfer_pos_vision` | `transfer.py` | `reports/experiments/transfer/pos*/transfer.md` |
| Cue transfer (v2r to v2rns, v2rso) | `transfer_cue_pooled` | `transfer.py` | `reports/experiments/transfer/cue_pooled/transfer.md` |
| Probe maps (object and shadow cells) | `deep_<set>` | `probe_maps.py` | `reports/experiments/<set>/maps/regions.md` |
| Shift test, blocks 4, 10, 18 | `plans_v2` (CPU), `intervene_causal` (GPU), `reanalyze_v2` (CPU) | `make_intervention_plan.py`, `dd3d.intervene.run`, `analyze_interventions.py` | `reports/experiments/causal/b*/results.csv` |
| Shift test, block 5 | `plans_v2_b05`, `intervene_causal_b05`, `reanalyze_v2` | as above | `reports/experiments/causal_b05/` |
| Shift test with more noise | `intervene_causal_sigma`, `reanalyze_v2` | as above | `reports/experiments/causal_sigma/` |
| Shift in all blocks | `plans_multi`, `intervene_multi`, `reanalyze_v2` | `make_multiblock_plan.py`, as above | `reports/experiments/causal_multi/` |
| Erasure with one eraser for each cell | `plans_erase_v2`, `intervene_erase`, `erase_v2` | `make_intervention_plan.py --cell-eraser --only-erase`, as above | `reports/experiments/plans/b*e/plan.json` (probe after erasure), `reports/experiments/causal_erase/` |
| Patch test | local plan, `intervene_patch`, `reanalyze_v2` | `make_patch_plan.py`, `dd3d.intervene.run_patch`, `analyze_patching.py` | `reports/experiments/patch/summary.csv` |
| Calibration of the output measures | `reanalyze_v2` | `calibrate_measures.py` | `reports/experiments/measures/measures.md` |
| Other figures and tables | the result files above | local `result_figures.py` (scan, causal and patch figures, Table 5), `make_article_figures.py` (conventions, image regions, examples) | `docs/article/figures/`, `docs/article/tables/` |
| Article PDF | — | `make article` (tectonic) | `docs/article/build/main.pdf` |

The plan of the patch test comes from `python scripts/experiments/make_patch_plan.py --only <pairs> --out <folder>` with the pairs of `configs/kaggle/patch_pairs.txt`. Upload its two plan files to the Kaggle data set `dd3d-plans` with the extension `.dat`. Kaggle unpacks `.npz` files of a data set, because they are zip files.

## 5. Make the tables and figures again on the laptop

After `sync_reports.py`, run these commands:

```bash
PYTHONPATH=src python scripts/experiments/rq1_table.py
PYTHONPATH=src python scripts/experiments/result_figures.py
make article
```

Then look at each changed figure and at the pages of the PDF.
