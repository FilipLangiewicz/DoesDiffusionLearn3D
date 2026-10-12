# SLURM jobs for the corrected extraction

Use these jobs on the Athena cluster (PLGrid). Do the steps in this order.

1. Copy the repository and `data/final/images` to the project folder on the cluster.
2. Install the package in the environment: `pip install -e ".[model,track]"`.
3. Make the folder for the logs: `mkdir -p logs`.
4. Make the prompt cache on one GPU: `dd3d extract --config configs/extract.yaml --task-id 0 --limit 1`.
5. Submit the main job: `sbatch scripts/slurm/extract_v1.slurm`.
6. Submit the controls: `sbatch scripts/slurm/extract_controls.slurm`.
7. When all tasks stop, join the chunks: `dd3d merge data/features/v1_sana` (also for the two control folders).
8. Copy only the blocks and timesteps that you need to the laptop (860 MB for each block and timestep).

The old jobs of the ZZSN project are in `legacy/scripts/`. Do not use them for new data, because they use the wrong noise rule.
