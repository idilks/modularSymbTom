#!/bin/bash
#SBATCH --job-name=cma_small_gen        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G         # total memory
#SBATCH --gres=gpu:1           # number of gpus per node
#SBATCH --time=10:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/cma_small_%j.out
#SBATCH --error=slurm_logs/cma_small_%j.err
#SBATCH --partition=h200_preemptable
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=idil.k.sahin.26@dartmouth.edu

export PIP_CACHE_DIR="/dartfs/rc/lab/F/FranklandS/.pip/cache"
export TRANSFORMERS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_HOME="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_DATASETS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export HF_METRICS_CACHE="/dartfs/rc/lab/F/FranklandS/models/cache"
export WANDB_API_KEY="bd1c08839d0c8c49e7c3efe9aabe2d9c644befb6"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
cd
mkdir -p .conda/pkgs/cache .conda/envs

cd /dartfs/rc/lab/F/FranklandS/tom
conda activate /dartfs/rc/lab/F/FranklandS/tom/envs/tom_analysis

python codebase/tasks/identity_rules/cma.py  --use_behavioral_tom --context_type abstract --base_rule ABA --template_names food_truck --prompt_num 30
