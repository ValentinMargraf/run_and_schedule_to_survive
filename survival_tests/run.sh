#!/bin/bash
#SBATCH -J "MAL"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH -A hpc-prf-aiafs
#SBATCH -t 0-05:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-aiafs/vmargraf/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-aiafs/vmargraf/clusterout/%x-%j
#SBATCH --array 1-100

cd /scratch-n2/hpc-prf-aiafs/vmargraf
conda config --add envs_dirs /scratch/hpc-prf-aiafs/vmargraf/.conda/envs
conda config --add pkgs_dirs /scratch/hpc-prf-aiafs/vmargraf/.conda/pkgs

# Load modules
cd $PFS_FOLDER/algorithm_survival_analysis-23-merge-changes-by-marcel/survival_tests/
ml lang
#ml Python/3.9.5
#ml Python/3.9.5-GCCcore-10.3.0

# activate environment
source $PFS_FOLDER/algorithm_survival_analysis-23-merge-changes-by-marcel/survival/bin/activate

export PYTHONPATH="/scratch/hpc-prf-aiafs/vmargraf/algorithm_survival_analysis-23-merge-changes-by-marcel:$PYTHONPATH"
# Set up environment variables
export PYTHONUSERBASE=$PFS_FOLDER/.local
export PATH=$PFS_FOLDER/.bin:$PATH
export PATH=$PFS_FOLDER/.local/bin:$PATH
which python
which pip

# Run the script in a loop for i from 0 to 5
#for i in 0; do
export SCRIPT_FILE=$PFS_FOLDER/algorithm_survival_analysis-23-merge-changes-by-marcel/survival_tests/run_schedule2.py
python $SCRIPT_FILE $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID

done

exit 0










