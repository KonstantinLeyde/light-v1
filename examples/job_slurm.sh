#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=galaxy_reconstruction

# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start.
#              d-hh:mm:ss
#SBATCH --time=0-48:00:00

# Define the partition on which the job shall run. May be omitted.
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1

#SBATCH --mem=80G
#SBATCH --cpus-per-task 16

# The output file for your job.
#SBATCH --output=logs/%A_%a.out
#SBATCH --array=0

conda deactivate
module purge
source PATH_TO_YOUR_CONDA_LIGHT_ENV/bin/activate

id_job=${SLURM_JOBID}

time srun python main_analysis_gibbs.py --id_job $id_job --init_file launch.yaml 

# Finish the script
exit 0
