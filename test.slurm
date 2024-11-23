#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment


##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=pdtb-llama3.1-8b-instruct
#SBATCH --time=1:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1    #Request 1 task
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mem=100G      
#SBATCH --output=pdtb-llama3.1-8b-instruct.%j      
#SBATCH --partition=gpu              #Request the GPU partition/queue

# modules needed for running DL jobs. Module restore will also work
module load GCCcore/12.2.0 Python/3.10.8

# Python venv
source $SCRATCH/venv/bin/activate

# scripts or executables
cd $SCRATCH/pdtb

chmod +x test.sh
./test.sh