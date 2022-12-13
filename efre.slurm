#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            	#Do not propagate environment
#SBATCH --get-user-env=L        	#Replicate login environment
        
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=efre8          	#Set the job name to "JobExample4"
#SBATCH --time=80:30:00          	#Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1               	#Request 1 task
#SBATCH --mem=9000MB              #Request 2560MB (2.5GB) per task
#SBATCH --output=log_efre8.%j  	  #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1             	#Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu          	#Request the GPU partition/queue

cd $SCRATCH/92EFRE_Public_Version/

ml GCCcore/10.2.0 Python/3.8.6
####**##module restore dl
source $SCRATCH/vfgenv/bin/activate

python GridS_main.py
