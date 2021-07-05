#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-04:00 # Runtime in D-HH:MM
#SBATCH -p shared # Partition to submit to
#SBATCH --mem=50000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outputs/hostname_%j.out # File to which STDOUT will be written
#SBATCH -e outputs/hostname_%j.err # File to which STDERR will be written

hostname

module load Anaconda3/5.0.1-fasrc02
source activate root3.6
# export R_LIBS_USER=$HOME/apps/R:$R_LIBS_USER
# Rscript feicheng.R ${RHO} ${VARY} ${VAR} ${PARAM1}

python main.py ${PARAM1} ${RHO} ${DIM} ${NT} ${COR} ${TYPE}
