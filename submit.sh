#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J LV_SINDy_Experiment 
# -- choose queue --
#BSUB -q hpc
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=8GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s194336@dtu.dk
# -- Output File --
# BSUB -o output_files/Output_%J.out
# -- Error File --
#BSUB -e error_files/Output_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 24:00 
# -- Number of cores requested -- 
#BSUB -n 20 
# -- Specify the distribution of the cores: on a single node --
# BSUB -R "span[hosts=1]"
# -- end of LSF options -- 

module load julia/1.11.5

#supported since julia-1.7 or so
export JULIA_NUM_THREADS=$LSB_DJOB_NUMPROC

julia scripts/LV/LV_SINDy.jl