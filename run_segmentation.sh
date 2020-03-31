#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./logs/tjob.out.%j
#SBATCH -e ./logs/tjob.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name :
#SBATCH -J guppy_seg
# Queue (Partition):
#SBATCH --partition=n0128
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
# Enable Hyperthreading:
# Request 180 GB of main memory per node in units of MB:
#SBATCH --mem=185000
#SBATCH --mail-type=none
#SBATCH --mail-user=ietheredge@orn.mpg.de
# Wall clock limit:
#SBATCH --time=24:00:00

### enable over-subscription of physical cores by MPI ranks
### launch one MPI process per physical core
### request fat memory nodes, if necessary
export PSM2_MULTI_EP=0
ulimit -n 50000

# Run the program:
module purge
module load gcc impi anaconda/3  mpi4py

srun python3 ./SegmentationMPI.py > logs/ts.out
