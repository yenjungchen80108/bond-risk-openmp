#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=cuda_run.out
#SBATCH -A anakano_429

# Same node, same bonds.csv: OpenMP (CPU) first, then CUDA (GPU), so the comparison is fair.
echo "##### node: $(hostname) #####"
nvidia-smi --query-gpu=name --format=csv,noheader

for t in 1 8; do
  echo "##### CPU OpenMP, $t thread(s), schedule dynamic,1000 #####"
  OMP_SCHEDULE="dynamic,1000" OMP_NUM_THREADS=$t ./portfolio     # writes results.csv
done

for b in 128 256 512; do
  echo "##### GPU CUDA, $b threads per block #####"
  ./portfolio_gpu $b                                              # compares with results.csv
done
