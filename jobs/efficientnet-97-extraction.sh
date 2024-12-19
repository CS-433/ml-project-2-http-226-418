#!/bin/bash
#SBATCH --chdir /scratch/izar/remmal/CS-433
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --mem 256G
#SBATCH --qos=gpu
#SBATCH --account cs433
#SBATCH --output ./logs/efficientnet-97-%J.out
#SBATCH --mail-user=slurm-jobs@groupes.epfl.ch
#SBATCH --mail-type=ALL

pyenv activate CS-433

jupyter nbconvert --to python notebooks/efficientnet-97-extraction.ipynb

python notebooks/efficientnet-97-extraction.py

rm -f notebooks/efficientnet-97-extraction.py
