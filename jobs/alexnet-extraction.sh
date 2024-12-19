#!/bin/bash
#SBATCH --chdir /scratch/izar/remmal/CS-433
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --gres=gpu:2
#SBATCH --time 24:00:00
#SBATCH --mem 128G
#SBATCH --qos=gpu
#SBATCH --account cs433
#SBATCH --output ./logs/alexnet-%J.out
#SBATCH --mail-type=ALL

pyenv activate CS-433

jupyter nbconvert --to python notebooks/alexnet-extraction.ipynb

python notebooks/alexnet-extraction.py

rm -f notebooks/alexnet-extraction.py
