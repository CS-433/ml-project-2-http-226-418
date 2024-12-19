#!/bin/bash
#SBATCH --chdir /scratch/izar/seo
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --gres=gpu:2
#SBATCH --time 24:00:00
#SBATCH --mem 32G
#SBATCH --qos=gpu
#SBATCH --account cs433

#SBATCH --mail-type=ALL
source ~/miniconda3/bin/activate
conda activate edvenv
python 50_resnet_ed.py