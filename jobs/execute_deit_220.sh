#!/bin/bash
#SBATCH --chdir /home/seo/vision/examples
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --mem 256G
#SBATCH --qos=gpu
#SBATCH --account cs433
#SBATCH --mail-user=cs433@aritravo.ma
#SBATCH --mail-type=ALL
source ~/miniconda3/bin/activate
conda activate myvenv2
python deit_feature_extraction_220.py