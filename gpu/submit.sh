#!/bin/bash
#SBATCH -A cc3uv3p # 2017 XSEDE Applications of Parallel Computing Course Allocation
#SBATCH -J auto-particle-gpu
#SBATCH -o auto-particle-gpu.stdout
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:k80:1
#SBATCH -t 00:20:00
#SBATCH -N 1
