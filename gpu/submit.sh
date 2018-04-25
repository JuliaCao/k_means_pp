#!/bin/bash
#SBATCH -A cc3uv3p # 2017 XSEDE Applications of Parallel Computing Course Allocation
#SBATCH -J kmeans++
#SBATCH -o results.txt
#SBATCH -e results.txt
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:k80:1
#SBATCH -t 00:05:00
#SBATCH -N 1


./gpu -n 100000 -m 5 -k 20
