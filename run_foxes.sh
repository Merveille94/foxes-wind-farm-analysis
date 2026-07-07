BATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --partition=rosa.p
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=all.out

module load hpc-env/12.2 hpc-env/13.1 Anaconda3

conda activate wind
~/.conda/envs/wind/bin/python -c "import foxes"

~/.conda/envs/wind/bin/python run_task4.py

