#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:59
#SBATCH --account=e723-brunel
#SBATCH --output=JobID-%j.output
#SBATCH --error=JobID-%j.error
#SBATCH --partition=highmem
#SBATCH --qos=highmem

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Log information about SLURM configuration
echo "SLURM Job Configuration:"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Total Tasks: $SLURM_NTASKS"

# Set the number of threads to 1
export OMP_NUM_THREADS=1

module load cray-python

export PYTHONUSERBASE=/work/e723/e723/mzr123/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH

# Navigate to flee output directory
OUTPUT_DIR="/work/e723/e723/mzr123/FabSim/results/nigeria2024_archer2_256"

if [ -d "$OUTPUT_DIR" ]; then
    cd "$OUTPUT_DIR"
else
    echo "Error: Directory $OUTPUT_DIR does not exist. Exiting..."
    exit 1
fi

# Log start of the process
echo "Starting process at $(date)..."
echo "Using $SLURM_CPUS_ON_NODE CPUs on node $SLURMD_NODENAME"

# Exit immediately on any error
set -e

# Check for errors during agents processing
echo "Starting agents processing..."
srun --distribution=block:block --hint=nomultithread python3 process_agents_pngs.py > process_agents_pngs.out 2> process_agents_pngs.err
if [ $? -ne 0 ]; then
    echo "Error: Agents processing failed. Check process_agents_pngs.err for details. Exiting..."
    exit 1
fi
echo "Agents processing completed successfully!"

# Check for errors during links processing
echo "Starting links processing..."
srun --distribution=block:block --hint=nomultithread python3 process_links_pngs.py > process_links_pngs.out 2> process_links_pngs.err
if [ $? -ne 0 ]; then
    echo "Error: Links processing failed. Check process_links_pngs.err for details. Exiting..."
    exit 1
fi
echo "Links processing completed successfully!"

echo "All processes completed successfully!"

