#!/bin/bash
#SBATCH --nodes=4
#SBATCH --tasks-per-node=256
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

module load cray-python

export PYTHONUSERBASE=/work/e723/e723/mzr123/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
export SLURM_CPU_FREQ_REQ=2250000
export ENV_DIR=/work/e723/e723/mzr123/VirtualEnv

# Navigate to flee output directory
cd /work/e723/e723/mzr123/FabSim/results/nigeria2024_archer2_128

# Log start of the process
echo "Starting process at $(date)..."
echo "Using $SLURM_CPUS_ON_NODE CPUs on node $SLURMD_NODENAME"

#!/bin/bash

set -e  # Exit on any error

echo "Starting agents processing..."
srun python3 process_agents_mpi2.py > process_agents.out 2> process_agents.err
if [ ! -f "agents_data.csv" ]; then
    echo "Error: agents_data.csv not found. Stopping execution."
    exit 1
fi
echo "Agents processing completed!"

echo "Generating PNG files for agents..."
srun python3 make_png_agents_mpi.py > make_png_agents.out 2> make_png_agents.err
echo "PNG files for agents generated!"

echo "Creating video from agents PNG files..."
python3 make_video_agents_mpi.py > make_video_agents.out 2> make_video_agents.err
echo "Video for agents created!"

echo "Starting links processing..."
srun python3 process_links_mpi2.py > process_links.out 2> process_links.err
if [ ! -f "links_data.csv" ]; then
    echo "Error: links_data.csv not found. Stopping execution."
    exit 1
fi
echo "Links processing completed!"

echo "Generating PNG files for links..."
srun python3 make_png_links_mpi.py > make_png_links.out 2> make_png_links.err
echo "PNG files for links generated!"

echo "Creating video from links PNG files..."
python3 make_video_links.py > make_video_links.out 2> make_video_links.err
echo "Video for links created!"

echo "All processes completed successfully!"
