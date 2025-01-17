# Required Packages

Below is a list of required Python packages with their versions:

- Python Environment (recommended)
- Python Version: Python 3.12.3
- basemap==1.4.1
- matplotlib==3.8.4
- moviepy==1.0.3
- mpi4py==4.0.1
- pandas==2.2.3

## Overview of the Workflow

### After running the Flee simulation

- **Output Files**: The simulation generates `agents.out.*` and `links.out.*` files, which record agents' movements and routes between locations over time.
- **Number of Logs**: The number of output files corresponds to the number of CPU cores allocated to the
simulation. For example:
- **128 CPU cores produce**:
  - 128 agents.out.* files (e.g., agents.out.0, agents.out.1, ..., agents.out.127).
  - 128 links.out.* files (e.g., links.out.0, links.out.1, ..., links.out.127).

Copy the content of this repository in HPC (e.g., ARCHER2) simulation output directory (e.g., nigeria_archer2_128), and execute the `run.slurm` script:

```bash
sbatch run.slurm
```

Or, continue with the workflow detailed below:

## MPI Parallelization

MPI parallelization ensures the workflow scales effectively with the number of CPU cores available, enabling faster processing for large datasets.

Here's how MPI parallelization is implemented:

**File Distribution**:

- The `agents.out.*` and `links.out.*` files are distributed among the available MPI ranks (processes) to ensure that each rank processes a unique subset of files. For example, if there are 128 files and 4 ranks, each rank will process 32 files.

**Parallel Processing**:

- Each rank processes its assigned files independently, extracting the necessary data and generating PNGs incrementally. This reduces the overall execution time as multiple ranks work simultaneously.

## Steps to Process and Visualize the Data

### Step 1: Process Agents Logs and Create PNG Files

**Script**: process_agents_and_png.py
**Description**: This script processes the agents.out.* files incrementally to extract data for each time step and generate PNG visualizations of agents' movements.

**Execution**: Add the following to your SLURM job script (e.g., run.slurm)

```bash
srun --distribution=block:block --hint=nomultithread python3 process_agents_pngs.py
```

**Output**:
Processed PNGs stored in current directory (e.g., agents_timestep_000.png, agents_timestep_001.png, etc.).

### Step 2: Process Links Logs and Create PNG Files

**Script**: process_links_and_png.py
**Description**: This script processes the links.out.* files incrementally to extract data for each time step and generate PNG visualizations of agent routes (links) between locations.

**Execution**: Add the following to your SLURM job script (e.g., run.slurm):

```bash
srun --distribution=block:block --hint=nomultithread python3 process_links_pngs.py
```

**Output**:
Processed PNGs stored in current directory (e.g., links_timestep_000.png, links_timestep_001.png, etc.).

### SLURM Job Submission

The `run.slurm` script is updated with the necessary steps to create MPI jobs and submit them:

```bash
sbatch run.slurm
```

Note: The SLURM jobs will be queued until processed. This could take a long time, therefore, we have create multiprocessing version of MPI implementations which can be executed as Python script.

For creating agents PNGs:

```bash
python3 process_agents_pngs_mp.py
```

For creating links PNGs:

```bash
python3 process_links_pngs_mp.py
```

### Step 3: Generate Videos from PNGs

#### 1. Create a Video from Agents PNG Files

**Script**: make_video_agents.py
**Description**: This script combines the agent PNG files into a video file.

**Execution**: Add the following to your SLURM job script (run.slurm):

```bash
python3 make_video_agents.py
```

**Output**:
A video file named agents_video.mp4 showing the agents' movements over time.

#### 2. Create a Video from Links PNG Files

**Script**: make_video_links.py
**Description**: This script combines the link PNG files into a video file.

**Execution**: Add the following to your SLURM job script (run.slurm):

```bash
python3 make_video_links.py
```

**Output**:
A video file named links_video.mp4 showing the routes between locations over time.

### Step 4: Overlay Agents and Links Videos

**Utility**: *ffmpeg*

For Ubuntu installation:

```bash
sudo apt update && sudo apt install ffmpeg
```

For Mac installation:

```zshrc
brew install ffmpeg
```

Or, optionally download and extract binary by visiting the official FFmpeg download page, or for Linux, use a static build from johnvansickle.com.

```bash
tar xvf ffmpeg-release-*.tar.xz
```

Add ffmpeg to your PATH:

```bash
export PATH=$PATH:/path/to/ffmpeg/bin
```

**Description**: The ffmpeg tool overlays the two videos (agents_video.mp4 and links_video.mp4) to create a combined video showing both agents and links simultaneously.

**Execution**: Run the following command in your terminal:

#### Resize videos

```bash
ffmpeg -i agent_movements_animation.mp4 -vf "scale=1280:720" agents_resized.mp4
ffmpeg -i link_movements_animation.mp4 -vf "scale=1280:720" links_resized.mp4
```

#### Overlay videos

```bash
ffmpeg -i agents_video.mp4 -i links_video.mp4 -filter_complex \
"[0:v]format=rgba,colorchannelmixer=aa=0.5[bg]; \
 [1:v]format=rgba,colorchannelmixer=aa=0.5[fg]; \
 [bg][fg]overlay=0:0" combined_video.mp4
```

**Output**:
A combined video file named combined_video.mp4.

### Known Issues and Fixes

**Error When Making Videos**:

**Issue**: When generating videos using moviepy, the following error may occur:

```bash
OSError: broken data stream when reading image file
```

**Cause**: This occurs if some of the PNG files are incomplete or corrupted during creation.

**Fix**: Add the following lines after importing packages in your scripts:

```python
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

### Basemap Installation Issues

**Issue**: Installing basemap can sometimes fail because it's not available on PyPI by default.

**Fix**: Use pip to install basemap precompile wheels:

```bash
pip install https://github.com/matplotlib/basemap/archive/master.zip
```

### Out-of-Memory Issues for Large Datasets

**Issue**: Large datasets can cause memory issues during processing or video generation.

**Fix**: Ensure incremental file processing is enabled in your scripts, as implemented in the provided codes.
