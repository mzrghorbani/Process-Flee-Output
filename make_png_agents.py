import os
import pandas as pd
from mpi4py import MPI

# os.environ["MPLCONFIGDIR"] = "/work/e723/e723/mzr123/matplotlib_config"

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to plot a single timestep
def plot_timestep(timestep, agents_data):
    plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=4, urcrnrlat=14, llcrnrlon=2, urcrnrlon=15, resolution='i')
    m.drawcountries()
    m.drawcoastlines()

    # Original locations for timestep t
    original_locations = agents_data[agents_data['#time'] == timestep]
    m.scatter(
        original_locations['gps_x0'].values,
        original_locations['gps_y0'].values,
        latlon=True,
        marker='o',
        color='blue',
        label='Original Locations',
        s=50,
        alpha=0.7
    )

    # Current locations for timestep t
    current_locations = agents_data[agents_data['#time'] == timestep]
    m.scatter(
        current_locations['gps_x'].values,
        current_locations['gps_y'].values,
        latlon=True,
        marker='o',
        color='red',
        label='Current Locations',
        s=50,
        alpha=0.7
    )

    plt.title(f"Agent Movements at Timestep {timestep}")
    plt.legend(loc='lower left')

    # Save image for each timestep
    output_path = f"agents_timestep_{timestep:03d}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

if __name__ == "__main__":
    # Load the data
    agents_data = pd.read_csv('agents_data.csv', index_col=False)
    agents_data.index.name = 'Index'  # Explicitly name the index column

    # Set the simulation duration
    agents_timesteps = sorted(agents_data['#time'].unique())

    # Distribute timesteps across ranks
    timesteps_per_rank = len(agents_timesteps) // size
    remainder = len(agents_timesteps) % size
    start = rank * timesteps_per_rank + min(rank, remainder)
    end = start + timesteps_per_rank + (1 if rank < remainder else 0)
    assigned_timesteps = agents_timesteps[start:end]

    print(f"Rank {rank}: Assigned {len(assigned_timesteps)} timesteps.")

    # Process assigned timesteps
    for timestep in assigned_timesteps:
        try:
            output_file = plot_timestep(timestep, agents_data)
            print(f"Rank {rank}: Saved {output_file}")
        except Exception as e:
            print(f"Rank {rank}: Error processing timestep {timestep}: {e}")

    comm.Barrier()  # Ensure all ranks finish
    if rank == 0:
        print("Plots created successfully.")
