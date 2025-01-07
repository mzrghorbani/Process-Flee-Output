import os
import pandas as pd
from mpi4py import MPI

# os.environ["MPLCONFIGDIR"] = "/work/e723/e723/mzr123/matplotlib_config"

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to plot a single timestep
def plot_timestep(timestep, links_data):
    plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=4, urcrnrlat=14, llcrnrlon=2, urcrnrlon=15, resolution='i')
    m.drawcountries()
    m.drawcoastlines()

    # Filter data for this timestep
    timestep_data = links_data[links_data['#time'] == timestep]
    
    # Create a colormap (e.g., blue to red)
    cmap = plt.colormaps['coolwarm']  # Updated method to get the colormap
    norm = colors.Normalize(vmin=0, vmax=1000)  # Normalise cumulative agents to a range [0, 1000]

    # Plot connections between locations
    for _, row in timestep_data.iterrows():
        # Get the normalised value for the colour
        norm_value = norm(min(row['cum_num_agents'], 1000))  # Cap the value at 1000
        link_color = cmap(norm_value)  # Map the normalised value to a colour
        capped_value = min(row['cum_num_agents'], 1000)  # Cap cumulative agents at 1000
        linewidth = min(0.5 + 0.005 * capped_value, 4.0)  # Scale and cap linewidth

        # Draw the link
        m.plot(
            [row['start_lon'], row['end_lon']],
            [row['start_lat'], row['end_lat']],
            latlon=True,
            color=link_color,
            alpha=0.8,
            linewidth=linewidth 
        )

    plt.title(f"Agent Movements Between Locations at Timestep {timestep}")
    output_path = f"links_timestep_{timestep:03d}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path

if __name__ == "__main__":
    # Load the data
    links_data = pd.read_csv('links_data.csv')

    # Unique timesteps in the data
    links_timesteps = sorted(links_data['#time'].unique())

    # Distribute timesteps across ranks
    timesteps_per_rank = len(links_timesteps) // size
    remainder = len(links_timesteps) % size
    start = rank * timesteps_per_rank + min(rank, remainder)
    end = start + timesteps_per_rank + (1 if rank < remainder else 0)
    assigned_timesteps = links_timesteps[start:end]

    print(f"Rank {rank}: Assigned {len(assigned_timesteps)} timesteps.")

    # Process assigned timesteps
    for timestep in assigned_timesteps:
        try:
            output_file = plot_timestep(timestep, links_data)
            print(f"Rank {rank}: Saved {output_file}")
        except Exception as e:
            print(f"Rank {rank}: Error processing timestep {timestep}: {e}")

    comm.Barrier()  # Ensure all ranks finish
    if rank == 0:
        print("Plots created successfully.")
