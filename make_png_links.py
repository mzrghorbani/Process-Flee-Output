import os
import pandas as pd
from mpi4py import MPI

os.environ["MPLCONFIGDIR"] = "/home/mghorbani/workspace/python/matplotlib_config"

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Function to plot a single timestep
def plot_timestep(timestep, links_data, output_dir):
    try:
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

        plt.title(f"Agent Movements Links at Timestep {timestep}")
        output_path = os.path.join(output_dir, f"links_timestep_{timestep:03d}.png")
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error in plot_timestep for timestep {timestep}: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        # Rank 0 loads the data and distributes it
        if rank == 0:
            agents_data = pd.read_csv('links_data.csv')
            agents_timesteps = sorted(agents_data['#time'].unique())

            # Distribute timesteps across ranks
            timesteps_per_rank = len(agents_timesteps) // size
            remainder = len(agents_timesteps) % size
            start = rank * timesteps_per_rank + min(rank, remainder)
            end = start + timesteps_per_rank + (1 if rank < remainder else 0)

            assigned_timesteps_list = []
            for r in range(size):
                r_start = r * timesteps_per_rank + min(r, remainder)
                r_end = r_start + timesteps_per_rank + (1 if r < remainder else 0)
                assigned_timesteps_list.append(agents_timesteps[r_start:r_end])

            # Create a list of dataframes for each rank
            split_data = [
                agents_data[agents_data['#time'].isin(assigned_timesteps_list[r])] for r in range(size)
            ]
        else:
            split_data = None

        # Scatter the data to each rank
        local_data = comm.scatter(split_data, root=0)

        # Process assigned timesteps
        for timestep in sorted(local_data['#time'].unique()):
            plot_timestep(timestep, local_data, output_dir)

        comm.Barrier()
        if rank == 0:
            print("All ranks completed successfully.")
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {traceback.format_exc()}")
