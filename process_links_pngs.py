from mpi4py import MPI
import pandas as pd
import glob
import os
import traceback

# os.environ["MPLCONFIGDIR"] = "/work/e723/e723/mzr123/matplotlib_config"

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_file(file):
    """
    Process a single file and return a DataFrame.
    """
    try:
        df = pd.read_csv(file, index_col=False)
        df.index.name = 'Index'
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    except pd.errors.EmptyDataError:
        print(f"Rank {rank}: Skipping empty file: {file}", flush=True)
        return None
    except Exception as e:
        print(f"Rank {rank}: Error processing file {file}: {e}", flush=True)
        return None

def plot_timestep(timestep, links_data, output_dir):
    """
    Generate a PNG for a given timestep.
    """
    try:
        plt.figure(figsize=(12, 8))
        m = Basemap(projection='merc', llcrnrlat=4, urcrnrlat=14, llcrnrlon=2, urcrnrlon=15, resolution='i')
        m.drawcountries()
        m.drawcoastlines()

        # Filter data for this timestep
        timestep_data = links_data[links_data['#time'] == timestep]

        # Create a colormap (e.g., blue to red)
        cmap = plt.colormaps['coolwarm']
        norm = colors.Normalize(vmin=0, vmax=1000)

        # Plot connections between locations
        for _, row in timestep_data.iterrows():
            norm_value = norm(min(row['cum_num_agents'], 1000))  # Cap value at 1000
            link_color = cmap(norm_value)
            capped_value = min(row['cum_num_agents'], 1000)
            linewidth = min(0.5 + 0.005 * capped_value, 3.0)

            m.plot(
                [row['start_lon'], row['end_lon']],
                [row['start_lat'], row['end_lat']],
                latlon=True,
                color=link_color,
                label='Routes',
                alpha=0.4,
                linewidth=linewidth
            )

        #plt.title(f"Agent Movements Links at Timestep {timestep}")
        output_path = os.path.join(output_dir, f"links_timestep_{timestep:03d}.png")
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error in plot_timestep for timestep {timestep}: {traceback.format_exc()}", flush=True)
        raise

if __name__ == "__main__":
    try:
        # Create output directory for PNGs, optionally
        output_dir = "."
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        comm.Barrier()

        # Master rank gathers the file list
        if rank == 0:
            file_list = sorted(glob.glob('links.out.*'), key=lambda x: int(x.split('.')[-1]))
        else:
            file_list = None

        # Broadcast file list to all ranks
        file_list = comm.bcast(file_list, root=0)

        # Distribute files across processors
        num_files = len(file_list)
        files_per_rank = num_files // size
        remainder = num_files % size

        if rank < remainder:
            start_idx = rank * (files_per_rank + 1)
            end_idx = start_idx + files_per_rank + 1
        else:
            start_idx = rank * files_per_rank + remainder
            end_idx = start_idx + files_per_rank

        assigned_files = file_list[start_idx:end_idx]
        print(f"Rank {rank}: Assigned {len(assigned_files)} files.", flush=True)

        # Process assigned files and generate PNGs
        for file in assigned_files:
            df = process_file(file)
            if df is not None:
                # Load the locations file for merging
                locations_df = pd.read_csv('input_csv/locations.csv')

                # Merge coordinates for start and end locations
                df = df.merge(
                    locations_df[['#name', 'latitude', 'longitude']].rename(
                        columns={'#name': 'start_location', 'latitude': 'start_lat', 'longitude': 'start_lon'}
                    ),
                    on='start_location',
                    how='left'
                )

                df = df.merge(
                    locations_df[['#name', 'latitude', 'longitude']].rename(
                        columns={'#name': 'end_location', 'latitude': 'end_lat', 'longitude': 'end_lon'}
                    ),
                    on='end_location',
                    how='left'
                )

                # Sort data by time
                df = df.sort_values(by=['#time'])

                # Generate PNGs for each timestep in this file
                for timestep in sorted(df['#time'].unique()):
                    plot_timestep(timestep, df, output_dir)
                    print(f"Rank {rank}: Generated PNG for timestep {timestep} from file {file}", flush=True)

        comm.Barrier()
        if rank == 0:
            print("All ranks completed PNG generation successfully.", flush=True)
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {traceback.format_exc()}", flush=True)


