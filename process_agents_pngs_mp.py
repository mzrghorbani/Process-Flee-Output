import pandas as pd
import glob
import os
import re
import traceback
from multiprocessing import Pool, cpu_count

# os.environ["MPLCONFIGDIR"] = "/work/e723/e723/mzr123/matplotlib_config"

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def process_file(file):
    """
    Function to process a single file and return a DataFrame.
    """
    try:
        # Read the file
        df = pd.read_csv(file, index_col=False)
        df.index.name = 'Index'

        # Drop unnecessary columns and keep only the essentials
        df = df[['#time', 'original_location', 'gps_x', 'gps_y', 'current_location']]

        # Drop rows with NaN values
        df = df.dropna()

        # Optionally downsample rows
        df = df.iloc[::16, :]  # Take every 16th row

        return df
    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {file}", flush=True)
        return None
    except Exception as e:
        print(f"Error processing file {file}: {e}", flush=True)
        return None


def clean_location(x):
    """
    Function to clean location strings.
    """
    if isinstance(x, str):
        return re.sub(r'L:.*?:', '', x)
    print(f"Skipping invalid value: {x}", flush=True)
    return None


def plot_timestep(timestep, agents_data, output_dir):
    """
    Function to generate a PNG for a given timestep.
    """
    try:
        plt.figure(figsize=(12, 8))
        m = Basemap(projection='merc', llcrnrlat=4, urcrnrlat=14, llcrnrlon=2, urcrnrlon=15, resolution='i')
        m.drawcountries()
        m.drawcoastlines()

        original_locations = agents_data[agents_data['#time'] == timestep]
        m.scatter(
            original_locations['gps_x0'].values,
            original_locations['gps_y0'].values,
            latlon=True,
            marker='*',
            color='red',
            label='Original Locations',
            s=90,
            alpha=0.7,
            zorder=3
        )

        current_locations = agents_data[agents_data['#time'] == timestep]
        m.scatter(
            current_locations['gps_y'].values,
            current_locations['gps_x'].values,
            latlon=True,
            marker='o',
            color='green',
            label='Current Locations',
            s=50,
            alpha=0.3,
            zorder=2
        )

        plt.legend(loc='lower left')
        output_path = os.path.join(output_dir, f"agents_timestep_{timestep:03d}.png")
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error in plot_timestep for timestep {timestep}: {traceback.format_exc()}")
        raise


def process_and_plot(file):
    """
    Function to process a file and generate PNGs for all timesteps in it.
    """
    try:
        df = process_file(file)
        if df is not None:
            # Clean `current_location` column
            df['current_location_clean'] = df['current_location'].apply(clean_location)

            # Add original location coordinates
            locations_df = pd.read_csv('input_csv/locations.csv')
            df = df.merge(
                locations_df[['#name', 'latitude', 'longitude']].rename(
                    columns={'#name': 'original_location', 'latitude': 'gps_y0', 'longitude': 'gps_x0'}
                ),
                on='original_location',
                how='left'
            )

            # Generate PNGs for each timestep in this file
            for timestep in sorted(df['#time'].unique()):
                plot_timestep(timestep, df, output_dir)
                print(f"Generated PNG for timestep {timestep} from file {file}", flush=True)
    except Exception as e:
        print(f"Error in processing file {file}: {traceback.format_exc()}")


if __name__ == "__main__":
    try:
        # Create output directory for PNGs
        output_dir = "./output_agents_pngs"
        os.makedirs(output_dir, exist_ok=True)

        # Gather all log files
        file_list = sorted(glob.glob('agents.out.*'), key=lambda x: int(x.split('.')[-1]))

        # Use multiprocessing to parallelize file processing
        num_workers = min(cpu_count(), len(file_list))  # Use available CPUs or the number of files, whichever is smaller
        
        print(f"Found {len(file_list)} files and {num_workers} workers to process.")
            
        with Pool(processes=num_workers) as pool:
            try:
                # Map the processing function across files
                pool.map(process_and_plot, file_list)
            except Exception as e:
                print(f"Error occurred during multiprocessing: {e}", flush=True)
            finally:
                # Ensure the pool is closed and joined properly
                pool.close()
                pool.join()

        print("All files processed and PNGs generated successfully.")
    except Exception as e:
        print(f"Error occurred: {traceback.format_exc()}")
