from mpi4py import MPI
import pandas as pd
import glob
import re

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_file(file):
    try:
        # Read the file
        df = pd.read_csv(file, index_col=False)
        df.index.name = 'Index'

        # Early filtering: Keep only rows where 'is_travelling' is True
        # df = df[df['is_travelling'] == True]

        # Drop unnecessary columns and keep only the essentials
        df = df[['#time', 'original_location', 'gps_x', 'gps_y', 'current_location']]

        # Drop rows with NaN values
        df = df.dropna()

        # Optionally downsample rows
        df = df.iloc[::2, :]  # Take every second row

        return df
    except pd.errors.EmptyDataError:
        print(f"Rank {rank}: Skipping empty file: {file}")
        return None
    except Exception as e:
        print(f"Rank {rank}: Error processing file {file}: {e}")
        return None

def clean_location(x):
    if isinstance(x, str):
        return re.sub(r'L:.*?:', '', x)
    print(f"Skipping invalid value: {x}")
    return None

if __name__ == "__main__":
    # Master rank gathers the file list
    if rank == 0:
        file_list = sorted(glob.glob('agents.out.*'), key=lambda x: int(x.split('.')[-1]))
    else:
        file_list = None

    # Broadcast file list to all ranks
    file_list = comm.bcast(file_list, root=0)

    # Divide tasks among ranks
    tasks_per_rank = len(file_list) // size
    start_idx = rank * tasks_per_rank
    end_idx = start_idx + tasks_per_rank if rank != size - 1 else len(file_list)

    assigned_files = file_list[start_idx:end_idx]
    print(f"Rank {rank} processing {len(assigned_files)} files.")

    # Process assigned files
    valid_dataframes = []
    for file in assigned_files:
        df = process_file(file)
        if df is not None:
            valid_dataframes.append(df)

    # Concatenate local DataFrames
    local_concatenated_df = pd.concat(valid_dataframes, ignore_index=True) if valid_dataframes else pd.DataFrame()

    # Guard for empty DataFrames
    if local_concatenated_df is None or local_concatenated_df.empty:
        local_concatenated_df = None

    # Gather all DataFrames at rank 0
    all_dataframes = comm.gather(local_concatenated_df, root=0)

    if rank == 0:
        # Combine all DataFrames at rank 0
        concatenated_df = pd.concat(all_dataframes, ignore_index=True)

        # Remove rows with any NaN values
        concatenated_df = concatenated_df.dropna()

        # Clean `current_location` column
        concatenated_df['current_location_clean'] = concatenated_df['current_location'].apply(clean_location)

        locations_df = pd.read_csv('input_csv/locations.csv')

        # Merge original location coordinates into agents DataFrame
        concatenated_df = concatenated_df.merge(
            locations_df[['#name', 'latitude', 'longitude']],
            left_on='original_location',
            right_on='#name',
            how='left'
        )

        # Rename merged columns for clarity
        concatenated_df.rename(columns={'latitude': 'gps_y0', 'longitude': 'gps_x0'}, inplace=True)

        # Drop the redundant '#name' column from the merge
        concatenated_df.drop(columns=['#name'], inplace=True)
        
        # Group by `#time` and sort within each group
        sorted_df = concatenated_df.sort_values(by=['#time'], ascending=True).reset_index(drop=True)

        # Save the processed DataFrame
        sorted_df.to_csv('agents_data.csv', index=False)
        print("Processed agents data saved to 'agents_data.csv'.")
        