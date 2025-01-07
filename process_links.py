from mpi4py import MPI
import pandas as pd
import glob

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_file(file):
    try:
        df = pd.read_csv(file, index_col=False)
        df.index.name = 'Index'
        return df
    except pd.errors.EmptyDataError:
        print(f"Rank {rank}: Skipping empty file: {file}")
        return None
    except Exception as e:
        print(f"Rank {rank}: Error processing file {file}: {e}")
        return None

if __name__ == "__main__":
    # Gather all links log files
    file_list = sorted(glob.glob('links.out.*'), key=lambda x: int(x.split('.')[-1]))

    # Distribute files across processors
    files_per_rank = len(file_list) // size
    remainder = len(file_list) % size
    start = rank * files_per_rank + min(rank, remainder)
    end = start + files_per_rank + (1 if rank < remainder else 0)
    assigned_files = file_list[start:end]

    print(f"Rank {rank}: Assigned {len(assigned_files)} files.")

    # Process assigned files
    valid_dataframes = []
    for file in assigned_files:
        df = process_file(file)
        if df is not None:
            valid_dataframes.append(df)

    # Combine results locally on each rank
    if valid_dataframes:
        concatenated_df = pd.concat(valid_dataframes, ignore_index=True)
    else:
        concatenated_df = pd.DataFrame()  # Empty DataFrame for consistency

    # Gather data on rank 0
    gathered_data = comm.gather(concatenated_df, root=0)

    if rank == 0:
        # Combine all gathered DataFrames
        combined_df = pd.concat(gathered_data, ignore_index=True)
        print(f"Rank 0: Total combined rows: {len(combined_df)}")

        # Sort data by time for easier plotting
        combined_df = combined_df.sort_values(by=['#time'])

        # Drop rows with NaN values
        combined_df = combined_df.dropna()

        # Load the locations file
        locations_df = pd.read_csv('input_csv/locations.csv')

        # Merge coordinates for start and end locations
        combined_df = combined_df.merge(
            locations_df[['#name', 'latitude', 'longitude']].rename(
                columns={'#name': 'start_location', 'latitude': 'start_lat', 'longitude': 'start_lon'}
            ),
            on='start_location',
            how='left'
        )

        combined_df = combined_df.merge(
            locations_df[['#name', 'latitude', 'longitude']].rename(
                columns={'#name': 'end_location', 'latitude': 'end_lat', 'longitude': 'end_lon'}
            ),
            on='end_location',
            how='left'
        )

        # Save the final combined DataFrame
        combined_df.to_csv('links_data.csv', index=False)
        print("Processed links data saved to 'links_data.csv'.")
