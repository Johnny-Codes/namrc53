import json
import os
from collections import defaultdict


def split_data_by_user(
    data_directory_prefix, num_folders, output_directory="user_breakdown"
):
    """
    Reads all metadata, groups frames by user, and saves the data into separate
    JSON files for each user.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process.
        output_directory (str): The name of the folder to save the output files.
    """

    # --- Data Aggregation Dictionary ---
    # The structure will be: {user_id: [frame_data_1, frame_data_2, ...]}
    user_data = defaultdict(list)

    # --- Generate File Paths ---
    training_file_paths = []
    for i in range(num_folders):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(data_directory_prefix, folder_name))

    # --- Main Processing Loop ---
    print("Reading all metadata files...")
    for current_dir_path in training_file_paths:
        metadata_path = os.path.join(current_dir_path, "metadata.jsonl")

        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    frame_data = json.loads(line.strip())
                    user_id = frame_data.get("user_label", "unknown_user")
                    # Append the entire frame's data to the list for that user
                    user_data[user_id].append(frame_data)
                except json.JSONDecodeError:
                    continue

    # --- Create Output Directory and Save Files ---
    os.makedirs(output_directory, exist_ok=True)
    print(f"\nFound data for {len(user_data)} unique users.")

    for user_id, frames in user_data.items():
        output_filename = os.path.join(output_directory, f"user_{user_id}_data.json")

        print(
            f"Saving data for User ID {user_id} ({len(frames)} frames) to {output_filename}..."
        )

        # Sort frames by datetime for each user to ensure chronological order
        frames.sort(key=lambda x: x.get("datetime", ""))

        with open(output_filename, "w") as f:
            json.dump(frames, f, indent=4)

    print("\nProcessing complete.")
    print(f"Check the '{output_directory}' directory for the output files.")


# --- Run the Script ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    # The range function is exclusive, so 34 will process folders 0 to 33.
    NUM_FOLDERS = 34
    split_data_by_user(DATA_PREFIX, NUM_FOLDERS)
