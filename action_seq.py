import json
import os
from collections import defaultdict
from datetime import datetime

# --- Optional: Use pandas for a nicer output table ---
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print(
        "Warning: pandas library not found. `pip install pandas` for a prettier output table."
    )


def analyze_action_transitions(
    data_directory_prefix, num_folders, transition_threshold_seconds=60.0
):
    """
    Analyzes the dataset to find which actions frequently occur back-to-back.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
        transition_threshold_seconds (float): The maximum time in seconds between two
                                              actions to be considered a direct transition.
    """

    # --- Step 1: Aggregate all frame data from metadata files ---
    all_frames = []
    training_file_paths = []
    for i in range(num_folders):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(data_directory_prefix, folder_name))

    print("Reading all metadata files...")
    for current_dir_path in training_file_paths:
        metadata_path = os.path.join(current_dir_path, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    all_frames.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    # --- Step 2: Group frames by action_number and find start/end times ---
    print("Grouping frames by action sequence...")
    action_sequences = defaultdict(
        lambda: {"start_time": None, "end_time": None, "label": "", "user": None}
    )

    for frame in all_frames:
        action_num = frame.get("action_number")
        if action_num is None:
            continue

        # The datetime format can vary, so we handle potential parsing errors
        try:
            timestamp = datetime.fromisoformat(frame["datetime"])
        except (ValueError, KeyError):
            continue

        seq = action_sequences[action_num]

        # Initialize or update sequence data
        if seq["start_time"] is None or timestamp < seq["start_time"]:
            seq["start_time"] = timestamp
            seq["label"] = frame.get("action_label", "unknown")
            seq["user"] = frame.get("user_label", "unknown")

        if seq["end_time"] is None or timestamp > seq["end_time"]:
            seq["end_time"] = timestamp

    # --- Step 3: Sort all sequences chronologically ---
    print("Sorting sequences by timestamp...")
    sorted_sequences = sorted(
        action_sequences.items(), key=lambda item: item[1]["start_time"]
    )

    # --- Step 4: Analyze transitions between consecutive actions ---
    print("Analyzing action transitions...")
    transition_counts = defaultdict(int)

    for i in range(len(sorted_sequences) - 1):
        current_seq_num, current_action = sorted_sequences[i]
        next_seq_num, next_action = sorted_sequences[i + 1]

        # Ensure we have valid timestamps and the user is the same
        if (
            current_action["end_time"]
            and next_action["start_time"]
            and current_action["user"] == next_action["user"]
        ):
            time_gap = (
                next_action["start_time"] - current_action["end_time"]
            ).total_seconds()

            # Check if the gap is within our threshold for a "back-to-back" action
            if 0 <= time_gap <= transition_threshold_seconds:
                transition = (current_action["label"], next_action["label"])
                transition_counts[transition] += 1

    # --- Step 5: Report the findings ---
    print("\n" + "=" * 60)
    print(" " * 15 + "ACTION TRANSITION REPORT")
    print("=" * 60)
    print(f"Found {len(transition_counts)} unique back-to-back action transitions.")
    print(
        f"(Threshold for 'back-to-back' is <= {transition_threshold_seconds} seconds between actions by the same user)\n"
    )

    if not transition_counts:
        print("No back-to-back transitions found with the current threshold.")
        return

    # Prepare data for tabulation
    report_data = []
    for (action_from, action_to), count in sorted(
        transition_counts.items(), key=lambda item: item[1], reverse=True
    ):
        report_data.append(
            {"From Action": action_from, "To Action": action_to, "Count": count}
        )

    # Display the report
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))
    else:
        print(f"{'From Action':<25} -> {'To Action':<25} | {'Count':>10}")
        print("-" * 70)
        for row in report_data:
            print(
                f"{row['From Action']:<25} -> {row['To Action']:<25} | {row['Count']:>10}"
            )

    print("=" * 60)


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34  # Process train-000 to train-032
    analyze_action_transitions(DATA_PREFIX, NUM_FOLDERS)
