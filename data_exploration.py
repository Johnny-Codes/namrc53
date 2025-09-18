import json
import os
from collections import defaultdict

# --- Optional: Use pandas for a nicer output table ---
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print(
        "Warning: pandas library not found. `pip install pandas` for a prettier output table."
    )


def print_latex_table(report_data):
    """
    Prints the report data as a LaTeX tabular table.

    Args:
        report_data (list): A list of dictionaries containing the report data.
    """
    if not report_data:
        return

    print("\n--- LaTeX Tabular Output ---")
    # Define table structure with 6 right-aligned columns
    print("\\begin{tabular}{lrrrrr}")
    print("\\hline")
    # Header row
    header = list(report_data[0].keys())
    # Escape underscores in header for LaTeX
    safe_header = [h.replace("_", "\\_") for h in header]
    print(
        f"\\textbf{{{safe_header[0]}}} & \\textbf{{{safe_header[1]}}} & \\textbf{{{safe_header[2]}}} & \\textbf{{{safe_header[3]}}} & \\textbf{{{safe_header[4]}}} & \\textbf{{{safe_header[5]}}} \\\\"
    )
    print("\\hline")

    # Data rows
    for row in report_data:
        # Add a horizontal line before the 'Total' row
        if row["Action Label"] == "Total":
            print("\\hline")

        # Escape special LaTeX characters in the data
        action_label = str(row["Action Label"]).replace("_", "\\_")
        total_frames = row["Total Frames"]
        frame_percent = str(row["Frame %"]).replace("%", "\\%")
        inner_frames = str(row["Inner Frames"]).replace("%", "\\%")
        outer_frames = str(row["Outer Frames"]).replace("%", "\\%")
        unique_sequences = row["Unique Sequences"]

        # Make the 'Total' row bold
        if row["Action Label"] == "Total":
            print(
                f"\\textbf{{{action_label}}} & \\textbf{{{total_frames}}} & \\textbf{{{frame_percent}}} & & & \\textbf{{{unique_sequences}}} \\\\"
            )
        else:
            print(
                f"{action_label} & {total_frames} & {frame_percent} & {inner_frames} & {outer_frames} & {unique_sequences} \\\\"
            )

    print("\\hline")
    print("\\end{tabular}")


def analyze_dataset(data_directory_prefix, num_folders):
    """
    Analyzes the dataset by iterating through all metadata files,
    collecting statistics on actions, views, and frames.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
    """

    # --- Data Aggregation Variables ---
    total_frames = 0
    # A global set to track truly unique action sequences across all folders
    all_unique_sequences = set()
    # defaultdict simplifies counting: defaultdict(int) initializes new keys with 0
    action_stats = defaultdict(
        lambda: {
            "total_frames": 0,
            "inner_frames": 0,
            "outer_frames": 0,
            "action_sequences": set(),
        }
    )
    user_stats = defaultdict(int)

    # --- Generate File Paths ---
    training_file_paths = []
    for i in range(num_folders):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(data_directory_prefix, folder_name))

    # --- Main Processing Loop ---
    for current_dir_path in training_file_paths:
        metadata_path = os.path.join(current_dir_path, "metadata.jsonl")

        if not os.path.exists(metadata_path):
            continue  # Skip if metadata file doesn't exist

        with open(metadata_path, "r") as f:
            for line in f:
                total_frames += 1
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

                # --- Extract and Tally Data ---
                action = data.get("action_label", "unknown")
                user_id = data.get("user_label", "unknown")
                action_num = data.get("action_number", None)  # Get None if missing
                file_name = data.get("file_name", "")

                # Tally frames per user
                user_stats[user_id] += 1

                # Tally action stats
                stats = action_stats[action]
                stats["total_frames"] += 1

                # Only count sequence if action_number is valid
                if action_num is not None:
                    # The action_number is the globally unique key, as per the paper
                    stats["action_sequences"].add(action_num)
                    all_unique_sequences.add(action_num)

                # Tally view based on file path
                if "inner_depths" in file_name:
                    stats["inner_frames"] += 1
                elif "outer_depths" in file_name:
                    stats["outer_frames"] += 1

    # --- Print Summary Report ---
    print("\n" + "=" * 60)
    print(" " * 17 + "DATASET ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total Frames Analyzed: {total_frames}")
    # The total count is now the length of our global set of unique action numbers
    print(f"Total Unique Action Sequences Found: {len(all_unique_sequences)}")

    # --- Data Integrity Check ---
    print("\n--- Data Integrity Check vs. Paper Claims ---")
    expected_sequences = set(range(1228))  # As per paper, action_number is 0-1227
    found_sequences = all_unique_sequences
    missing_sequences = sorted(list(expected_sequences - found_sequences))

    if len(missing_sequences) > 0:
        print(f"Discrepancy Found: The paper claims 1228 unique interactions (0-1227).")
        print(f"Your dataset contains {len(found_sequences)} unique interactions.")
        print(
            f"The following {len(missing_sequences)} action numbers are missing from the metadata files:"
        )
        print(f"  {missing_sequences}")
    else:
        print(
            "Data Integrity Check Passed: Found all 1228 unique action sequences (0-1227)."
        )

    print("\n--- User Activity ---")
    if total_frames > 0:
        for user, count in sorted(user_stats.items()):
            print(f"User ID {user}: {count} frames ({count/total_frames:.2%})")

    print("\n--- Action Label Breakdown ---")

    # Prepare data for tabulation
    report_data = []
    for action, stats in sorted(action_stats.items()):
        total_action_frames = stats["total_frames"]

        # Avoid division by zero when calculating percentages
        if total_action_frames > 0:
            inner_percent_str = f"({stats['inner_frames'] / total_action_frames:.2%})"
            outer_percent_str = f"({stats['outer_frames'] / total_action_frames:.2%})"
        else:
            inner_percent_str = "(0.00%)"
            outer_percent_str = "(0.00%)"

        report_data.append(
            {
                "Action Label": action,
                "Total Frames": total_action_frames,
                "Frame %": (
                    f"{total_action_frames / total_frames:.2%}"
                    if total_frames > 0
                    else "0.00%"
                ),
                "Inner Frames": f"{stats['inner_frames']} {inner_percent_str}",
                "Outer Frames": f"{stats['outer_frames']} {outer_percent_str}",
                "Unique Sequences": len(stats["action_sequences"]),
            }
        )

    # --- Add a Total row for the report ---
    if report_data:
        report_data.append(
            {
                "Action Label": "Total",
                "Total Frames": total_frames,
                "Frame %": "100.00%",
                "Inner Frames": "",  # Not applicable for total
                "Outer Frames": "",  # Not applicable for total
                "Unique Sequences": len(all_unique_sequences),
            }
        )

    # Display the report
    if PANDAS_AVAILABLE and report_data:
        df = pd.DataFrame(report_data)
        # Set wider column for better display
        pd.set_option("display.max_colwidth", 25)
        print(df.to_string(index=False))
    elif report_data:
        # Fallback to manual printing if pandas is not available
        header = list(report_data[0].keys())
        print(
            f"{header[0]:<25} | {header[1]:<15} | {header[2]:<10} | {header[3]:<22} | {header[4]:<22} | {header[5]:<20}"
        )
        print("-" * 130)
        for row in report_data:
            if row["Action Label"] == "Total":
                print("-" * 130)
            print(
                f"{row['Action Label']:<25} | {row['Total Frames']:<15} | {row['Frame %']:<10} | {row['Inner Frames']:<22} | {row['Outer Frames']:<22} | {row['Unique Sequences']:<20}"
            )

    # Print the LaTeX version of the table
    print_latex_table(report_data)

    print("=" * 60)


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34  # Process train-000 to train-033
    analyze_dataset(DATA_PREFIX, NUM_FOLDERS)
