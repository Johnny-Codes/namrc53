import json
import os
from collections import Counter


def summarize_user_actions(input_directory="user_breakdown"):
    """
    Reads the user-specific JSON files and prints a summary of action
    labels for each user.

    Args:
        input_directory (str): The directory containing the user data files.
    """
    if not os.path.exists(input_directory):
        print(f"Error: Directory '{input_directory}' not found.")
        print("Please run the 'user-breakdown-script' first to generate the data.")
        return

    # Find all user data files in the directory
    user_files = [
        f
        for f in os.listdir(input_directory)
        if f.startswith("user_") and f.endswith(".json")
    ]

    if not user_files:
        print(f"No user data files found in '{input_directory}'.")
        return

    print("\n" + "=" * 50)
    print(" " * 14 + "USER ACTION SUMMARY")
    print("=" * 50)

    # Process each user file
    for filename in sorted(user_files):
        file_path = os.path.join(input_directory, filename)

        with open(file_path, "r") as f:
            try:
                user_frames = json.load(f)
            except json.JSONDecodeError:
                print(f"\nError reading {filename}. Skipping.")
                continue

        if not user_frames:
            continue

        # Extract the user ID from the filename
        user_id = filename.replace("user_", "").replace("_data.json", "")
        total_frames = len(user_frames)

        # Count the occurrences of each action label
        action_labels = [frame.get("action_label", "unknown") for frame in user_frames]
        action_counts = Counter(action_labels)

        # Print the summary for the current user
        print(
            f"\n--- Summary for User ID: {user_id} (Total Frames: {total_frames}) ---"
        )

        # Sort actions by the most frequent for readability
        for action, count in action_counts.most_common():
            percentage = (count / total_frames) * 100
            print(f"  - {action:<25} : {count:>7} frames ({percentage:.2f}%)")

    print("\n" + "=" * 50)


# --- Run the Script ---
if __name__ == "__main__":
    summarize_user_actions()
