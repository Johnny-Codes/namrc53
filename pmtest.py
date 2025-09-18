# import json
# import os
# from collections import defaultdict
# from datetime import datetime, timedelta
# import random

# # --- Add the Graphviz bin directory to the system's PATH for this script's execution ---
# # This is a robust way to ensure the script can find the 'dot.exe' executable
# # without requiring manual changes to system environment variables.
# # Update this path if you installed Graphviz in a different location.
# graphviz_bin_path = "C:/Program Files/Graphviz/bin"
# if os.path.exists(graphviz_bin_path):
#     os.environ["PATH"] += os.pathsep + graphviz_bin_path
# else:
#     print(
#         f"Warning: Graphviz path not found at '{graphviz_bin_path}'. Visualization may fail."
#     )
#     print("Please update the 'graphviz_bin_path' variable in the script if needed.")


# # --- pm4py is required for this script ---
# try:
#     import pandas as pd
#     import pm4py

#     PM4PY_AVAILABLE = True
# except ImportError:
#     PM4PY_AVAILABLE = False
#     print("Error: Required libraries not found. Please run:")
#     print("pip install pm4py pandas")


# def run_process_mining_on_filtered_sessions(
#     data_directory_prefix, num_folders, session_timeout_minutes=5, filter_action=None
# ):
#     """
#     Performs process mining on a filtered subset of user sessions from the dataset.

#     Args:
#         data_directory_prefix (str): The root path to the data folders (e.g., "./data").
#         num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
#         session_timeout_minutes (int): The maximum time in minutes between two actions
#                                        before a new session is considered to have started.
#         filter_action (str): The specific action label to filter sessions by. Only sessions
#                              containing this action will be analyzed.
#     """
#     if not PM4PY_AVAILABLE:
#         return

#     # --- Step 1: Aggregate all frame data from metadata files ---
#     all_frames = []
#     training_file_paths = []
#     for i in range(num_folders):
#         folder_name = f"train-{i:03d}"
#         training_file_paths.append(os.path.join(data_directory_prefix, folder_name))

#     print("Reading all metadata files...")
#     for current_dir_path in training_file_paths:
#         metadata_path = os.path.join(current_dir_path, "metadata.jsonl")
#         if not os.path.exists(metadata_path):
#             continue
#         with open(metadata_path, "r") as f:
#             for line in f:
#                 try:
#                     all_frames.append(json.loads(line.strip()))
#                 except json.JSONDecodeError:
#                     continue

#     # --- Step 2: Group frames by action_number to find start/end times ---
#     print("Grouping frames by action sequence...")
#     action_sequences = defaultdict(
#         lambda: {"start_time": None, "end_time": None, "label": "", "user": None}
#     )

#     for frame in all_frames:
#         action_num = frame.get("action_number")
#         if action_num is None:
#             continue

#         try:
#             timestamp = datetime.fromisoformat(frame["datetime"])
#         except (ValueError, KeyError):
#             continue

#         seq = action_sequences[action_num]

#         if seq["start_time"] is None or timestamp < seq["start_time"]:
#             seq["start_time"] = timestamp
#             seq["label"] = frame.get("action_label", "unknown")
#             seq["user"] = frame.get("user_label", "unknown")

#         if seq["end_time"] is None or timestamp > seq["end_time"]:
#             seq["end_time"] = timestamp

#     # --- Step 3: Sort all sequences chronologically ---
#     print("Sorting sequences by timestamp...")
#     sorted_sequences = sorted(action_sequences.values(), key=lambda x: x["start_time"])

#     # --- Step 4: Group sorted actions into sessions (cases) ---
#     print(
#         f"Grouping actions into sessions (timeout = {session_timeout_minutes} minutes)..."
#     )
#     sessions = []
#     if not sorted_sequences:
#         print("No valid sequences found.")
#         return

#     current_session = []
#     last_action_time = sorted_sequences[0]["end_time"]
#     last_user = sorted_sequences[0]["user"]

#     for action in sorted_sequences:
#         time_gap = (action["start_time"] - last_action_time).total_seconds() / 60.0

#         if action["user"] != last_user or time_gap > session_timeout_minutes:
#             if current_session:
#                 sessions.append(current_session)
#             current_session = []

#         current_session.append(action)
#         last_action_time = action["end_time"]
#         last_user = action["user"]

#     if current_session:
#         sessions.append(current_session)

#     print(f"Found {len(sessions)} total distinct user sessions.")

#     # --- Step 5: Filter sessions to include only those with the target action ---
#     if filter_action:
#         print(f"\nFiltering for sessions that include the action: '{filter_action}'...")
#         filtered_sessions = []
#         for session in sessions:
#             if any(action["label"] == filter_action for action in session):
#                 filtered_sessions.append(session)
#         print(f"Found {len(filtered_sessions)} sessions containing '{filter_action}'.")
#     else:
#         # If no filter is specified, analyze all sessions
#         filtered_sessions = sessions

#     # --- Step 6: Format the filtered sessions for pm4py ---
#     if not filtered_sessions:
#         print("Could not find any sessions matching the filter criteria to analyze.")
#         return

#     print(f"\nAnalyzing {len(filtered_sessions)} filtered sessions...")

#     event_log_list = []
#     for i, session in enumerate(filtered_sessions):
#         if not session:
#             continue

#         session_user = session[0]["user"]
#         session_id = f"User_{session_user}_Session_{i+1}"

#         for action in session:
#             event_log_list.append(
#                 {
#                     "case:concept:name": session_id,
#                     "concept:name": action["label"],
#                     "time:timestamp": action["start_time"],
#                 }
#             )

#     if not event_log_list:
#         print("No events found to process.")
#         return

#     df = pd.DataFrame(event_log_list)

#     # --- Step 7: Use pm4py to discover and visualize the process ---
#     print("Generating process map for the filtered dataset...")

#     # Discover a Directly-Follows Graph (DFG) from the filtered event log
#     dfg, start_activities, end_activities = pm4py.discover_dfg(df)

#     # Visualize the DFG
#     pm4py.view_dfg(dfg, start_activities, end_activities, format="png")

#     print("\nProcess map visualization complete.")
#     print(
#         "The map has been saved as 'dfg_process_map.png' and should open in a new window."
#     )


# # --- Run the Analysis ---
# if __name__ == "__main__":
#     DATA_PREFIX = "./data"
#     NUM_FOLDERS = 34  # Process train-000 to train-032
#     # Specify the action you want to focus on here
#     ACTION_TO_ANALYZE = "inspecting_buildplate"

#     run_process_mining_on_filtered_sessions(
#         DATA_PREFIX, NUM_FOLDERS, filter_action=None
#     )
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import random

# --- Add the Graphviz bin directory to the system's PATH for this script's execution ---
# This is a robust way to ensure the script can find the 'dot.exe' executable
# without requiring manual changes to system environment variables.
# Update this path if you installed Graphviz in a different location.
graphviz_bin_path = "C:/Program Files/Graphviz/bin"
if os.path.exists(graphviz_bin_path):
    os.environ["PATH"] += os.pathsep + graphviz_bin_path
else:
    print(
        f"Warning: Graphviz path not found at '{graphviz_bin_path}'. Visualization may fail."
    )
    print("Please update the 'graphviz_bin_path' variable in the script if needed.")


# --- pm4py is required for this script ---
try:
    import pandas as pd
    import pm4py

    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("Error: Required libraries not found. Please run:")
    print("pip install pm4py pandas")


def run_process_mining_on_filtered_sessions(
    data_directory_prefix, num_folders, session_timeout_minutes=10, filter_action=None
):
    """
    Performs process mining on a filtered subset of user sessions from the dataset.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
        session_timeout_minutes (int): The maximum time in minutes between two actions
                                       before a new session is considered to have started.
        filter_action (str): The specific action label to filter sessions by. Only sessions
                             containing this action will be analyzed.
    """
    if not PM4PY_AVAILABLE:
        return

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

    # --- Step 2: Group frames by action_number to find start/end times ---
    print("Grouping frames by action sequence...")
    action_sequences = defaultdict(
        lambda: {"start_time": None, "end_time": None, "label": "", "user": None}
    )

    for frame in all_frames:
        action_num = frame.get("action_number")
        if action_num is None:
            continue

        try:
            timestamp = datetime.fromisoformat(frame["datetime"])
        except (ValueError, KeyError):
            continue

        seq = action_sequences[action_num]

        if seq["start_time"] is None or timestamp < seq["start_time"]:
            seq["start_time"] = timestamp
            seq["label"] = frame.get("action_label", "unknown")
            seq["user"] = frame.get("user_label", "unknown")

        if seq["end_time"] is None or timestamp > seq["end_time"]:
            seq["end_time"] = timestamp

    # --- Step 3: Sort all sequences chronologically ---
    print("Sorting sequences by timestamp...")
    sorted_sequences = sorted(action_sequences.values(), key=lambda x: x["start_time"])

    # --- Step 4: Group sorted actions into sessions (cases) ---
    print(
        f"Grouping actions into sessions (timeout = {session_timeout_minutes} minutes)..."
    )
    sessions = []
    if not sorted_sequences:
        print("No valid sequences found.")
        return

    current_session = []
    last_action_time = sorted_sequences[0]["end_time"]
    last_user = sorted_sequences[0]["user"]

    for action in sorted_sequences:
        time_gap = (action["start_time"] - last_action_time).total_seconds() / 60.0

        if action["user"] != last_user or time_gap > session_timeout_minutes:
            if current_session:
                sessions.append(current_session)
            current_session = []

        current_session.append(action)
        last_action_time = action["end_time"]
        last_user = action["user"]

    if current_session:
        sessions.append(current_session)

    print(f"Found {len(sessions)} total distinct user sessions.")

    # --- Step 5: Filter sessions to include only those with the target action ---
    if filter_action:
        print(f"\nFiltering for sessions that include the action: '{filter_action}'...")
        filtered_sessions = []
        for session in sessions:
            if any(action["label"] == filter_action for action in session):
                filtered_sessions.append(session)
        print(f"Found {len(filtered_sessions)} sessions containing '{filter_action}'.")
    else:
        # If no filter is specified, analyze all sessions
        filtered_sessions = sessions

    # --- Step 6: Format the filtered sessions for pm4py ---
    if not filtered_sessions:
        print("Could not find any sessions matching the filter criteria to analyze.")
        return

    print(f"\nAnalyzing {len(filtered_sessions)} filtered sessions...")

    event_log_list = []
    for i, session in enumerate(filtered_sessions):
        if not session:
            continue

        session_user = session[0]["user"]
        session_id = f"User_{session_user}_Session_{i+1}"

        for action in session:
            event_log_list.append(
                {
                    "case:concept:name": session_id,
                    "concept:name": action["label"],
                    "time:timestamp": action["start_time"],
                }
            )

    if not event_log_list:
        print("No events found to process.")
        return

    df = pd.DataFrame(event_log_list)

    # Convert dataframe to a pm4py event log object, which is needed for discovery algorithms
    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    # --- Step 7: Use pm4py to discover and visualize a cleaner process model ---
    print("Generating a structured process map using the Inductive Miner...")

    # Discover a Petri Net using the Inductive Miner algorithm.
    # This creates a much cleaner, more structured process map than a simple DFG.
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)

    # Visualize the Petri Net directly using the correct function, decorated with frequency information
    pm4py.view_petri_net(net, initial_marking, final_marking, format="png", log=log)

    print("\nProcess map visualization complete.")
    print("The map has been saved as a temporary file and should open in a new window.")


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    # The range function is exclusive, so 34 will process folders 0 to 33.
    NUM_FOLDERS = 34
    # Set to None to analyze all actions, or specify an action like 'inspecting_buildplate'
    ACTION_TO_ANALYZE = None

    run_process_mining_on_filtered_sessions(
        DATA_PREFIX, NUM_FOLDERS, filter_action=ACTION_TO_ANALYZE
    )
