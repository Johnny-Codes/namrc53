import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd

# --- pm4py is required for this script ---
try:
    import pm4py

    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("Error: Required libraries not found. Please run: pip install pm4py pandas")

# --- Configuration ---
# A list of all tasks to analyze. The script will loop through each one.
ALL_TASKS_TO_ANALYZE = [
    "using_control_panel",
    "using_flexpendant_mounted",
    "using_flexpendant_mobile",
    "inspecting_buildplate",
    "preparing_buildplate",
    "refit_buildplate",
    "grinding_buildplate",
    "toggle_lights",
    "open_door",
    "close_door",
    "turning_gas_knobs",
    "adjusting_tool",
    "wiring",
    "donning_ppe",
    "doffing_ppe",
    "observing",
    "walking",
]


def analyze_tacit_knowledge(
    data_directory_prefix, num_folders, session_timeout_minutes=15
):
    """
    Performs process mining to analyze and quantify operator variability and tacit knowledge
    for all manufacturing tasks.
    """
    if not PM4PY_AVAILABLE:
        return

    # --- Step 1 & 2: Aggregate all actions and group by action_number ---
    print("Reading all metadata files and consolidating actions...")
    action_sequences = defaultdict(
        lambda: {"start_time": None, "end_time": None, "label": "", "user": None}
    )
    all_frames = []
    training_file_paths = []
    for i in range(num_folders):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(data_directory_prefix, folder_name))

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

    # --- Step 3: Group all actions into sessions ---
    print("Grouping all actions into user sessions...")
    sorted_sequences = sorted(action_sequences.values(), key=lambda x: x["start_time"])
    sessions = []
    current_session = []
    if sorted_sequences:
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

    # --- Main Analysis Loop ---
    full_report_string = ""
    output_maps_dir = "process_maps"
    os.makedirs(output_maps_dir, exist_ok=True)

    for task_to_analyze in ALL_TASKS_TO_ANALYZE:
        # --- Step 4: Filter sessions for the target task ---
        task_sessions = [
            s for s in sessions if any(a["label"] == task_to_analyze for a in s)
        ]

        header = f"\n\n{'='*120}\n"
        header += f"{'OPERATOR VARIABILITY ANALYSIS REPORT':^120}\n"
        header += f"{'(Task: ' + task_to_analyze + ')':^120}\n"
        header += f"{'='*120}\n"

        print(header)
        full_report_string += header

        if not task_sessions:
            no_session_msg = (
                f"No sessions found related to the task '{task_to_analyze}'.\n"
            )
            print(no_session_msg)
            full_report_string += no_session_msg
            continue

        # --- Step 5: Discover the overall most common workflow to use as the baseline ---
        all_task_events = []
        for i, session in enumerate(task_sessions):
            for action in session:
                all_task_events.append(
                    {
                        "case:concept:name": f"session_{i}",
                        "concept:name": action["label"],
                        "time:timestamp": action["start_time"],
                    }
                )

        overall_df = pd.DataFrame(all_task_events)
        overall_log = pm4py.format_dataframe(
            overall_df,
            case_id="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )
        overall_variants = pm4py.get_variants_as_tuples(overall_log)
        baseline_sequence = (
            max(overall_variants, key=overall_variants.get)
            if overall_variants
            else tuple()
        )

        baseline_msg = (
            f"Discovered Baseline Process: {' -> '.join(baseline_sequence)}\n"
        )
        print(baseline_msg)
        full_report_string += baseline_msg

        # --- ADDED: Generate and save a process map for the current task ---
        print(f"Generating and saving process map for '{task_to_analyze}'...")
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
            overall_log
        )
        map_file_path = os.path.join(
            output_maps_dir, f"process_map_{task_to_analyze}.png"
        )
        pm4py.save_vis_petri_net(net, initial_marking, final_marking, map_file_path)
        print(f"Map saved to {map_file_path}")

        # --- Step 6: Group task sessions by user and analyze ---
        user_task_sessions = defaultdict(list)
        for session in task_sessions:
            user_id = session[0]["user"]
            user_task_sessions[user_id].append(session)

        report_data = []
        for user_id, user_sessions in sorted(user_task_sessions.items()):
            event_log_list = []
            for i, session in enumerate(user_sessions):
                for action in session:
                    event_log_list.append(
                        {
                            "case:concept:name": f"session_{i}",
                            "concept:name": action["label"],
                            "time:timestamp": action["start_time"],
                        }
                    )

            if not event_log_list:
                continue

            df = pd.DataFrame(event_log_list)
            log = pm4py.format_dataframe(
                df,
                case_id="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
            variants = pm4py.get_variants_as_tuples(log)
            most_common_sequence = (
                max(variants, key=variants.get) if variants else tuple()
            )

            total_task_duration, total_steps, total_step_duration = 0, 0, 0
            for session in user_sessions:
                if len(session) > 1:
                    total_task_duration += (
                        session[-1]["end_time"] - session[0]["start_time"]
                    ).total_seconds()
                for action in session:
                    total_steps += 1
                    total_step_duration += (
                        action["end_time"] - action["start_time"]
                    ).total_seconds()

            avg_task_duration = (
                total_task_duration / len(user_sessions) if user_sessions else 0
            )
            avg_step_duration = (
                total_step_duration / total_steps if total_steps > 0 else 0
            )

            baseline_set = set(baseline_sequence)
            user_set = set(most_common_sequence)
            skipped_steps = list(baseline_set - user_set)
            added_steps = list(user_set - baseline_set)
            deviations = []
            if skipped_steps:
                deviations.append(f"Skips: {', '.join(skipped_steps)}")
            if added_steps:
                deviations.append(f"Adds: {', '.join(added_steps)}")

            report_data.append(
                {
                    "Operator ID": f"User {user_id}",
                    "Discovered Step Sequence": " -> ".join(most_common_sequence),
                    "Avg Step Duration (s)": f"{avg_step_duration:.1f}",
                    "Avg Total Task Duration (s)": f"{avg_task_duration:.1f}",
                    "Key Deviations from Baseline": (
                        ". ".join(deviations)
                        if deviations
                        else "Follows baseline process."
                    ),
                }
            )

        report_data.append(
            {
                "Operator ID": "Most Common Process (Baseline)",
                "Discovered Step Sequence": " -> ".join(baseline_sequence),
                "Avg Step Duration (s)": "N/A",
                "Avg Total Task Duration (s)": "N/A",
                "Key Deviations from Baseline": "Data-driven baseline model.",
            }
        )

        # --- Step 7: Display and store the final report ---
        report_df = pd.DataFrame(report_data)
        pd.set_option("display.max_colwidth", 80)
        pd.set_option("display.width", 150)
        report_table_string = report_df.to_string(index=False)
        print(report_table_string)
        full_report_string += report_table_string + "\n"

    # --- Save the full report to a file ---
    report_filename = "operator_analysis_report.txt"
    with open(report_filename, "w") as f:
        f.write(full_report_string)
    print(f"\n\nFull analysis saved to '{report_filename}'")


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34  # Process train-000 to train-033
    analyze_tacit_knowledge(DATA_PREFIX, NUM_FOLDERS)
