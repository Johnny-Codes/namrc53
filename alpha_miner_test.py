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


def run_alpha_miner_on_sessions(
    data_directory_prefix,
    num_folders,
    session_timeout_minutes=10,
    filter_action=None,
    noise_threshold=0.1,
):
    """
    Performs Alpha Miner process discovery on user sessions from the dataset.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
        session_timeout_minutes (int): The maximum time in minutes between two actions
                                       before a new session is considered to have started.
        filter_action (str): The specific action label to filter sessions by. Only sessions
                             containing this action will be analyzed.
        noise_threshold (float): Threshold for filtering out noisy relations (0.0 to 1.0).
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
    total_files = 0
    for current_dir_path in training_file_paths:
        metadata_path = os.path.join(current_dir_path, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            print(f"Warning: {metadata_path} not found")
            continue

        total_files += 1
        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    all_frames.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    print(f"Processed {total_files} metadata files with {len(all_frames)} total frames")

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
    print(f"Found {len(sorted_sequences)} action sequences")

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

    # Show session statistics
    session_lengths = [len(session) for session in filtered_sessions]
    avg_length = sum(session_lengths) / len(session_lengths) if session_lengths else 0
    print(f"Average session length: {avg_length:.2f} actions")
    print(
        f"Session length range: {min(session_lengths)} - {max(session_lengths)} actions"
    )

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

    # Convert dataframe to a pm4py event log object
    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    # Show activity statistics
    activities = df["concept:name"].unique()
    print(f"\nFound {len(activities)} unique activities:")
    activity_counts = df["concept:name"].value_counts()
    for activity, count in activity_counts.head(10).items():
        print(f"  {activity}: {count} occurrences")

    # --- Step 7: Apply Alpha Miner algorithm ---
    print("\n" + "=" * 60)
    print("APPLYING ALPHA MINER ALGORITHM")
    print("=" * 60)

    try:
        # Apply noise filtering if specified - FIXED
        if noise_threshold > 0:
            print(f"Applying noise filtering with threshold: {noise_threshold}")
            # Use the correct function and parameter name
            original_cases = len(log)
            log = pm4py.filter_activities_rework(log, activity_key="concept:name")

            # Alternative: Filter by most frequent activities
            activity_counts = pm4py.get_attribute_values(log, "concept:name")
            total_activities = sum(activity_counts.values())
            min_occurrences = int(total_activities * noise_threshold)

            frequent_activities = [
                activity
                for activity, count in activity_counts.items()
                if count >= min_occurrences
            ]

            print(
                f"Keeping {len(frequent_activities)} activities with >= {min_occurrences} occurrences"
            )
            print(f"Filtered activities: {frequent_activities}")

            # Filter log to keep only frequent activities
            log = pm4py.filter_event_attribute_values(
                log, "concept:name", frequent_activities, level="event", retain=True
            )

            print(f"Log after noise filtering: {len(log)} cases (was {original_cases})")

        # Discover Petri Net using Alpha Miner
        print("Discovering process model using Alpha Miner...")
        net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(log)

        print(f"Discovered Petri Net with:")
        print(f"  - {len(net.places)} places")
        print(f"  - {len(net.transitions)} transitions")
        print(f"  - {len(net.arcs)} arcs")

        # --- Step 8: Analyze the discovered model ---
        print("\n" + "=" * 50)
        print("MODEL ANALYSIS")
        print("=" * 50)

        # Show transitions (activities)
        print("\nTransitions (Activities):")
        visible_transitions = []
        invisible_transitions = []

        for i, trans in enumerate(net.transitions):
            if trans.label:  # Visible transition
                visible_transitions.append(trans.label)
                print(f"  {i+1}. {trans.label}")
            else:  # Invisible/tau transition
                invisible_transitions.append(trans)

        print(f"\nVisible transitions: {len(visible_transitions)}")
        print(f"Invisible transitions: {len(invisible_transitions)}")

        # Show places
        print(f"\nPlaces: {len(net.places)} total")

        # Analyze control flow patterns
        print("\nControl Flow Analysis:")
        print(f"  - Start activities: {pm4py.get_start_activities(log)}")
        print(f"  - End activities: {pm4py.get_end_activities(log)}")

        # Calculate fitness and precision
        print("\nEvaluating model quality...")
        try:
            fitness = pm4py.fitness_token_based_replay(
                log, net, initial_marking, final_marking
            )
            print(f"Model Fitness: {fitness['log_fitness']:.3f}")
            print(f"  (1.0 = perfect fitness, 0.0 = no fitness)")
        except Exception as e:
            print(f"Could not calculate fitness: {e}")

        try:
            precision = pm4py.precision_token_based_replay(
                log, net, initial_marking, final_marking
            )
            print(f"Model Precision: {precision:.3f}")
            print(f"  (1.0 = no additional behavior, 0.0 = allows everything)")
        except Exception as e:
            print(f"Could not calculate precision: {e}")

        # Additional model analysis
        print("\nAdditional Analysis:")

        # Check for deadlocks or livelocks
        try:
            from pm4py.algo.analysis.woflan import algorithm as woflan

            is_sound = woflan.apply(net, initial_marking, final_marking)
            print(f"Model soundness: {'Sound' if is_sound else 'Not sound'}")
        except Exception as e:
            print(f"Could not check soundness: {e}")

        # --- Step 9: Visualize the Alpha Miner result ---
        print("\n" + "=" * 50)
        print("VISUALIZATION")
        print("=" * 50)

        # Visualize the Petri Net
        print("Generating Alpha Miner process visualization...")
        try:
            pm4py.view_petri_net(net, initial_marking, final_marking, format="png")
            print("✓ Petri Net visualization generated")
        except Exception as e:
            print(f"✗ Failed to generate Petri Net visualization: {e}")

        # Also create a DFG for comparison
        print("Generating Directly-Follows Graph for comparison...")
        try:
            dfg, start_activities, end_activities = pm4py.discover_dfg(log)
            pm4py.view_dfg(dfg, start_activities, end_activities, format="png")
            print("✓ DFG visualization generated")
        except Exception as e:
            print(f"✗ Failed to generate DFG visualization: {e}")

        # Save the model for later use
        print("Saving discovered model...")
        try:
            pm4py.write_pnml(
                net, initial_marking, final_marking, "alpha_miner_model.pnml"
            )
            print("✓ Model saved to alpha_miner_model.pnml")
        except Exception as e:
            print(f"✗ Failed to save model: {e}")

        # Generate process map
        print("Generating process map...")
        try:
            process_map = pm4py.view_process_tree(
                pm4py.discover_process_tree_inductive(log)
            )
            print("✓ Process map generated")
        except Exception as e:
            print(f"✗ Failed to generate process map: {e}")

        print("\nAlpha Miner analysis complete!")
        print("Generated files:")
        print("  - Petri Net visualization (PNG)")
        print("  - DFG visualization (PNG)")
        print("  - alpha_miner_model.pnml (saved model)")

        # --- Step 10: Pattern Analysis ---
        print("\n" + "=" * 50)
        print("PATTERN ANALYSIS")
        print("=" * 50)

        # Analyze common patterns
        print("Most common activity sequences:")
        variants = pm4py.get_variants(log)
        sorted_variants = sorted(
            variants.items(), key=lambda x: len(x[1]), reverse=True
        )

        for i, (variant, cases) in enumerate(sorted_variants[:5]):
            print(f"  {i+1}. {' → '.join(variant)} ({len(cases)} cases)")

        # Resource analysis if available
        print("\nResource Analysis:")
        cases_by_user = {}
        for case in log:
            user = (
                case.attributes.get("concept:name", "Unknown").split("_")[1]
                if "_" in case.attributes.get("concept:name", "")
                else "Unknown"
            )
            cases_by_user[user] = cases_by_user.get(user, 0) + 1

        print(f"  - {len(cases_by_user)} different users")
        print(f"  - Average cases per user: {len(log) / len(cases_by_user):.1f}")

    except Exception as e:
        print(f"Error applying Alpha Miner: {e}")
        print("This could be due to:")
        print("  - Insufficient data")
        print("  - Complex or noisy process behavior")
        print("  - Activities with very low frequency")
        print("  - Alpha Miner assumptions not met")

        # Fallback to DFG
        print("\nFalling back to Directly-Follows Graph...")
        try:
            dfg, start_activities, end_activities = pm4py.discover_dfg(log)
            pm4py.view_dfg(dfg, start_activities, end_activities, format="png")
            print("✓ DFG fallback generated successfully")

            # Also try Inductive Miner as fallback
            print("Trying Inductive Miner as alternative...")
            tree = pm4py.discover_process_tree_inductive(log)
            net_ind, im_ind, fm_ind = pm4py.convert_to_petri_net(tree)
            pm4py.view_petri_net(net_ind, im_ind, fm_ind, format="png")
            print("✓ Inductive Miner alternative generated")

        except Exception as fallback_error:
            print(f"✗ Fallback also failed: {fallback_error}")


def run_simplified_alpha_miner(
    data_directory_prefix, num_folders, min_session_length=3, max_session_length=20
):
    """
    Run Alpha Miner with simplified preprocessing for better results.
    """
    print("=" * 60)
    print("SIMPLIFIED ALPHA MINER ANALYSIS")
    print("=" * 60)

    # Same data loading logic...
    # [Copy the data loading logic from the main function]

    # Focus on well-structured sessions
    print(
        f"Filtering for sessions with {min_session_length}-{max_session_length} activities..."
    )

    # This would use the same data preparation but with additional filtering
    # Implementation would go here...
    pass


def compare_discovery_algorithms(
    data_directory_prefix, num_folders, session_timeout_minutes=10, filter_action=None
):
    """
    Compare Alpha Miner with other process discovery algorithms.
    """
    if not PM4PY_AVAILABLE:
        return

    print("=" * 60)
    print("COMPARING PROCESS DISCOVERY ALGORITHMS")
    print("=" * 60)

    # Use a simple example first, then apply to real data
    event_log_list = [
        {
            "case:concept:name": "Case1",
            "concept:name": "A",
            "time:timestamp": "2023-01-01 10:00:00",
        },
        {
            "case:concept:name": "Case1",
            "concept:name": "B",
            "time:timestamp": "2023-01-01 10:01:00",
        },
        {
            "case:concept:name": "Case1",
            "concept:name": "C",
            "time:timestamp": "2023-01-01 10:02:00",
        },
        {
            "case:concept:name": "Case2",
            "concept:name": "A",
            "time:timestamp": "2023-01-01 11:00:00",
        },
        {
            "case:concept:name": "Case2",
            "concept:name": "C",
            "time:timestamp": "2023-01-01 11:01:00",
        },
        {
            "case:concept:name": "Case3",
            "concept:name": "A",
            "time:timestamp": "2023-01-01 12:00:00",
        },
        {
            "case:concept:name": "Case3",
            "concept:name": "B",
            "time:timestamp": "2023-01-01 12:01:00",
        },
        {
            "case:concept:name": "Case3",
            "concept:name": "C",
            "time:timestamp": "2023-01-01 12:02:00",
        },
    ]

    df = pd.DataFrame(event_log_list)
    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    algorithms = {
        "Alpha Miner": pm4py.discover_petri_net_alpha,
        "Inductive Miner": pm4py.discover_petri_net_inductive,
        "Heuristic Miner": pm4py.discover_petri_net_heuristics,
    }

    results = {}
    for name, algorithm in algorithms.items():
        print(f"\n--- {name} ---")
        try:
            net, im, fm = algorithm(log)
            print(f"Places: {len(net.places)}, Transitions: {len(net.transitions)}")

            # Calculate fitness
            fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
            fitness_score = fitness["log_fitness"]
            print(f"Fitness: {fitness_score:.3f}")

            results[name] = {
                "places": len(net.places),
                "transitions": len(net.transitions),
                "fitness": fitness_score,
            }

        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = {"error": str(e)}

    # Summary comparison
    print("\n" + "=" * 40)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 40)
    for name, result in results.items():
        if "error" not in result:
            print(
                f"{name:15}: {result['places']} places, {result['transitions']} transitions, fitness={result['fitness']:.3f}"
            )
        else:
            print(f"{name:15}: ERROR - {result['error']}")


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    # The range function is exclusive, so 34 will process folders 0 to 33.
    NUM_FOLDERS = 34
    # Set to None to analyze all actions, or specify an action like 'inspecting_buildplate'
    ACTION_TO_ANALYZE = None

    # Run Alpha Miner analysis
    run_alpha_miner_on_sessions(
        DATA_PREFIX,
        NUM_FOLDERS,
        session_timeout_minutes=10,
        filter_action=ACTION_TO_ANALYZE,
        noise_threshold=0.05,  # Reduced threshold - keep activities that occur in >=5% of cases
    )

    # Uncomment to compare algorithms
    print("\n" + "=" * 60)
    compare_discovery_algorithms(DATA_PREFIX, NUM_FOLDERS)
