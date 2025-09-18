import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
import networkx as nx
import seaborn as sns

# --- Add the Graphviz bin directory to the system's PATH ---
graphviz_bin_path = "C:/Program Files/Graphviz/bin"
if os.path.exists(graphviz_bin_path):
    os.environ["PATH"] += os.pathsep + graphviz_bin_path
else:
    print(
        f"Warning: Graphviz path not found at '{graphviz_bin_path}'. Visualization may fail."
    )

try:
    import pm4py

    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("Error: pm4py required. Run: pip install pm4py")

# Global variable for output directory
OUTPUT_DIR = None


def setup_output_directory(base_name="tacit_workflow_analysis"):
    """Create output directory with timestamp"""
    global OUTPUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"{base_name}_{timestamp}"

    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create subdirectories for different types of outputs
    subdirs = ["dfg_visualizations", "analysis_plots", "control_charts", "reports"]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    print(f"Output directory created: {OUTPUT_DIR}")
    return OUTPUT_DIR


def get_output_path(filename, subdir=None):
    """Get full path for output file"""
    if subdir:
        return os.path.join(OUTPUT_DIR, subdir, filename)
    else:
        return os.path.join(OUTPUT_DIR, filename)


def discover_tacit_workflows(
    data_directory_prefix,
    num_folders,
    sequence_break_minutes=15,  # Time gap that indicates a new sequence
    min_sequence_length=2,
    activity_filter=None,
):
    """
    Discover tacit knowledge workflows by identifying complete sequences
    from initiation to natural completion.

    Args:
        data_directory_prefix: Path to data folders
        num_folders: Number of folders to process
        sequence_break_minutes: Minutes of inactivity that indicates sequence end
        min_sequence_length: Minimum activities in a sequence
        activity_filter: List of activities to focus on (None for all)
    """

    # Setup output directory
    setup_output_directory()

    print("=" * 80)
    print("TACIT WORKFLOW DISCOVERY")
    print("=" * 80)
    print(f"Sequence break threshold: {sequence_break_minutes} minutes")
    print(f"Minimum sequence length: {min_sequence_length} activities")
    print(f"Output directory: {OUTPUT_DIR}")

    # --- Step 1: Load and prepare activity data ---
    all_activities = load_activity_data(data_directory_prefix, num_folders)

    if not all_activities:
        print("No activity data found!")
        return

    print(f"Loaded {len(all_activities)} activity records")

    # --- Step 2: Identify complete workflow sequences ---
    workflow_sequences = extract_workflow_sequences(
        all_activities, sequence_break_minutes, min_sequence_length, activity_filter
    )

    print(f"Discovered {len(workflow_sequences)} complete workflow sequences")

    # --- Step 3: Analyze sequences by starting activity ---
    sequences_by_starter = group_sequences_by_starter(workflow_sequences)

    # --- Step 4: Create directly-follows graphs for each starting activity ---
    dfg_results = {}

    for starting_activity, sequences in sequences_by_starter.items():
        if len(sequences) >= 3:  # Only analyze activities with sufficient data
            print(f"\n--- Analyzing workflows starting with: {starting_activity} ---")

            dfg_result = analyze_activity_workflows(
                starting_activity, sequences, sequence_break_minutes
            )

            dfg_results[starting_activity] = dfg_result

    # --- Step 5: Create comprehensive analysis ---
    create_tacit_knowledge_analysis(dfg_results, workflow_sequences)

    # --- Step 6: Generate process control charts ---
    create_process_control_charts(workflow_sequences, sequences_by_starter)

    # --- Step 7: Create summary report ---
    create_analysis_summary(dfg_results, workflow_sequences)

    return dfg_results, workflow_sequences


def load_activity_data(data_directory_prefix, num_folders):
    """Load and prepare activity data from metadata files"""
    all_activities = []

    print("Loading activity data...")
    for i in range(num_folders):
        folder_name = f"train-{i:03d}"
        metadata_path = os.path.join(
            data_directory_prefix, folder_name, "metadata.jsonl"
        )

        if not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    frame = json.loads(line.strip())
                    action_num = frame.get("action_number")
                    if action_num is None:
                        continue

                    all_activities.append(
                        {
                            "action_number": action_num,
                            "activity": frame.get("action_label", "unknown"),
                            "user": frame.get("user_label", "unknown"),
                            "timestamp": datetime.fromisoformat(frame["datetime"]),
                            "folder": folder_name,
                        }
                    )
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

    # Group by action_number and get start/end times for each activity
    action_sequences = defaultdict(
        lambda: {
            "activity": "",
            "user": "",
            "start_time": None,
            "end_time": None,
            "folder": "",
        }
    )

    for record in all_activities:
        action_num = record["action_number"]
        seq = action_sequences[action_num]

        if seq["start_time"] is None or record["timestamp"] < seq["start_time"]:
            seq["start_time"] = record["timestamp"]
            seq["activity"] = record["activity"]
            seq["user"] = record["user"]
            seq["folder"] = record["folder"]

        if seq["end_time"] is None or record["timestamp"] > seq["end_time"]:
            seq["end_time"] = record["timestamp"]

    # Convert to sorted list
    sorted_activities = []
    for action_num, seq in action_sequences.items():
        if seq["start_time"] and seq["end_time"]:
            sorted_activities.append(
                {
                    "action_number": action_num,
                    "activity": seq["activity"],
                    "user": seq["user"],
                    "start_time": seq["start_time"],
                    "end_time": seq["end_time"],
                    "folder": seq["folder"],
                    "duration": (seq["end_time"] - seq["start_time"]).total_seconds()
                    / 60.0,
                }
            )

    return sorted(sorted_activities, key=lambda x: (x["user"], x["start_time"]))


def extract_workflow_sequences(
    all_activities, sequence_break_minutes, min_sequence_length, activity_filter
):
    """Extract complete workflow sequences based on temporal breaks"""

    sequences = []
    sequence_break = timedelta(minutes=sequence_break_minutes)

    # Group activities by user
    user_activities = defaultdict(list)
    for activity in all_activities:
        user_activities[activity["user"]].append(activity)

    print(f"Processing activities for {len(user_activities)} users...")

    for user, activities in user_activities.items():
        # Sort activities by start time for this user
        activities.sort(key=lambda x: x["start_time"])

        current_sequence = []

        for i, activity in enumerate(activities):
            # Apply activity filter if specified
            if activity_filter and activity["activity"] not in activity_filter:
                continue

            # Check if this starts a new sequence
            if (
                current_sequence
                and activity["start_time"] - current_sequence[-1]["end_time"]
                > sequence_break
            ):

                # Save current sequence if it meets minimum length
                if len(current_sequence) >= min_sequence_length:
                    sequences.append(
                        {
                            "user": user,
                            "activities": current_sequence.copy(),
                            "start_time": current_sequence[0]["start_time"],
                            "end_time": current_sequence[-1]["end_time"],
                            "duration": (
                                current_sequence[-1]["end_time"]
                                - current_sequence[0]["start_time"]
                            ).total_seconds()
                            / 60.0,
                            "sequence_id": len(sequences),
                        }
                    )

                current_sequence = []

            current_sequence.append(activity)

        # Don't forget the last sequence
        if len(current_sequence) >= min_sequence_length:
            sequences.append(
                {
                    "user": user,
                    "activities": current_sequence.copy(),
                    "start_time": current_sequence[0]["start_time"],
                    "end_time": current_sequence[-1]["end_time"],
                    "duration": (
                        current_sequence[-1]["end_time"]
                        - current_sequence[0]["start_time"]
                    ).total_seconds()
                    / 60.0,
                    "sequence_id": len(sequences),
                }
            )

    return sequences


def group_sequences_by_starter(workflow_sequences):
    """Group workflow sequences by their starting activity"""
    sequences_by_starter = defaultdict(list)

    for sequence in workflow_sequences:
        starting_activity = sequence["activities"][0]["activity"]
        sequences_by_starter[starting_activity].append(sequence)

    # Sort by frequency
    sorted_starters = sorted(
        sequences_by_starter.items(), key=lambda x: len(x[1]), reverse=True
    )

    print("\nWorkflow sequences by starting activity:")
    for starting_activity, sequences in sorted_starters[:10]:
        avg_length = sum(len(seq["activities"]) for seq in sequences) / len(sequences)
        avg_duration = sum(seq["duration"] for seq in sequences) / len(sequences)
        print(
            f"  {starting_activity}: {len(sequences)} sequences, "
            f"avg {avg_length:.1f} activities, {avg_duration:.1f} min"
        )

    return dict(sequences_by_starter)


def analyze_activity_workflows(starting_activity, sequences, sequence_break_minutes):
    """Create detailed analysis for workflows starting with a specific activity"""

    print(f"Creating directly-follows analysis for '{starting_activity}'")
    print(f"  Analyzing {len(sequences)} sequences")

    # Convert sequences to pm4py event log format
    event_log_data = []

    for seq_idx, sequence in enumerate(sequences):
        case_id = f"{starting_activity}_Seq_{seq_idx}_{sequence['user']}"

        for act_idx, activity in enumerate(sequence["activities"]):
            event_log_data.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": activity["activity"],
                    "time:timestamp": activity["start_time"],
                    "user": sequence["user"],
                    "sequence_position": act_idx,
                    "total_sequence_length": len(sequence["activities"]),
                    "sequence_duration": sequence["duration"],
                }
            )

    # Create pm4py event log
    df = pd.DataFrame(event_log_data)
    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )

    # Generate directly-follows graph
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)

    # Create visualization with custom save path
    try:
        # Clean filename for starting activity
        safe_filename = (
            starting_activity.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        dfg_path = get_output_path(f"dfg_{safe_filename}.png", "dfg_visualizations")

        pm4py.save_vis_dfg(dfg, start_activities, end_activities, dfg_path)
        print(f"  ✓ DFG visualization saved to {dfg_path}")
    except Exception as e:
        print(f"  ✗ Failed to create DFG visualization: {e}")

    # Analyze patterns
    patterns = analyze_workflow_patterns(sequences, starting_activity)

    return {
        "starting_activity": starting_activity,
        "sequence_count": len(sequences),
        "dfg": dfg,
        "start_activities": start_activities,
        "end_activities": end_activities,
        "log": log,
        "patterns": patterns,
        "avg_sequence_length": sum(len(seq["activities"]) for seq in sequences)
        / len(sequences),
        "avg_duration": sum(seq["duration"] for seq in sequences) / len(sequences),
    }


def analyze_workflow_patterns(sequences, starting_activity):
    """Analyze patterns in workflow sequences"""

    patterns = {
        "common_sequences": Counter(),
        "next_activities": Counter(),
        "completion_activities": Counter(),
        "sequence_lengths": [],
        "durations": [],
        "user_variations": defaultdict(list),
    }

    for sequence in sequences:
        activities = [act["activity"] for act in sequence["activities"]]

        # Full sequence pattern
        patterns["common_sequences"][tuple(activities)] += 1

        # What comes after the starting activity
        if len(activities) > 1:
            patterns["next_activities"][activities[1]] += 1

        # How sequences typically end
        patterns["completion_activities"][activities[-1]] += 1

        # Sequence characteristics
        patterns["sequence_lengths"].append(len(activities))
        patterns["durations"].append(sequence["duration"])

        # User-specific patterns
        patterns["user_variations"][sequence["user"]].append(activities)

    return patterns


def create_tacit_knowledge_analysis(dfg_results, workflow_sequences):
    """Create comprehensive analysis of tacit knowledge patterns"""

    print("\n" + "=" * 60)
    print("TACIT KNOWLEDGE ANALYSIS")
    print("=" * 60)

    # Create summary dashboard
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Tacit Knowledge Discovery Dashboard", fontsize=16, fontweight="bold")

    # 1. Sequence length distribution
    all_lengths = [len(seq["activities"]) for seq in workflow_sequences]
    axes[0, 0].hist(all_lengths, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Sequence Length (Activities)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Workflow Sequence Lengths")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Duration distribution
    all_durations = [seq["duration"] for seq in workflow_sequences]
    axes[0, 1].hist(all_durations, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Duration (Minutes)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Workflow Durations")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Starting activities frequency
    starting_activities = Counter(
        [seq["activities"][0]["activity"] for seq in workflow_sequences]
    )
    top_starters = starting_activities.most_common(10)
    activities, counts = zip(*top_starters)

    axes[0, 2].barh(range(len(activities)), counts)
    axes[0, 2].set_yticks(range(len(activities)))
    axes[0, 2].set_yticklabels(activities, fontsize=8)
    axes[0, 2].set_xlabel("Number of Sequences")
    axes[0, 2].set_title("Most Common Starting Activities")

    # 4. Ending activities frequency
    ending_activities = Counter(
        [seq["activities"][-1]["activity"] for seq in workflow_sequences]
    )
    top_enders = ending_activities.most_common(10)
    activities, counts = zip(*top_enders)

    axes[1, 0].barh(range(len(activities)), counts)
    axes[1, 0].set_yticks(range(len(activities)))
    axes[1, 0].set_yticklabels(activities, fontsize=8)
    axes[1, 0].set_xlabel("Number of Sequences")
    axes[1, 0].set_title("Most Common Ending Activities")

    # 5. User sequence count
    user_sequences = Counter([seq["user"] for seq in workflow_sequences])
    top_users = user_sequences.most_common(10)
    users, counts = zip(*top_users)

    axes[1, 1].bar(range(len(users)), counts)
    axes[1, 1].set_xticks(range(len(users)))
    axes[1, 1].set_xticklabels([f"User_{u}" for u in users], rotation=45, fontsize=8)
    axes[1, 1].set_ylabel("Number of Sequences")
    axes[1, 1].set_title("Sequences by User")

    # 6. Complexity analysis (unique patterns vs repetitions)
    pattern_counts = Counter()
    for seq in workflow_sequences:
        pattern = tuple([act["activity"] for act in seq["activities"]])
        pattern_counts[pattern] += 1

    complexity_data = {
        "Unique Patterns": len([p for p, c in pattern_counts.items() if c == 1]),
        "Repeated Patterns": len([p for p, c in pattern_counts.items() if c > 1]),
        "Highly Repeated (5+)": len([p for p, c in pattern_counts.items() if c >= 5]),
    }

    axes[1, 2].pie(
        complexity_data.values(), labels=complexity_data.keys(), autopct="%1.1f%%"
    )
    axes[1, 2].set_title("Pattern Complexity Distribution")

    plt.tight_layout()

    # Save to subfolder
    dashboard_path = get_output_path("tacit_knowledge_dashboard.png", "analysis_plots")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Dashboard saved to: {dashboard_path}")

    # Generate detailed reports
    generate_tacit_knowledge_reports(dfg_results, workflow_sequences)


def create_process_control_charts(workflow_sequences, sequences_by_starter):
    """Create process control charts for workflow analysis"""

    print("\n" + "=" * 50)
    print("PROCESS CONTROL CHARTS")
    print("=" * 50)

    # Select top starting activities for control charts
    top_starters = sorted(
        sequences_by_starter.items(), key=lambda x: len(x[1]), reverse=True
    )[:6]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        "Process Control Charts - Workflow Characteristics",
        fontsize=16,
        fontweight="bold",
    )
    axes = axes.flatten()

    for idx, (starting_activity, sequences) in enumerate(top_starters):
        if idx >= 6:
            break

        # Calculate sequence lengths over time
        sequence_data = []
        for seq in sequences:
            sequence_data.append(
                {
                    "timestamp": seq["start_time"],
                    "length": len(seq["activities"]),
                    "duration": seq["duration"],
                    "user": seq["user"],
                }
            )

        sequence_data.sort(key=lambda x: x["timestamp"])

        # Create control chart for sequence length
        lengths = [s["length"] for s in sequence_data]
        mean_length = sum(lengths) / len(lengths)
        std_length = (
            sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        ) ** 0.5

        ucl = mean_length + 3 * std_length  # Upper Control Limit
        lcl = max(0, mean_length - 3 * std_length)  # Lower Control Limit

        x_axis = range(len(lengths))

        axes[idx].plot(x_axis, lengths, "bo-", markersize=3, linewidth=1, alpha=0.7)
        axes[idx].axhline(
            y=mean_length,
            color="green",
            linestyle="-",
            label=f"Mean ({mean_length:.1f})",
        )
        axes[idx].axhline(y=ucl, color="red", linestyle="--", label=f"UCL ({ucl:.1f})")
        axes[idx].axhline(y=lcl, color="red", linestyle="--", label=f"LCL ({lcl:.1f})")

        axes[idx].set_title(
            f"{starting_activity}\n({len(sequences)} sequences)", fontsize=10
        )
        axes[idx].set_xlabel("Sequence Order")
        axes[idx].set_ylabel("Activities Count")
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

        # Identify out-of-control points
        out_of_control = [i for i, l in enumerate(lengths) if l > ucl or l < lcl]
        if out_of_control:
            axes[idx].scatter(
                [x_axis[i] for i in out_of_control],
                [lengths[i] for i in out_of_control],
                color="red",
                s=50,
                marker="x",
                linewidth=2,
            )

    plt.tight_layout()

    # Save to subfolder
    control_charts_path = get_output_path(
        "process_control_charts.png", "control_charts"
    )
    plt.savefig(control_charts_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Control charts saved to: {control_charts_path}")


def generate_tacit_knowledge_reports(dfg_results, workflow_sequences):
    """Generate detailed reports about discovered tacit knowledge"""

    print("\nGenerating detailed tacit knowledge reports...")

    # 1. Workflow Pattern Report
    pattern_report = []

    for starting_activity, result in dfg_results.items():
        patterns = result["patterns"]

        # Most common next activities
        top_next = patterns["next_activities"].most_common(3)
        next_activities_str = ", ".join([f"{act} ({cnt})" for act, cnt in top_next])

        # Most common completion activities
        top_completions = patterns["completion_activities"].most_common(3)
        completions_str = ", ".join([f"{act} ({cnt})" for act, cnt in top_completions])

        pattern_report.append(
            {
                "Starting Activity": starting_activity,
                "Total Sequences": result["sequence_count"],
                "Avg Length": f"{result['avg_sequence_length']:.1f}",
                "Avg Duration (min)": f"{result['avg_duration']:.1f}",
                "Common Next Activities": next_activities_str,
                "Common Completions": completions_str,
            }
        )

    # Save to reports subfolder
    report_path = get_output_path("workflow_patterns_report.csv", "reports")
    pd.DataFrame(pattern_report).to_csv(report_path, index=False)

    # 2. Complete Sequence Catalog
    sequence_catalog = []

    for seq in workflow_sequences:
        activities_str = " → ".join([act["activity"] for act in seq["activities"]])
        sequence_catalog.append(
            {
                "Sequence ID": seq["sequence_id"],
                "User": seq["user"],
                "Start Time": seq["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "Duration (min)": f"{seq['duration']:.1f}",
                "Length": len(seq["activities"]),
                "Complete Sequence": activities_str,
                "Starting Activity": seq["activities"][0]["activity"],
                "Ending Activity": seq["activities"][-1]["activity"],
            }
        )

    catalog_path = get_output_path("complete_sequence_catalog.csv", "reports")
    pd.DataFrame(sequence_catalog).to_csv(catalog_path, index=False)

    # 3. User Knowledge Patterns
    user_patterns = defaultdict(
        lambda: {
            "sequences": 0,
            "avg_length": 0,
            "favorite_starters": Counter(),
            "unique_patterns": set(),
            "total_activities": 0,
        }
    )

    for seq in workflow_sequences:
        user = seq["user"]
        user_patterns[user]["sequences"] += 1
        user_patterns[user]["total_activities"] += len(seq["activities"])
        user_patterns[user]["favorite_starters"][seq["activities"][0]["activity"]] += 1

        pattern = tuple([act["activity"] for act in seq["activities"]])
        user_patterns[user]["unique_patterns"].add(pattern)

    user_report = []
    for user, data in user_patterns.items():
        top_starter = (
            data["favorite_starters"].most_common(1)[0]
            if data["favorite_starters"]
            else ("None", 0)
        )

        user_report.append(
            {
                "User": user,
                "Total Sequences": data["sequences"],
                "Avg Sequence Length": f"{data['total_activities'] / data['sequences']:.1f}",
                "Unique Patterns": len(data["unique_patterns"]),
                "Pattern Diversity": f"{len(data['unique_patterns']) / data['sequences']:.2f}",
                "Favorite Starting Activity": f"{top_starter[0]} ({top_starter[1]})",
            }
        )

    user_patterns_path = get_output_path("user_knowledge_patterns.csv", "reports")
    pd.DataFrame(user_report).to_csv(user_patterns_path, index=False)

    print("Generated reports:")
    print(f"  - {report_path}")
    print(f"  - {catalog_path}")
    print(f"  - {user_patterns_path}")

    # 4. Print key insights
    print("\n" + "=" * 60)
    print("KEY TACIT KNOWLEDGE INSIGHTS")
    print("=" * 60)

    # Most standardized vs most variable workflows
    pattern_variability = {}
    for starting_activity, result in dfg_results.items():
        sequences = result["patterns"]["common_sequences"]
        total_sequences = result["sequence_count"]
        unique_patterns = len(sequences)
        variability = unique_patterns / total_sequences
        pattern_variability[starting_activity] = variability

    most_standardized = min(pattern_variability.items(), key=lambda x: x[1])
    most_variable = max(pattern_variability.items(), key=lambda x: x[1])

    print(
        f"Most Standardized Workflow: {most_standardized[0]} (variability: {most_standardized[1]:.2f})"
    )
    print(
        f"Most Variable Workflow: {most_variable[0]} (variability: {most_variable[1]:.2f})"
    )

    # Expert vs novice indicators
    user_expertise = {}
    for user, data in user_patterns.items():
        avg_length = data["total_activities"] / data["sequences"]
        pattern_diversity = len(data["unique_patterns"]) / data["sequences"]
        expertise_score = avg_length * (
            1 - pattern_diversity
        )  # Longer, more consistent = expert
        user_expertise[user] = expertise_score

    expert_users = sorted(user_expertise.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\nPotential Expert Users (consistent, efficient workflows):")
    for user, score in expert_users:
        print(f"  User {user}: expertise score {score:.2f}")


def create_analysis_summary(dfg_results, workflow_sequences):
    """Create a comprehensive summary report"""
    print("\n" + "=" * 50)
    print("CREATING ANALYSIS SUMMARY")
    print("=" * 50)

    summary_path = get_output_path("analysis_summary.txt")

    with open(summary_path, "w") as f:
        f.write("TACIT WORKFLOW DISCOVERY - ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {OUTPUT_DIR}\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"- Total workflow sequences discovered: {len(workflow_sequences)}\n")
        f.write(f"- Unique starting activities analyzed: {len(dfg_results)}\n")
        f.write(
            f"- Average sequence length: {sum(len(seq['activities']) for seq in workflow_sequences) / len(workflow_sequences):.1f}\n"
        )
        f.write(
            f"- Average sequence duration: {sum(seq['duration'] for seq in workflow_sequences) / len(workflow_sequences):.1f} minutes\n\n"
        )

        f.write("GENERATED FILES:\n")
        f.write("1. DFG Visualizations:\n")
        for activity in dfg_results.keys():
            safe_filename = (
                activity.replace(" ", "_").replace("/", "_").replace("\\", "_")
            )
            f.write(f"   - dfg_visualizations/dfg_{safe_filename}.png\n")

        f.write("\n2. Analysis Plots:\n")
        f.write("   - analysis_plots/tacit_knowledge_dashboard.png\n")

        f.write("\n3. Control Charts:\n")
        f.write("   - control_charts/process_control_charts.png\n")

        f.write("\n4. Reports:\n")
        f.write("   - reports/workflow_patterns_report.csv\n")
        f.write("   - reports/complete_sequence_catalog.csv\n")
        f.write("   - reports/user_knowledge_patterns.csv\n")

    print(f"Analysis summary saved to: {summary_path}")


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34

    # Run tacit workflow discovery
    dfg_results, workflow_sequences = discover_tacit_workflows(
        DATA_PREFIX,
        NUM_FOLDERS,
        sequence_break_minutes=5,  # 15 minutes of inactivity indicates new workflow
        min_sequence_length=2,  # At least 2 activities in a sequence
        activity_filter=None,  # None = analyze all activities
    )

    print(f"\nAnalysis complete!")
    print(f"Discovered {len(workflow_sequences)} complete workflow sequences")
    print(f"Generated DFG analysis for {len(dfg_results)} starting activities")
    print(f"All outputs saved in: {OUTPUT_DIR}")
