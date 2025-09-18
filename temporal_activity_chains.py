import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Add the Graphviz bin directory to the system's PATH for this script's execution ---
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
    print("Note: pm4py not available, but we'll still analyze temporal chains")


def analyze_temporal_activity_chains(
    data_directory_prefix,
    num_folders,
    time_window_minutes=10,
    min_chain_length=2,
    max_chain_length=10,
):
    """
    Analyze temporal chains of activities that occur within specified time windows.

    Args:
        data_directory_prefix (str): Root path to data folders
        num_folders (int): Number of training folders to process
        time_window_minutes (int): Time window to look for subsequent activities
        min_chain_length (int): Minimum length of activity chains to consider
        max_chain_length (int): Maximum length of chains to build
    """

    print("=" * 80)
    print("TEMPORAL ACTIVITY CHAIN ANALYSIS")
    print("=" * 80)
    print(f"Time window: {time_window_minutes} minutes")
    print(f"Chain length: {min_chain_length}-{max_chain_length} activities")

    # --- Step 1: Load and prepare all activity data ---
    all_activities = []

    print("\nLoading activity data...")
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

    print(f"Loaded {len(all_activities)} activity records")

    # --- Step 2: Group by action_number and get start/end times ---
    print("Grouping activities by action sequence...")
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
                }
            )

    # Sort by start time
    sorted_activities.sort(key=lambda x: x["start_time"])
    print(f"Found {len(sorted_activities)} unique activity sequences")

    # --- Step 3: Build temporal chains for each activity ---
    print(f"\nBuilding temporal chains within {time_window_minutes}-minute windows...")

    activity_chains = defaultdict(list)  # activity -> list of chains that start with it
    chain_frequencies = Counter()  # count of each chain pattern
    time_window = timedelta(minutes=time_window_minutes)

    for i, current_activity in enumerate(sorted_activities):
        current_end = current_activity["end_time"]
        current_user = current_activity["user"]

        # Find all activities that start within the time window after this activity ends
        chain = [current_activity["activity"]]
        last_end_time = current_end

        # Look for subsequent activities within time windows
        for length in range(1, max_chain_length):
            next_activity = None

            # Find the next activity within the time window for the same user
            for j in range(i + 1, len(sorted_activities)):
                candidate = sorted_activities[j]

                # Skip if different user
                if candidate["user"] != current_user:
                    continue

                # Check if it starts within our time window from the last activity's end
                time_gap = candidate["start_time"] - last_end_time
                if time_gap <= time_window:
                    next_activity = candidate
                    break
                elif time_gap > time_window:
                    # Since activities are sorted, no more candidates for this chain
                    break

            if next_activity:
                chain.append(next_activity["activity"])
                last_end_time = next_activity["end_time"]
            else:
                break

        # Store chain if it meets minimum length requirement
        if len(chain) >= min_chain_length:
            chain_tuple = tuple(chain)
            activity_chains[current_activity["activity"]].append(chain)
            chain_frequencies[chain_tuple] += 1

    # --- Step 4: Analyze results ---
    print("\n" + "=" * 60)
    print("TEMPORAL CHAIN ANALYSIS RESULTS")
    print("=" * 60)

    # Activity statistics
    print("\nActivity Statistics:")
    activity_counts = Counter([act["activity"] for act in sorted_activities])
    print(f"Total unique activities: {len(activity_counts)}")
    for activity, count in activity_counts.most_common(10):
        print(f"  {activity}: {count} occurrences")

    # Chain starting activities
    print(f"\nActivities that start chains: {len(activity_chains)}")
    for activity in sorted(activity_chains.keys()):
        chain_count = len(activity_chains[activity])
        print(f"  {activity}: {chain_count} chains")

    # Most common chain patterns
    print(f"\nMost common temporal chain patterns:")
    for i, (chain, frequency) in enumerate(chain_frequencies.most_common(20)):
        print(f"  {i+1:2d}. {' → '.join(chain)} (occurs {frequency} times)")

    # --- Step 5: Activity co-occurrence analysis ---
    print("\n" + "=" * 50)
    print("ACTIVITY CO-OCCURRENCE ANALYSIS")
    print("=" * 50)

    # Find what activities commonly follow each activity
    following_activities = defaultdict(Counter)

    for activity, chains in activity_chains.items():
        for chain in chains:
            for i in range(len(chain) - 1):
                current = chain[i]
                next_act = chain[i + 1]
                following_activities[current][next_act] += 1

    # Show top followers for each activity
    print("\nActivities that commonly follow each activity:")
    for activity in sorted(following_activities.keys()):
        followers = following_activities[activity]
        total_followers = sum(followers.values())
        print(f"\n{activity} ({total_followers} total transitions):")

        for next_activity, count in followers.most_common(5):
            percentage = (count / total_followers) * 100
            print(f"  → {next_activity}: {count} times ({percentage:.1f}%)")

    # --- Step 6: Temporal patterns analysis ---
    print("\n" + "=" * 50)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("=" * 50)

    # Analyze timing between activities
    transition_times = defaultdict(list)

    for activity, chains in activity_chains.items():
        for chain in chains:
            if len(chain) >= 2:
                # Find the actual activities in sorted_activities to get timing
                for i, current_act in enumerate(sorted_activities[:-1]):
                    if current_act["activity"] == chain[0]:
                        # Look for the next activity in the chain
                        for j in range(i + 1, len(sorted_activities)):
                            next_act = sorted_activities[j]
                            if (
                                next_act["activity"] == chain[1]
                                and next_act["user"] == current_act["user"]
                            ):

                                time_diff = (
                                    next_act["start_time"] - current_act["end_time"]
                                ).total_seconds() / 60.0
                                if time_diff <= time_window_minutes:
                                    transition_key = f"{current_act['activity']} → {next_act['activity']}"
                                    transition_times[transition_key].append(time_diff)
                                break

    print("Average time between activity transitions:")
    for transition, times in sorted(transition_times.items()):
        if len(times) >= 3:  # Only show transitions with enough data
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(
                f"  {transition}: {avg_time:.1f}min avg (range: {min_time:.1f}-{max_time:.1f}min, n={len(times)})"
            )

    # --- Step 7: Create visualizations ---
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    # Create activity transition network
    try:
        create_activity_network(following_activities, min_frequency=3)
        print("✓ Activity network diagram created")
    except Exception as e:
        print(f"✗ Failed to create network diagram: {e}")

    # Create chain length distribution
    try:
        create_chain_analysis_plots(activity_chains, chain_frequencies)
        print("✓ Chain analysis plots created")
    except Exception as e:
        print(f"✗ Failed to create chain plots: {e}")

    # Create temporal heatmap
    try:
        create_temporal_heatmap(sorted_activities, time_window_minutes)
        print("✓ Temporal activity heatmap created")
    except Exception as e:
        print(f"✗ Failed to create temporal heatmap: {e}")

    # --- Step 8: Export results ---
    print("\n" + "=" * 50)
    print("EXPORTING RESULTS")
    print("=" * 50)

    try:
        export_chain_results(
            activity_chains, chain_frequencies, following_activities, transition_times
        )
        print("✓ Results exported to CSV files")
    except Exception as e:
        print(f"✗ Failed to export results: {e}")

    return activity_chains, chain_frequencies, following_activities, transition_times


def create_activity_network(following_activities, min_frequency=2):
    """Create a network graph showing activity transitions"""
    G = nx.DiGraph()

    # Add edges with weights
    for source_activity, followers in following_activities.items():
        for target_activity, frequency in followers.items():
            if frequency >= min_frequency:
                G.add_edge(source_activity, target_activity, weight=frequency)

    # Create visualization
    plt.figure(figsize=(16, 12))

    # Calculate layout
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Draw nodes
    node_sizes = [
        len(following_activities.get(node, [])) * 100 + 300 for node in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7
    )

    # Draw edges with varying thickness based on frequency
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [3 * (weight / max_weight) for weight in weights]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    # Add edge labels for high-frequency transitions
    edge_labels = {
        (u, v): str(G[u][v]["weight"])
        for u, v in edges
        if G[u][v]["weight"] >= min_frequency * 2
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title(
        f"Activity Transition Network\n(Minimum {min_frequency} occurrences)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("activity_transition_network.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_chain_analysis_plots(activity_chains, chain_frequencies):
    """Create plots analyzing chain patterns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Chain length distribution
    chain_lengths = [len(chain) for chain in chain_frequencies.keys()]
    ax1.hist(
        chain_lengths,
        bins=range(min(chain_lengths), max(chain_lengths) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("Chain Length")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Chain Lengths")
    ax1.grid(True, alpha=0.3)

    # 2. Activity frequency as chain starter
    starter_counts = {
        activity: len(chains) for activity, chains in activity_chains.items()
    }
    activities = list(starter_counts.keys())[:10]  # Top 10
    counts = [starter_counts[act] for act in activities]

    ax2.barh(range(len(activities)), counts)
    ax2.set_yticks(range(len(activities)))
    ax2.set_yticklabels(activities, fontsize=8)
    ax2.set_xlabel("Number of Chains Started")
    ax2.set_title("Activities as Chain Starters")

    # 3. Most frequent chain patterns
    top_chains = list(chain_frequencies.most_common(10))
    chain_labels = [
        " → ".join(chain[:3]) + ("..." if len(chain) > 3 else "")
        for chain, _ in top_chains
    ]
    chain_counts = [count for _, count in top_chains]

    ax3.barh(range(len(chain_labels)), chain_counts)
    ax3.set_yticks(range(len(chain_labels)))
    ax3.set_yticklabels(chain_labels, fontsize=8)
    ax3.set_xlabel("Frequency")
    ax3.set_title("Most Common Chain Patterns")

    # 4. Chain frequency distribution
    frequencies = list(chain_frequencies.values())
    ax4.hist(frequencies, bins=20, alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Chain Frequency")
    ax4.set_ylabel("Number of Unique Chains")
    ax4.set_title("Distribution of Chain Frequencies")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chain_analysis_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_temporal_heatmap(sorted_activities, time_window_minutes):
    """Create a heatmap showing when activities typically occur"""
    # Extract hour and day of week from activities
    activity_times = defaultdict(lambda: defaultdict(int))

    for activity_record in sorted_activities:
        activity = activity_record["activity"]
        timestamp = activity_record["start_time"]
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday

        activity_times[activity][(day_of_week, hour)] += 1

    # Create heatmap for top activities
    top_activities = sorted(
        activity_times.keys(),
        key=lambda x: sum(activity_times[x].values()),
        reverse=True,
    )[:8]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))

    for i, activity in enumerate(top_activities):
        if i >= 8:
            break

        # Create matrix for heatmap
        matrix = [
            [activity_times[activity][(day, hour)] for hour in hours]
            for day in range(7)
        ]

        im = axes[i].imshow(matrix, cmap="YlOrRd", aspect="auto")
        axes[i].set_title(activity, fontsize=10)
        axes[i].set_xlabel("Hour of Day")
        axes[i].set_ylabel("Day of Week")
        axes[i].set_xticks(range(0, 24, 4))
        axes[i].set_xticklabels(range(0, 24, 4))
        axes[i].set_yticks(range(7))
        axes[i].set_yticklabels(days)

        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle("Activity Temporal Patterns (Day of Week vs Hour of Day)", fontsize=16)
    plt.tight_layout()
    plt.savefig("temporal_activity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()


def export_chain_results(
    activity_chains, chain_frequencies, following_activities, transition_times
):
    """Export analysis results to CSV files"""

    # 1. Export chain patterns
    chain_data = []
    for chain, frequency in chain_frequencies.most_common():
        chain_data.append(
            {
                "chain": " → ".join(chain),
                "length": len(chain),
                "frequency": frequency,
                "first_activity": chain[0],
                "last_activity": chain[-1],
            }
        )

    pd.DataFrame(chain_data).to_csv("temporal_chains.csv", index=False)

    # 2. Export activity transitions
    transition_data = []
    for source_activity, followers in following_activities.items():
        for target_activity, frequency in followers.items():
            transition_data.append(
                {
                    "source_activity": source_activity,
                    "target_activity": target_activity,
                    "frequency": frequency,
                    "percentage": (frequency / sum(followers.values())) * 100,
                }
            )

    pd.DataFrame(transition_data).to_csv("activity_transitions.csv", index=False)

    # 3. Export timing analysis
    timing_data = []
    for transition, times in transition_times.items():
        if len(times) >= 2:
            timing_data.append(
                {
                    "transition": transition,
                    "avg_minutes": sum(times) / len(times),
                    "min_minutes": min(times),
                    "max_minutes": max(times),
                    "count": len(times),
                }
            )

    pd.DataFrame(timing_data).to_csv("transition_timings.csv", index=False)

    print("Exported files:")
    print("  - temporal_chains.csv: Chain patterns and frequencies")
    print("  - activity_transitions.csv: Activity transition probabilities")
    print("  - transition_timings.csv: Timing between activity transitions")


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34

    # Analyze temporal activity chains
    activity_chains, chain_frequencies, following_activities, transition_times = (
        analyze_temporal_activity_chains(
            DATA_PREFIX,
            NUM_FOLDERS,
            time_window_minutes=10,  # Look for activities within 10 minutes
            min_chain_length=2,  # Minimum chain length to consider
            max_chain_length=10,  # Maximum chain length to build
        )
    )

    # Additional analysis functions
    print("\n" + "=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)

    # Show some specific examples
    print("\nExample chains for 'using_flexpendant_mounted':")
    if "using_flexpendant_mounted" in activity_chains:
        for i, chain in enumerate(activity_chains["using_flexpendant_mounted"][:5]):
            print(f"  {i+1}. {' → '.join(chain)}")

    print("\nExample chains for 'adjusting_tool':")
    if "adjusting_tool" in activity_chains:
        for i, chain in enumerate(activity_chains["adjusting_tool"][:5]):
            print(f"  {i+1}. {' → '.join(chain)}")
