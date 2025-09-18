import json
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_and_categorize_activities(data_dir="./data"):
    """Load activity data and categorize by camera location"""
    print("Loading and categorizing activity data by camera location...")
    print(f"Looking for training data in: {data_dir}")

    inner_activities = []
    outer_activities = []

    # Convert to Path object
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist!")
        return inner_activities, outer_activities

    # Process all train directories in the data folder
    train_dirs = sorted(data_path.glob("train-*"))

    if not train_dirs:
        print(f"No train-* directories found in {data_dir}")
        print("Available directories:")
        for item in data_path.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return inner_activities, outer_activities

    print(f"Found {len(train_dirs)} training directories")

    for train_dir in train_dirs:
        if train_dir.is_dir():
            print(f"Processing {train_dir.name}...")

            # Load metadata for this train directory
            metadata_file = train_dir / "metadata.jsonl"
            if not metadata_file.exists():
                print(f"  No metadata file found for {train_dir.name}")
                continue

            # Count records processed
            records_processed = 0
            inner_count = 0
            outer_count = 0

            # Read metadata line by line
            with open(metadata_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records_processed += 1

                        # Check if this record has the required fields
                        if "file_name" not in record:
                            continue

                        file_name = record["file_name"]

                        # Determine camera type from file_name
                        if file_name.startswith("inner_depths/"):
                            camera_type = "inner"
                            target_list = inner_activities
                            inner_count += 1
                        elif file_name.startswith("outer_depths/"):
                            camera_type = "outer"
                            target_list = outer_activities
                            outer_count += 1
                        else:
                            continue  # Skip files that don't match our camera types

                        # Add train directory and camera type to record
                        record["train_dir"] = train_dir.name
                        record["camera_type"] = camera_type
                        record["full_path"] = str(train_dir / file_name)

                        target_list.append(record)

                    except json.JSONDecodeError as e:
                        print(
                            f"  Error parsing line {line_num} in {metadata_file}: {e}"
                        )
                        continue
                    except Exception as e:
                        print(
                            f"  Error processing line {line_num} in {metadata_file}: {e}"
                        )
                        continue

            print(
                f"  Processed {records_processed} records: {inner_count} inner, {outer_count} outer"
            )

    print(f"\nLoaded {len(inner_activities)} inner camera activities")
    print(f"Loaded {len(outer_activities)} outer camera activities")

    # Debug info
    if inner_activities:
        print(f"Sample inner activity: {inner_activities[0].get('file_name', 'N/A')}")
        print(f"Sample inner train_dir: {inner_activities[0].get('train_dir', 'N/A')}")
    if outer_activities:
        print(f"Sample outer activity: {outer_activities[0].get('file_name', 'N/A')}")
        print(f"Sample outer train_dir: {outer_activities[0].get('train_dir', 'N/A')}")

    return inner_activities, outer_activities


def create_temporal_chains(
    activities, time_window_minutes=15, min_chain_length=2, max_chain_length=10
):
    """Create temporal activity chains from camera-specific activities"""
    if not activities:
        print("No activities provided for chain creation")
        return []

    print(f"Creating temporal chains from {len(activities)} activities...")

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(activities)

    # Check if we have the required columns
    required_cols = ["action_number", "file_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return []

    # Sort by action_number (assuming this represents temporal order)
    df = df.sort_values("action_number").reset_index(drop=True)

    chains = []
    time_window_actions = time_window_minutes  # Using action_number as proxy for time

    for i in range(len(df)):
        current_action = df.iloc[i]["action_number"]

        # Find activities within the time window
        window_end = current_action + time_window_actions
        window_activities = df[
            (df["action_number"] >= current_action)
            & (df["action_number"] <= window_end)
        ]

        if len(window_activities) >= min_chain_length:
            # Limit chain length
            if len(window_activities) > max_chain_length:
                window_activities = window_activities.head(max_chain_length)

            # Create chain
            chain = {
                "start_action": current_action,
                "end_action": window_activities.iloc[-1]["action_number"],
                "length": len(window_activities),
                "activities": window_activities["file_name"].tolist(),
                "action_numbers": window_activities["action_number"].tolist(),
                "train_dirs": (
                    window_activities["train_dir"].tolist()
                    if "train_dir" in window_activities.columns
                    else []
                ),
                "action_labels": (
                    window_activities["action_label"].tolist()
                    if "action_label" in window_activities.columns
                    else []
                ),
            }
            chains.append(chain)

    print(f"Created {len(chains)} temporal chains")
    return chains


def analyze_chain_patterns(chains, camera_type):
    """Analyze patterns in temporal chains"""
    if not chains:
        return {
            "camera_type": camera_type,
            "total_chains": 0,
            "avg_chain_length": 0,
            "length_distribution": Counter(),
            "top_patterns": [],
            "avg_duration": 0,
            "duration_distribution": [],
        }

    # Chain length distribution
    lengths = [chain["length"] for chain in chains]

    # Activity sequence patterns based on action labels (more meaningful than image numbers)
    sequence_patterns = Counter()
    action_label_patterns = Counter()

    for chain in chains:
        # Pattern based on action labels if available
        if chain.get("action_labels"):
            action_sequence = chain["action_labels"][
                :5
            ]  # Limit to first 5 for readability
            # Use ASCII arrow for Windows compatibility
            action_pattern = " -> ".join(action_sequence)
            action_label_patterns[action_pattern] += 1

        # Pattern based on image numbers (fallback)
        sequence = []
        for activity in chain["activities"]:
            # Extract number from file like "inner_depths/002384.png"
            try:
                img_num = activity.split("/")[-1].split(".")[0]
                sequence.append(img_num)
            except:
                sequence.append("unknown")

        if len(sequence) >= 2:
            # Use ASCII arrow for Windows compatibility
            pattern = " -> ".join(sequence[:5])  # Limit pattern length for readability
            sequence_patterns[pattern] += 1

    # Duration analysis (using action number differences)
    durations = []
    for chain in chains:
        if len(chain["action_numbers"]) >= 2:
            duration = chain["action_numbers"][-1] - chain["action_numbers"][0]
            durations.append(duration)

    analysis = {
        "camera_type": camera_type,
        "total_chains": len(chains),
        "avg_chain_length": np.mean(lengths) if lengths else 0,
        "length_distribution": Counter(lengths),
        "top_patterns": (
            action_label_patterns.most_common(10)
            if action_label_patterns
            else sequence_patterns.most_common(10)
        ),
        "top_image_patterns": sequence_patterns.most_common(10),
        "avg_duration": np.mean(durations) if durations else 0,
        "duration_distribution": durations,
    }

    return analysis


def compare_camera_analyses(inner_analysis, outer_analysis):
    """Compare analyses between inner and outer cameras"""
    comparison = {
        "chain_count_ratio": 0,
        "avg_length_difference": 0,
        "pattern_similarity": 0,
        "duration_difference": 0,
    }

    if (
        inner_analysis.get("total_chains", 0) > 0
        and outer_analysis.get("total_chains", 0) > 0
    ):
        comparison["chain_count_ratio"] = (
            inner_analysis["total_chains"] / outer_analysis["total_chains"]
        )
        comparison["avg_length_difference"] = (
            inner_analysis["avg_chain_length"] - outer_analysis["avg_chain_length"]
        )
        comparison["duration_difference"] = (
            inner_analysis["avg_duration"] - outer_analysis["avg_duration"]
        )

        # Simple pattern similarity (count of common pattern prefixes)
        inner_patterns = set()
        outer_patterns = set()

        for pattern, _ in inner_analysis["top_patterns"]:
            if " -> " in pattern:  # Changed from "→" to "->"
                inner_patterns.add(pattern.split(" -> ")[0])
            else:
                inner_patterns.add(pattern)

        for pattern, _ in outer_analysis["top_patterns"]:
            if " -> " in pattern:  # Changed from "→" to "->"
                outer_patterns.add(pattern.split(" -> ")[0])
            else:
                outer_patterns.add(pattern)

        if inner_patterns or outer_patterns:
            comparison["pattern_similarity"] = len(
                inner_patterns.intersection(outer_patterns)
            ) / len(inner_patterns.union(outer_patterns))

    return comparison


def create_visualizations(inner_analysis, outer_analysis, comparison, output_dir):
    """Create visualization plots"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Chain Length Distribution Comparison
    if inner_analysis.get("length_distribution") or outer_analysis.get(
        "length_distribution"
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Inner camera
        if inner_analysis.get("length_distribution"):
            lengths = list(inner_analysis["length_distribution"].keys())
            counts = list(inner_analysis["length_distribution"].values())
            ax1.bar(lengths, counts, alpha=0.7, color="blue")
        ax1.set_title("Inner Camera: Chain Length Distribution")
        ax1.set_xlabel("Chain Length")
        ax1.set_ylabel("Frequency")

        # Outer camera
        if outer_analysis.get("length_distribution"):
            lengths = list(outer_analysis["length_distribution"].keys())
            counts = list(outer_analysis["length_distribution"].values())
            ax2.bar(lengths, counts, alpha=0.7, color="red")
        ax2.set_title("Outer Camera: Chain Length Distribution")
        ax2.set_xlabel("Chain Length")
        ax2.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(
            plots_dir / "chain_length_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 2. Duration Distribution Comparison
    if inner_analysis.get("duration_distribution") or outer_analysis.get(
        "duration_distribution"
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        if inner_analysis.get("duration_distribution"):
            ax1.hist(
                inner_analysis["duration_distribution"],
                bins=20,
                alpha=0.7,
                color="blue",
            )
        ax1.set_title("Inner Camera: Duration Distribution")
        ax1.set_xlabel("Duration (action steps)")
        ax1.set_ylabel("Frequency")

        if outer_analysis.get("duration_distribution"):
            ax2.hist(
                outer_analysis["duration_distribution"], bins=20, alpha=0.7, color="red"
            )
        ax2.set_title("Outer Camera: Duration Distribution")
        ax2.set_xlabel("Duration (action steps)")
        ax2.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(
            plots_dir / "duration_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # 3. Top Patterns Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Inner camera patterns
    if inner_analysis.get("top_patterns"):
        patterns = [p[0] for p in inner_analysis["top_patterns"][:10]]
        counts = [p[1] for p in inner_analysis["top_patterns"][:10]]

        # Truncate long pattern names for display
        display_patterns = [p[:50] + "..." if len(p) > 50 else p for p in patterns]

        y_pos = np.arange(len(display_patterns))
        ax1.barh(y_pos, counts, alpha=0.7, color="blue")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(display_patterns, fontsize=8)
        ax1.set_xlabel("Frequency")
        ax1.set_title("Inner Camera: Top Activity Patterns")

    # Outer camera patterns
    if outer_analysis.get("top_patterns"):
        patterns = [p[0] for p in outer_analysis["top_patterns"][:10]]
        counts = [p[1] for p in outer_analysis["top_patterns"][:10]]

        # Truncate long pattern names for display
        display_patterns = [p[:50] + "..." if len(p) > 50 else p for p in patterns]

        y_pos = np.arange(len(display_patterns))
        ax2.barh(y_pos, counts, alpha=0.7, color="red")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(display_patterns, fontsize=8)
        ax2.set_xlabel("Frequency")
        ax2.set_title("Outer Camera: Top Activity Patterns")

    plt.tight_layout()
    plt.savefig(plots_dir / "top_patterns_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Comparison Summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Chain counts
    cameras = ["Inner", "Outer"]
    chain_counts = [
        inner_analysis.get("total_chains", 0),
        outer_analysis.get("total_chains", 0),
    ]
    ax1.bar(cameras, chain_counts, color=["blue", "red"], alpha=0.7)
    ax1.set_title("Total Chain Counts")
    ax1.set_ylabel("Number of Chains")

    # Average chain lengths
    avg_lengths = [
        inner_analysis.get("avg_chain_length", 0),
        outer_analysis.get("avg_chain_length", 0),
    ]
    ax2.bar(cameras, avg_lengths, color=["blue", "red"], alpha=0.7)
    ax2.set_title("Average Chain Length")
    ax2.set_ylabel("Average Length")

    # Average durations
    avg_durations = [
        inner_analysis.get("avg_duration", 0),
        outer_analysis.get("avg_duration", 0),
    ]
    ax3.bar(cameras, avg_durations, color=["blue", "red"], alpha=0.7)
    ax3.set_title("Average Chain Duration")
    ax3.set_ylabel("Average Duration (action steps)")

    # Comparison metrics
    metrics = ["Chain Ratio", "Length Diff", "Pattern Sim", "Duration Diff"]
    values = [
        comparison.get("chain_count_ratio", 0),
        comparison.get("avg_length_difference", 0),
        comparison.get("pattern_similarity", 0),
        comparison.get("duration_difference", 0),
    ]
    bars = ax4.bar(metrics, values, color="green", alpha=0.7)
    ax4.set_title("Comparison Metrics")
    ax4.set_ylabel("Value")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualizations saved to {plots_dir}")


def save_detailed_report(inner_analysis, outer_analysis, comparison, output_dir):
    """Save detailed analysis report"""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / "temporal_chain_analysis_report.txt"

    with open(report_file, "w", encoding="utf-8") as f:  # Add UTF-8 encoding
        f.write("CAMERA-BASED TEMPORAL ACTIVITY CHAIN ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Inner Camera Analysis
        f.write("INNER CAMERA ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Chains: {inner_analysis.get('total_chains', 0)}\n")
        f.write(
            f"Average Chain Length: {inner_analysis.get('avg_chain_length', 0):.2f}\n"
        )
        f.write(
            f"Average Duration: {inner_analysis.get('avg_duration', 0):.2f} action steps\n\n"
        )

        if inner_analysis.get("top_patterns"):
            f.write("Top Activity Patterns:\n")
            for i, (pattern, count) in enumerate(inner_analysis["top_patterns"][:5], 1):
                # Replace arrow character with ASCII version for Windows compatibility
                pattern_safe = pattern.replace("→", "->")
                f.write(f"  {i}. {pattern_safe} (occurs {count} times)\n")
        f.write("\n")

        # Outer Camera Analysis
        f.write("OUTER CAMERA ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Chains: {outer_analysis.get('total_chains', 0)}\n")
        f.write(
            f"Average Chain Length: {outer_analysis.get('avg_chain_length', 0):.2f}\n"
        )
        f.write(
            f"Average Duration: {outer_analysis.get('avg_duration', 0):.2f} action steps\n\n"
        )

        if outer_analysis.get("top_patterns"):
            f.write("Top Activity Patterns:\n")
            for i, (pattern, count) in enumerate(outer_analysis["top_patterns"][:5], 1):
                # Replace arrow character with ASCII version for Windows compatibility
                pattern_safe = pattern.replace("→", "->")
                f.write(f"  {i}. {pattern_safe} (occurs {count} times)\n")
        f.write("\n")

        # Comparison
        f.write("CAMERA COMPARISON\n")
        f.write("-" * 20 + "\n")
        f.write(
            f"Chain Count Ratio (Inner/Outer): {comparison.get('chain_count_ratio', 0):.2f}\n"
        )
        f.write(
            f"Average Length Difference: {comparison.get('avg_length_difference', 0):.2f}\n"
        )
        f.write(f"Pattern Similarity: {comparison.get('pattern_similarity', 0):.2f}\n")
        f.write(
            f"Duration Difference: {comparison.get('duration_difference', 0):.2f} action steps\n\n"
        )

        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 15 + "\n")

        ratio = comparison.get("chain_count_ratio", 0)
        if ratio > 1.2:
            f.write("- Inner camera shows significantly more activity chains\n")
        elif ratio < 0.8:
            f.write("- Outer camera shows significantly more activity chains\n")
        else:
            f.write("- Similar activity levels between cameras\n")

        length_diff = comparison.get("avg_length_difference", 0)
        if length_diff > 1:
            f.write("- Inner camera activities tend to form longer chains\n")
        elif length_diff < -1:
            f.write("- Outer camera activities tend to form longer chains\n")
        else:
            f.write("- Similar chain lengths between cameras\n")

        similarity = comparison.get("pattern_similarity", 0)
        if similarity > 0.5:
            f.write("- High pattern similarity between cameras\n")
        elif similarity > 0.2:
            f.write("- Moderate pattern similarity between cameras\n")
        else:
            f.write("- Low pattern similarity between cameras\n")

    print(f"Detailed report saved to {report_file}")


def main():
    # Configuration
    TIME_WINDOW_MINUTES = 5  # Time window for grouping activities
    MIN_CHAIN_LENGTH = 2  # Minimum activities in a chain
    MAX_CHAIN_LENGTH = 10  # Maximum activities in a chain
    DATA_DIR = "./data"  # Data directory

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"camera_temporal_analysis_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("CAMERA-BASED TEMPORAL ACTIVITY CHAIN ANALYSIS")
    print("=" * 80)
    print(f"Time window: {TIME_WINDOW_MINUTES} minutes")
    print(f"Chain length: {MIN_CHAIN_LENGTH}-{MAX_CHAIN_LENGTH} activities")
    print(f"Output directory: {output_dir}")
    print()

    # Load and categorize activities by camera
    inner_activities, outer_activities = load_and_categorize_activities(DATA_DIR)

    if not inner_activities and not outer_activities:
        print("No camera-specific activities found!")
        print("Please check your data structure and file paths.")
        return

    # Create temporal chains for each camera
    print("\nCreating temporal activity chains...")
    inner_chains = create_temporal_chains(
        inner_activities, TIME_WINDOW_MINUTES, MIN_CHAIN_LENGTH, MAX_CHAIN_LENGTH
    )
    outer_chains = create_temporal_chains(
        outer_activities, TIME_WINDOW_MINUTES, MIN_CHAIN_LENGTH, MAX_CHAIN_LENGTH
    )

    # Analyze patterns
    print("\nAnalyzing activity patterns...")
    inner_analysis = analyze_chain_patterns(inner_chains, "inner")
    outer_analysis = analyze_chain_patterns(outer_chains, "outer")

    # Compare between cameras
    comparison = compare_camera_analyses(inner_analysis, outer_analysis)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(inner_analysis, outer_analysis, comparison, output_dir)

    # Save detailed report
    print("\nSaving detailed report...")
    save_detailed_report(inner_analysis, outer_analysis, comparison, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(
        f"Inner Camera: {inner_analysis.get('total_chains', 0)} chains, avg length {inner_analysis.get('avg_chain_length', 0):.2f}"
    )
    print(
        f"Outer Camera: {outer_analysis.get('total_chains', 0)} chains, avg length {outer_analysis.get('avg_chain_length', 0):.2f}"
    )
    print(
        f"Chain Count Ratio (Inner/Outer): {comparison.get('chain_count_ratio', 0):.2f}"
    )
    print(f"Pattern Similarity: {comparison.get('pattern_similarity', 0):.2f}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
