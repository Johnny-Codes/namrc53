"""
TLDR: use reba_test_bu.py

Of course. Here is a compare and contrast analysis of the ergonomics-analysis-script currently in your canvas and the new code you provided.

Overall, both scripts share the same goal of automating REBA scoring from your dataset and use the same foundational libraries and data. However, they differ significantly in their completeness, calculation methods, and robustness. The script in your canvas is a more complete and reliable implementation, while the new code is a good conceptual start but is unfinished and makes some risky assumptions.

Similarities
Both scripts are built on the same core principles:

Foundation: They both use Python with numpy for mathematical calculations and pandas for reporting.

Configuration: They use the exact same proxy dictionaries (ACTION_FORCE_PROXY, ACTION_COUPLING_PROXY) to estimate force and grip quality from the action_label.

REBA Tables: Both correctly implement the standard REBA lookup tables (Table A, Table B, and Table C) for score aggregation.

Data Handling: They both process the data by first reading all frames, grouping them into sequences by action_number, and then calculating an average score for each action_label.

Key Differences
The differences lie in the details of the implementation, which have a significant impact on the accuracy and reliability of the results.

1. Completeness of the REBA Assessment (Major Difference)
This is the most critical distinction between the two scripts.

Your Canvas Script: Implements a complete scoring logic for all major body parts required by REBA: trunk, neck, both legs, both upper arms, both lower arms, and both wrists.

New Script: Is incomplete. It explicitly uses a placeholder for the leg score (leg_score = 1 # Placeholder - needs proper implementation) and the wrist score. This means it cannot produce a valid final REBA score, as it's missing key inputs.

2. Angle Calculation Methodology (Technical Difference)
The scripts use fundamentally different approaches to calculate posture angles.

Your Canvas Script: Calculates angles based on the geometric relationship between three joints. For example, the elbow angle is found using the 3D coordinates of the shoulder, elbow, and wrist. This method is robust because it measures the body's posture relative to itself, regardless of camera orientation.

New Script: Calculates angles by creating a vector for a body part (e.g., trunk_vector = spine_chest - pelvis) and measuring its angle against a fixed universal axis (e.g., vertical_up = np.array([0, 0, 1])). This method is valid only if the camera's coordinate system is perfectly aligned with the world, where the Z-axis is always "up". If the camera is tilted, these angle calculations will be incorrect.

3. Error and Data Quality Handling (Robustness Difference)
The scripts handle missing or invalid skeleton data differently.

Your Canvas Script: If a required joint for a calculation is missing in a frame, the script skips that entire frame from the analysis. It then reports the total number of skipped frames at the end, providing transparency about data quality.

New Script: Uses a broad try...except block. If it fails to calculate the arm scores (Group B), it returns a partial score based only on the trunk and neck (Group A). This is risky because it produces an artificially low and incorrect REBA score for that frame instead of discarding the bad data.

Conclusion
Feature	Your Canvas Script (ergonomics-analysis-script)	New Script (from Claude Sonnet 4.0)
Overall Status	✅ Complete & Functional	🚧 Incomplete & Conceptual
Leg & Wrist Scoring	✅ Fully implemented for both sides.	❌ Missing; uses placeholders.
Angle Calculation	✅ Robust; relative to body joints.	⚠️ Risky; relative to a fixed world axis.
Error Handling	✅ Safe; skips bad frames.	⚠️ Unsafe; returns partial, incorrect scores.
Reporting	✅ Provides detailed statistics (avg, std dev, etc.) and data quality summary.	✅ Provides a similar final report structure.
"""

import json
import os
from collections import defaultdict
import numpy as np
from datetime import datetime

# --- Configuration ---
# This dictionary serves as the proxy for force/load based on the action label.
# Scores are based on REBA guidelines: 0 (<5kg), 1 (5-10kg), 2 (>10kg).
ACTION_FORCE_PROXY = {
    "grinding_buildplate": 2,
    "refit_buildplate": 2,
    "preparing_buildplate": 1,
    "donning_ppe": 1,
    "doffing_ppe": 1,
    "adjusting_tool": 1,
    "wiring": 1,
    "turning_gas_knobs": 1,
    "open_door": 1,
    "close_door": 1,
    "using_flexpendant_mobile": 0,
    "using_control_panel": 0,
    "using_flexpendant_mounted": 0,
    "inspecting_buildplate": 0,
    "toggle_lights": 0,
    "observing": 0,
    "walking": 0,
    "unknown": 0,
}

# Proxy for grip/coupling quality. 0=Good, 1=Fair, 2=Poor
ACTION_COUPLING_PROXY = {
    "grinding_buildplate": 2,
    "refit_buildplate": 2,
    "adjusting_tool": 1,
    "wiring": 1,
    "turning_gas_knobs": 1,
    "open_door": 1,
    "close_door": 1,
    "preparing_buildplate": 1,
    "donning_ppe": 0,
    "doffing_ppe": 0,
    "using_flexpendant_mobile": 0,
    "using_control_panel": 0,
    "using_flexpendant_mounted": 0,
    "inspecting_buildplate": 0,
    "toggle_lights": 0,
    "observing": 0,
    "walking": 0,
    "unknown": 0,
}

# Threshold in seconds for an action to be considered "sustained" or "repetitive",
# which adds +1 to the final REBA score.
SUSTAINED_ACTION_THRESHOLD_SECONDS = 60

# --- Joint Mapping ---
JOINT_NAMES = [
    "PELVIS",
    "SPINE_NAVAL",
    "SPINE_CHEST",
    "NECK",
    "CLAVICLE_LEFT",
    "SHOULDER_LEFT",
    "ELBOW_LEFT",
    "WRIST_LEFT",
    "HAND_LEFT",
    "HANDTIP_LEFT",
    "THUMB_LEFT",
    "CLAVICLE_RIGHT",
    "SHOULDER_RIGHT",
    "ELBOW_RIGHT",
    "WRIST_RIGHT",
    "HAND_RIGHT",
    "HANDTIP_RIGHT",
    "THUMB_RIGHT",
    "HIP_LEFT",
    "KNEE_LEFT",
    "ANKLE_LEFT",
    "FOOT_LEFT",
    "HIP_RIGHT",
    "KNEE_RIGHT",
    "ANKLE_RIGHT",
    "FOOT_RIGHT",
    "HEAD",
    "NOSE",
    "EYE_LEFT",
    "EAR_LEFT",
    "EYE_RIGHT",
    "EAR_RIGHT",
]

# --- REBA Scoring Tables ---
TABLE_A = [
    [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[6, 7, 8, 9], [7, 8, 9, 9], [8, 9, 9, 9]],
]
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[7, 8, 8], [8, 9, 9]],
]
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0
    value = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.degrees(np.arccos(value))


def get_risk_level_from_score(score):
    score = round(score)
    if score == 1:
        return "Negligible"
    if 2 <= score <= 3:
        return "Low"
    if 4 <= score <= 7:
        return "Medium"
    if 8 <= score <= 10:
        return "High"
    if score >= 11:
        return "Very High"
    return "Unknown"


def get_reba_score(skeleton_data, action_label):
    pelvis = np.array(skeleton_data["PELVIS"])
    spine_chest = np.array(skeleton_data["SPINE_CHEST"])
    neck = np.array(skeleton_data["NECK"])
    head = np.array(skeleton_data["HEAD"])

    # Assuming Z is vertical (adjust if Y is vertical in your system)
    vertical_up = np.array([0, 0, 1])

    # Trunk flexion from vertical
    trunk_vector = spine_chest - pelvis
    trunk_flexion = calculate_angle([0, 0, 0], trunk_vector, vertical_up)

    # REBA trunk scoring
    trunk_score = 1
    if trunk_flexion <= 20:
        trunk_score = 1
    elif 20 < trunk_flexion <= 60:
        trunk_score = 3
    elif trunk_flexion > 60:
        trunk_score = 4
    # TODO: Add +1 for lateral bending or twisting

    # Neck flexion
    neck_vector = head - neck
    spine_vector = spine_chest - neck
    neck_flexion = calculate_angle([0, 0, 0], neck_vector, spine_vector)

    neck_score = 1
    if neck_flexion > 20:
        neck_score = 2
    # TODO: Add +1 for lateral bending

    # Legs - need better logic for REBA leg assessment
    # Current knee angle approach is oversimplified
    leg_score = 1  # Placeholder - needs proper implementation

    # Calculate posture score A
    posture_score_a = TABLE_A[trunk_score - 1][neck_score - 1][leg_score - 1]
    force_score = ACTION_FORCE_PROXY.get(action_label, 0)
    score_a = posture_score_a + force_score

    # Arms calculation - need to implement the full calculation
    # For now, let's add a basic arm calculation
    try:
        # Upper arm calculation
        shoulder_left = np.array(skeleton_data["SHOULDER_LEFT"])
        elbow_left = np.array(skeleton_data["ELBOW_LEFT"])
        shoulder_right = np.array(skeleton_data["SHOULDER_RIGHT"])
        elbow_right = np.array(skeleton_data["ELBOW_RIGHT"])

        # Calculate upper arm angles (simplified)
        upper_arm_left = elbow_left - shoulder_left
        upper_arm_right = elbow_right - shoulder_right

        # Use the worst case (higher angle) for scoring
        left_arm_angle = calculate_angle([0, 0, 0], upper_arm_left, vertical_up)
        right_arm_angle = calculate_angle([0, 0, 0], upper_arm_right, vertical_up)
        upper_arm_angle = max(left_arm_angle, right_arm_angle)

        # Upper arm scoring
        upper_arm_score = 1
        if upper_arm_angle > 20:
            upper_arm_score = 2
        if upper_arm_angle > 45:
            upper_arm_score = 3
        if upper_arm_angle > 90:
            upper_arm_score = 4

        # Lower arm calculation (simplified)
        wrist_left = np.array(skeleton_data["WRIST_LEFT"])
        wrist_right = np.array(skeleton_data["WRIST_RIGHT"])

        lower_arm_left = wrist_left - elbow_left
        lower_arm_right = wrist_right - elbow_right

        # Lower arm angle (simplified - should be measured differently)
        left_lower_angle = calculate_angle([0, 0, 0], lower_arm_left, upper_arm_left)
        right_lower_angle = calculate_angle([0, 0, 0], lower_arm_right, upper_arm_right)
        lower_arm_angle = max(left_lower_angle, right_lower_angle)

        # Lower arm scoring
        lower_arm_score = 1
        if lower_arm_angle < 60 or lower_arm_angle > 100:
            lower_arm_score = 2

        # Wrist scoring (simplified)
        wrist_score = 1  # Placeholder

        # Calculate posture score B
        posture_score_b = TABLE_B[upper_arm_score - 1][lower_arm_score - 1][
            wrist_score - 1
        ]
        coupling_score = ACTION_COUPLING_PROXY.get(action_label, 0)
        score_b = posture_score_b + coupling_score

        # Final REBA score from Table C
        final_score = TABLE_C[score_a - 1][score_b - 1]

        return final_score

    except (KeyError, IndexError) as e:
        # If we can't calculate arm scores, return a basic score
        return score_a


def analyze_ergonomics(
    data_directory_prefix, num_folders, output_csv_file="ergonomic_analysis_report.csv"
):
    try:
        import pandas as pd

        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False

    print("Reading all metadata files...")
    all_frames = []
    found_action_labels = set()
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
                    data = json.loads(line.strip())
                    all_frames.append(data)
                    found_action_labels.add(data.get("action_label", "unknown"))
                except json.JSONDecodeError:
                    continue

    # --- Data Coverage Check ---
    print("\n--- Checking Data Coverage ---")
    proxy_actions = set(ACTION_FORCE_PROXY.keys())
    missing_actions = found_action_labels - proxy_actions
    if missing_actions:
        print(
            f"Warning: The following {len(missing_actions)} actions were found in the data but are MISSING from the proxy dictionaries:"
        )
        for action in sorted(list(missing_actions)):
            print(f"  - {action}")
        print(
            "These will default to zero force/coupling, potentially underestimating risk.\n"
        )
    else:
        print("All found action labels are covered in the proxy dictionaries.\n")

    print("Grouping frames by action sequence...")
    action_sequences = defaultdict(list)
    for frame in all_frames:
        action_num = frame.get("action_number")
        if action_num is not None:
            action_sequences[action_num].append(frame)

    print("\nCalculating average REBA score for each action type...")
    action_risk_scores = defaultdict(list)
    total_frames_processed = len(all_frames)
    skipped_frames_count = 0

    for seq_id, frames in action_sequences.items():
        if not frames:
            continue

        action_label = frames[0].get("action_label", "unknown")
        total_reba_score = 0
        valid_frames = 0

        start_time = datetime.fromisoformat(frames[0]["datetime"])
        end_time = datetime.fromisoformat(frames[-1]["datetime"])
        duration_seconds = (end_time - start_time).total_seconds()

        for frame in frames:
            skeleton_coords = frame.get("skeleton")
            if not skeleton_coords or len(skeleton_coords) != len(JOINT_NAMES):
                skipped_frames_count += 1
                continue

            skeleton_dict = dict(
                zip(JOINT_NAMES, [list(map(float, coord)) for coord in skeleton_coords])
            )
            try:
                reba_score = get_reba_score(skeleton_dict, action_label)
                total_reba_score += reba_score
                valid_frames += 1
            except (KeyError, IndexError):
                skipped_frames_count += 1
                pass

        if valid_frames > 0:
            avg_reba_score = total_reba_score / valid_frames
            if duration_seconds > SUSTAINED_ACTION_THRESHOLD_SECONDS:
                avg_reba_score += 1
            action_risk_scores[action_label].append(avg_reba_score)

    # --- Print Summary Report ---
    print("\n" + "=" * 80)
    print(" " * 25 + "ERGONOMIC ANALYSIS REPORT (REBA)")
    print("=" * 80)

    report_data = []
    for action, scores in sorted(action_risk_scores.items()):
        avg_score = np.mean(scores)
        risk_level = get_risk_level_from_score(avg_score)

        report_data.append(
            {
                "Action Label": action,
                "Avg Score": f"{avg_score:.2f}",
                "Std Dev": f"{np.std(scores):.2f}",
                "Min Score": f"{np.min(scores):.2f}",
                "Max Score": f"{np.max(scores):.2f}",
                "Risk Level": risk_level,
                "Sequences Analyzed": len(scores),
            }
        )

    if PANDAS_AVAILABLE and report_data:
        df = pd.DataFrame(report_data)
        print(df.to_string(index=False))
        # --- Save to CSV ---
        try:
            df.to_csv(output_csv_file, index=False)
            print(f"\nReport successfully saved to '{output_csv_file}'")
        except Exception as e:
            print(f"\nError saving report to CSV: {e}")

    elif report_data:
        for item in report_data:
            print(
                f"{item['Action Label']:<25} | Avg Score: {item['Avg Score']:<6} | Risk: {item['Risk Level']:<10} | Sequences: {item['Sequences Analyzed']}"
            )

    print("=" * 80)

    # --- Data Quality Summary ---
    print("\n--- Data Quality Summary ---")
    if total_frames_processed > 0:
        skipped_percentage = (skipped_frames_count / total_frames_processed) * 100
        print(f"Total frames read: {total_frames_processed}")
        print(
            f"Frames skipped (invalid/missing skeleton): {skipped_frames_count} ({skipped_percentage:.2f}%)"
        )
    else:
        print("No frames were processed.")
    print("=" * 80)


# --- Run the Analysis ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34
    analyze_ergonomics(DATA_PREFIX, NUM_FOLDERS)
