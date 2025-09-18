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
    vertical_ref = pelvis + [0, 100, 0]
    trunk_flexion = calculate_angle(skeleton_data["SPINE_CHEST"], pelvis, vertical_ref)
    trunk_score = 1
    if 0 <= trunk_flexion <= 20:
        trunk_score = 2 if trunk_flexion > 5 else 1
    elif 20 < trunk_flexion <= 60:
        trunk_score = 3
    elif trunk_flexion > 60:
        trunk_score = 4

    neck_flexion = calculate_angle(
        skeleton_data["HEAD"], skeleton_data["NECK"], skeleton_data["SPINE_CHEST"]
    )
    neck_score = 1
    if neck_flexion > 20:
        neck_score = 2

    left_knee_angle = 180 - calculate_angle(
        skeleton_data["HIP_LEFT"],
        skeleton_data["KNEE_LEFT"],
        skeleton_data["ANKLE_LEFT"],
    )
    right_knee_angle = 180 - calculate_angle(
        skeleton_data["HIP_RIGHT"],
        skeleton_data["KNEE_RIGHT"],
        skeleton_data["ANKLE_RIGHT"],
    )
    max_knee_angle = max(left_knee_angle, right_knee_angle)
    leg_score = 1
    if 30 < max_knee_angle <= 60:
        leg_score = 2
    elif max_knee_angle > 60:
        leg_score = 3

    posture_score_a = TABLE_A[trunk_score - 1][neck_score - 1][leg_score - 1]
    force_score = ACTION_FORCE_PROXY.get(action_label, 0)
    score_a = posture_score_a + force_score

    left_upper_arm_angle = calculate_angle(
        skeleton_data["ELBOW_LEFT"],
        skeleton_data["SHOULDER_LEFT"],
        skeleton_data["SPINE_CHEST"],
    )
    right_upper_arm_angle = calculate_angle(
        skeleton_data["ELBOW_RIGHT"],
        skeleton_data["SHOULDER_RIGHT"],
        skeleton_data["SPINE_CHEST"],
    )
    max_upper_arm_angle = max(left_upper_arm_angle, right_upper_arm_angle)
    upper_arm_score = 1
    if 20 < max_upper_arm_angle <= 45:
        upper_arm_score = 2
    elif 45 < max_upper_arm_angle <= 90:
        upper_arm_score = 3
    elif max_upper_arm_angle > 90:
        upper_arm_score = 4

    left_lower_arm_angle = 180 - calculate_angle(
        skeleton_data["SHOULDER_LEFT"],
        skeleton_data["ELBOW_LEFT"],
        skeleton_data["WRIST_LEFT"],
    )
    right_lower_arm_angle = 180 - calculate_angle(
        skeleton_data["SHOULDER_RIGHT"],
        skeleton_data["ELBOW_RIGHT"],
        skeleton_data["WRIST_RIGHT"],
    )
    max_lower_arm_angle = max(left_lower_arm_angle, right_lower_arm_angle)
    lower_arm_score = 2
    if 60 <= max_lower_arm_angle <= 100:
        lower_arm_score = 1

    left_wrist_angle = 180 - calculate_angle(
        skeleton_data["ELBOW_LEFT"],
        skeleton_data["WRIST_LEFT"],
        skeleton_data["HAND_LEFT"],
    )
    right_wrist_angle = 180 - calculate_angle(
        skeleton_data["ELBOW_RIGHT"],
        skeleton_data["WRIST_RIGHT"],
        skeleton_data["HAND_RIGHT"],
    )
    max_wrist_angle = max(left_wrist_angle, right_wrist_angle)
    wrist_score = 1
    if max_wrist_angle > 15:
        wrist_score = 2

    posture_score_b = TABLE_B[upper_arm_score - 1][lower_arm_score - 1][wrist_score - 1]
    coupling_score = ACTION_COUPLING_PROXY.get(action_label, 0)
    score_b = posture_score_b + coupling_score

    score_a_clamped = min(score_a, 12)
    score_b_clamped = min(score_b, 12)
    table_c_score = TABLE_C[score_a_clamped - 1][score_b_clamped - 1]

    return table_c_score


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
