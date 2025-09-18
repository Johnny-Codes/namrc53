import json
import os
from collections import defaultdict
import random
import math


def generate_split_config(
    data_directory_prefix, num_folders, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
):
    """
    Analyzes the dataset to create a stratified group split based on action sequences.
    This prevents data leakage and handles class imbalance.

    Args:
        data_directory_prefix (str): The root path to the data folders (e.g., "./data").
        num_folders (int): The number of training folders to process (e.g., 33 for 0-32).
        train_ratio (float): The proportion of data to allocate for training.
        val_ratio (float): The proportion of data to allocate for validation.
        test_ratio (float): The proportion of data to allocate for testing.

    Returns:
        dict: A dictionary containing the lists of action sequence IDs for each set.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of train, val, and test ratios must be 1.0")

    # --- Step 1: Map each unique sequence to its action label ---
    sequence_to_label = {}
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
                    action_num = data.get("action_number")
                    action_label = data.get("action_label")
                    if action_num is not None and action_label is not None:
                        # The action_number is globally unique
                        sequence_to_label[action_num] = action_label
                except json.JSONDecodeError:
                    continue

    # --- Step 2: Group the unique sequence IDs by their action label ---
    label_to_sequences = defaultdict(list)
    for seq_id, label in sequence_to_label.items():
        label_to_sequences[label].append(seq_id)

    print("--- Sequence Count per Action ---")
    for label, seq_list in sorted(label_to_sequences.items()):
        print(f"{label:<25}: {len(seq_list)} sequences")
    print("-" * 40)

    # --- Step 3: Perform a stratified split on the sequences for each class ---
    train_seq_ids = []
    val_seq_ids = []
    test_seq_ids = []

    for label, sequences in label_to_sequences.items():
        random.shuffle(sequences)  # Shuffle to ensure random selection

        n_total = len(sequences)

        if n_total < 3:
            # If a class has too few samples, put them all in training to avoid empty sets.
            train_seq_ids.extend(sequences)
            continue

        # Guarantee at least one sample in val and test sets
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        # If training ends up with 0 or negative samples, adjust.
        if n_train <= 0:
            n_train = 1
            # Recalculate val and test based on the remainder
            n_val = max(
                1, int((n_total - n_train) * (val_ratio / (val_ratio + test_ratio)))
            )
            n_test = n_total - n_train - n_val

        # Split the list of sequences for the current class
        class_train = sequences[:n_train]
        class_val = sequences[n_train : n_train + n_val]
        class_test = sequences[n_train + n_val :]

        # Add the splits to the final lists
        train_seq_ids.extend(class_train)
        val_seq_ids.extend(class_val)
        test_seq_ids.extend(class_test)

    # --- Step 4 & 5: Finalize sets and report ---
    split_config = {
        "train_sequences": sorted(train_seq_ids),
        "validation_sequences": sorted(val_seq_ids),
        "test_sequences": sorted(test_seq_ids),
    }

    print("\n--- Split Summary ---")
    print(f"Total Sequences: {len(sequence_to_label)}")
    print(f"Training Sequences:   {len(split_config['train_sequences'])}")
    print(f"Validation Sequences: {len(split_config['validation_sequences'])}")
    print(f"Test Sequences:       {len(split_config['test_sequences'])}")

    # Save the configuration to a file
    output_path = "split_config.json"
    with open(output_path, "w") as f:
        json.dump(split_config, f, indent=4)

    print(f"\nSuccessfully created data split configuration at: {output_path}")

    return split_config, sequence_to_label


def verify_split_distribution(split_config, sequence_to_label):
    """
    Verifies and prints the distribution of action classes across the splits.
    """
    print("\n" + "=" * 60)
    print(" " * 12 + "VERIFICATION OF SPLIT DISTRIBUTION")
    print("=" * 60)

    # Invert the map for easy lookup
    all_labels = sorted(list(set(sequence_to_label.values())))

    distribution = {
        label: {"train": 0, "validation": 0, "test": 0, "total": 0}
        for label in all_labels
    }

    for seq_id in split_config["train_sequences"]:
        label = sequence_to_label.get(seq_id)
        if label:
            distribution[label]["train"] += 1

    for seq_id in split_config["validation_sequences"]:
        label = sequence_to_label.get(seq_id)
        if label:
            distribution[label]["validation"] += 1

    for seq_id in split_config["test_sequences"]:
        label = sequence_to_label.get(seq_id)
        if label:
            distribution[label]["test"] += 1

    # Calculate totals
    total_train = 0
    total_val = 0
    total_test = 0
    grand_total = 0

    for label, counts in distribution.items():
        counts["total"] = counts["train"] + counts["validation"] + counts["test"]
        total_train += counts["train"]
        total_val += counts["validation"]
        total_test += counts["test"]
        grand_total += counts["total"]

    # Print the distribution table
    print(
        f"{'Action Label':<25} | {'Train':>10} | {'Validation':>12} | {'Test':>10} | {'Total':>10}"
    )
    print("-" * 78)
    for label, counts in distribution.items():
        print(
            f"{label:<25} | {counts['train']:>10} | {counts['validation']:>12} | {counts['test']:>10} | {counts['total']:>10}"
        )

    # Print the total row
    print("-" * 78)
    print(
        f"{'Total':<25} | {total_train:>10} | {total_val:>12} | {total_test:>10} | {grand_total:>10}"
    )
    print("=" * 60)


# --- Run the Splitting Process and Verification ---
if __name__ == "__main__":
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34  # Process train-000 to train-032

    # Generate the split
    final_split_config, seq_to_label_map = generate_split_config(
        DATA_PREFIX, NUM_FOLDERS
    )

    # Verify the generated split
    verify_split_distribution(final_split_config, seq_to_label_map)
