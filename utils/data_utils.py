import torch
from collections import defaultdict, Counter
import random
import numpy as np


def collate_fn(batch):
    """Optimized collate function for DataLoader"""
    frames = torch.stack([item["frames"] for item in batch])
    edge_indices = [item["edge_index"] for item in batch]
    batch_indices = [item["batch"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    batch_size, seq_len, channels, height, width = frames.shape
    frames = frames.view(batch_size * seq_len, channels, height, width)

    return {
        "frames": frames,
        "edge_indices": edge_indices,
        "batch_indices": batch_indices,
        "labels": labels,
    }


class StratifiedVideoDatasetSplitter:
    """Creates stratified train/val/test splits ensuring each class has proper representation"""

    def __init__(
        self,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        min_samples_per_split=1,
        seed=42,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_samples_per_split = min_samples_per_split
        self.seed = seed

        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
            self.train_ratio /= total_ratio
            self.val_ratio /= total_ratio
            self.test_ratio /= total_ratio

    def calculate_split_sizes(self, class_count):
        """Calculate train/val/test sizes for a given class count"""
        if class_count < 3:
            if class_count == 1:
                return 1, 0, 0
            elif class_count == 2:
                return 1, 1, 0
            else:
                return 1, 1, 1

        train_size = max(
            self.min_samples_per_split, int(class_count * self.train_ratio)
        )
        val_size = max(self.min_samples_per_split, int(class_count * self.val_ratio))
        test_size = max(self.min_samples_per_split, int(class_count * self.test_ratio))

        total_assigned = train_size + val_size + test_size
        if total_assigned > class_count:
            excess = total_assigned - class_count
            if train_size > self.min_samples_per_split:
                reduction = min(excess, train_size - self.min_samples_per_split)
                train_size -= reduction
                excess -= reduction
            if excess > 0 and val_size > self.min_samples_per_split:
                reduction = min(excess, val_size - self.min_samples_per_split)
                val_size -= reduction
                excess -= reduction
            if excess > 0 and test_size > self.min_samples_per_split:
                test_size -= excess

        remaining = class_count - (train_size + val_size + test_size)
        train_size += remaining

        return train_size, val_size, test_size

    def create_stratified_split(self, dataset):
        """Create stratified train/val/test splits"""
        all_labels = []
        for i in range(len(dataset)):
            sequence = dataset.sequences[i]
            frames_data = sequence["frames"]
            labels = [item["action_label"] for item in frames_data]
            majority_label = max(set(labels), key=labels.count)
            all_labels.append(majority_label)

        # Convert string labels to indices
        label_to_idx = dataset.label_to_idx
        all_label_indices = [label_to_idx[label] for label in all_labels]

        class_counts = Counter(all_label_indices)
        print(f"\nClass distribution analysis:")
        print(f"{'Class':<25} {'Count':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-" * 65)

        class_splits = {}
        total_train, total_val, total_test = 0, 0, 0

        for class_idx in range(dataset.num_classes):
            count = class_counts.get(class_idx, 0)
            class_name = dataset.idx_to_label[class_idx]
            train_size, val_size, test_size = self.calculate_split_sizes(count)
            class_splits[class_idx] = (train_size, val_size, test_size)
            total_train += train_size
            total_val += val_size
            total_test += test_size

            print(
                f"{class_name:<25} {count:<8} {train_size:<8} {val_size:<8} {test_size:<8}"
            )

        print("-" * 65)
        print(
            f"{'TOTAL':<25} {len(dataset):<8} {total_train:<8} {total_val:<8} {total_test:<8}"
        )

        # Create indices for each class
        class_indices = defaultdict(list)
        for idx, label_idx in enumerate(all_label_indices):
            class_indices[label_idx].append(idx)

        # Set random seed for reproducible splits
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Split each class
        train_indices, val_indices, test_indices = [], [], []

        for class_idx, indices in class_indices.items():
            train_size, val_size, test_size = class_splits[class_idx]

            class_indices_shuffled = indices.copy()
            random.shuffle(class_indices_shuffled)

            train_end = train_size
            val_end = train_end + val_size

            train_indices.extend(class_indices_shuffled[:train_end])
            val_indices.extend(class_indices_shuffled[train_end:val_end])
            test_indices.extend(class_indices_shuffled[val_end : val_end + test_size])

        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        print(f"\nFinal split sizes:")
        print(f"Train: {len(train_indices)} samples")
        print(f"Val: {len(val_indices)} samples")
        print(f"Test: {len(test_indices)} samples")

        return train_indices, val_indices, test_indices, class_splits, class_counts
