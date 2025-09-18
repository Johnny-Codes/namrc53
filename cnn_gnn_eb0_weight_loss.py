import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import warnings
import pandas as pd
import time
import random
import os

warnings.filterwarnings("ignore")


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All seeds set to {seed} for reproducibility")


# Move collate_fn outside of main to avoid pickling issues
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(
        f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )


class WeightedLossCalculator:
    """Calculate class weights for imbalanced datasets with enhanced weighting for rare classes"""

    def __init__(self, weighting_strategy="inverse_sqrt", rare_class_boost=2.0):
        """
        Args:
            weighting_strategy: 'inverse', 'inverse_sqrt', 'balanced', 'log_balanced'
            rare_class_boost: Additional multiplier for classes below median frequency
        """
        self.weighting_strategy = weighting_strategy
        self.rare_class_boost = rare_class_boost

    def calculate_class_weights(self, class_counts, num_classes):
        """
        Calculate class weights with enhanced penalties for rare classes

        Args:
            class_counts: Counter object with class frequencies
            num_classes: Total number of classes

        Returns:
            torch.Tensor: Class weights for CrossEntropyLoss
        """
        # Get class frequencies in order
        class_freqs = np.zeros(num_classes)
        for class_idx, count in class_counts.items():
            class_freqs[class_idx] = count

        # Avoid division by zero
        class_freqs = np.maximum(class_freqs, 1)

        # Calculate base weights using selected strategy
        if self.weighting_strategy == "inverse":
            weights = 1.0 / class_freqs
        elif self.weighting_strategy == "inverse_sqrt":
            weights = 1.0 / np.sqrt(class_freqs)
        elif self.weighting_strategy == "balanced":
            total_samples = np.sum(class_freqs)
            weights = total_samples / (num_classes * class_freqs)
        elif self.weighting_strategy == "log_balanced":
            total_samples = np.sum(class_freqs)
            weights = np.log(total_samples / class_freqs + 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # Apply additional boost to rare classes (below median frequency)
        median_freq = np.median(class_freqs)
        rare_class_mask = class_freqs < median_freq
        weights[rare_class_mask] *= self.rare_class_boost

        # Normalize weights to prevent extreme values
        weights = weights / np.mean(weights)

        # Convert to tensor
        class_weights = torch.FloatTensor(weights).to(device)

        return class_weights

    def print_weight_analysis(self, class_weights, class_names, class_counts):
        """Print detailed analysis of calculated weights"""
        print(f"\n{'='*80}")
        print("🎯 CLASS WEIGHT ANALYSIS")
        print(f"{'='*80}")
        print(f"Weighting Strategy: {self.weighting_strategy}")
        print(f"Rare Class Boost: {self.rare_class_boost}x")
        print(f"\n{'Class Name':<25} {'Count':<8} {'Weight':<12} {'Penalty':<10}")
        print("-" * 65)

        weights_cpu = class_weights.cpu().numpy()

        for i, (class_name, weight) in enumerate(zip(class_names, weights_cpu)):
            count = class_counts.get(i, 0)
            penalty = weight / weights_cpu.min()
            print(f"{class_name:<25} {count:<8} {weight:<12.4f} {penalty:<10.2f}x")

        print("-" * 65)
        print(f"Weight Range: {weights_cpu.min():.4f} - {weights_cpu.max():.4f}")
        print(f"Max Penalty Ratio: {(weights_cpu.max() / weights_cpu.min()):.2f}x")
        print(f"Mean Weight: {weights_cpu.mean():.4f}")
        print(f"Std Weight: {weights_cpu.std():.4f}")


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


class EfficientNetSpatialCNN(nn.Module):
    """EfficientNet-B0 backbone for spatial feature extraction"""

    def __init__(self, feature_dim=256, pretrained=True):
        super(EfficientNetSpatialCNN, self).__init__()

        if pretrained:
            self.backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = efficientnet_b0(weights=None)

        self.backbone_dim = 1280
        self.backbone.classifier = nn.Identity()

        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        print(
            f"Initialized EfficientNet-B0 with {self.backbone_dim} -> {feature_dim} features"
        )

    def forward(self, x):
        features = self.backbone.features(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        projected_features = self.feature_projector(pooled_features)
        return projected_features


class OptimizedTemporalGNN(nn.Module):
    """Simplified and optimized GNN module"""

    def __init__(self, feature_dim=256, hidden_dim=128, num_layers=2, num_heads=4):
        super(OptimizedTemporalGNN, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=0.1,
                    concat=True,
                )
            )

        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x, edge_index, batch):
        h = self.input_proj(x)

        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.relu(h)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max], dim=1)

        temporal_features = self.temporal_pool(h_combined)
        return temporal_features


class WeightedEfficientNetCNNGNN(nn.Module):
    """Weighted loss CNN-GNN model using EfficientNet-B0"""

    def __init__(
        self,
        num_classes,
        cnn_feature_dim=256,
        gnn_hidden_dim=128,
        gnn_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super(WeightedEfficientNetCNNGNN, self).__init__()

        self.num_classes = num_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        self.spatial_cnn = EfficientNetSpatialCNN(feature_dim=cnn_feature_dim)

        self.temporal_gnn = OptimizedTemporalGNN(
            feature_dim=cnn_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, num_classes),
        )

    def forward(self, frames, edge_indices, batch_indices, verbose=False):
        batch_size = len(edge_indices)

        with torch.cuda.amp.autocast():
            spatial_features = self.spatial_cnn(frames)

        total_frames = spatial_features.size(0)
        seq_len = total_frames // batch_size

        gnn_outputs = []

        for i in range(batch_size):
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            seq_features = spatial_features[start_idx:end_idx]

            try:
                gnn_output = self.temporal_gnn(
                    seq_features, edge_indices[i], batch_indices[i]
                )
                gnn_outputs.append(gnn_output)
            except Exception as e:
                if verbose:
                    print(f"Error in GNN for sequence {i}: {e}")
                fallback_output = torch.mean(seq_features, dim=0, keepdim=True)
                fallback_output = self.temporal_gnn.temporal_pool(
                    torch.cat([fallback_output, fallback_output], dim=1)
                )
                gnn_outputs.append(fallback_output)

        if gnn_outputs:
            temporal_features = torch.stack(
                [out.squeeze(0) if out.dim() > 1 else out for out in gnn_outputs]
            )
        else:
            temporal_features = torch.zeros(
                batch_size, self.gnn_hidden_dim, device=frames.device
            )

        logits = self.classifier(temporal_features)
        return logits


class StratifiedVideoDataset(Dataset):
    """Video dataset with proper stratified splitting capabilities"""

    def __init__(
        self,
        data_dir,
        sequence_length=8,
        image_size=(224, 224),
        temporal_window=2,
        train=True,
        max_sequences=None,
        cache_frames=True,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.temporal_window = temporal_window
        self.train = train
        self.max_sequences = max_sequences
        self.cache_frames = cache_frames
        self.frame_cache = {}

        # Load sequences
        self.sequences = self._create_sequences()

        # Create label mapping
        all_labels = set()
        for seq in self.sequences:
            all_labels.update([item["action_label"] for item in seq["frames"]])

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        print(f"Found {len(self.sequences)} sequences with {self.num_classes} classes")

        # Analyze class distribution in sequences
        sequence_labels = []
        for seq in self.sequences:
            labels = [item["action_label"] for item in seq["frames"]]
            majority_label = max(set(labels), key=labels.count)
            sequence_labels.append(majority_label)

        class_counts = Counter(sequence_labels)
        print("\nSequence-level class distribution:")
        for label, count in sorted(class_counts.items()):
            print(
                f"  {label}: {count} sequences ({count/len(self.sequences)*100:.1f}%)"
            )

        # Store sequence labels for stratification
        self.sequence_labels = sequence_labels

        # Optimized transforms
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _create_sequences(self):
        """Create sequences from video data"""
        sequences = []
        train_dirs = sorted(self.data_dir.glob("train-*"))

        print(f"Processing {len(train_dirs)} training directories...")

        for train_dir in tqdm(train_dirs, desc="Loading directories"):
            metadata_file = train_dir / "metadata.jsonl"
            if not metadata_file.exists():
                continue

            frames_data = []
            with open(metadata_file, "r") as f:
                for line in f:
                    try:
                        frame_data = json.loads(line.strip())
                        if all(
                            field in frame_data
                            for field in ["file_name", "action_label", "action_number"]
                        ):
                            frame_data["train_dir"] = train_dir.name
                            frames_data.append(frame_data)
                    except:
                        continue

            frames_data.sort(key=lambda x: x.get("action_number", 0))

            step_size = self.sequence_length
            for i in range(0, len(frames_data) - self.sequence_length + 1, step_size):
                sequence_frames = frames_data[i : i + self.sequence_length]

                if len(sequence_frames) == self.sequence_length:
                    sequences.append(
                        {"frames": sequence_frames, "train_dir": train_dir}
                    )

                    if self.max_sequences and len(sequences) >= self.max_sequences:
                        return sequences

        return sequences

    def _create_temporal_graph(self, sequence_frames):
        """Create temporal graph for sequence"""
        num_nodes = len(sequence_frames)
        edge_indices = []

        for i in range(num_nodes - 1):
            edge_indices.append([i, i + 1])
            edge_indices.append([i + 1, i])

        for i in range(num_nodes):
            for j in range(
                max(0, i - self.temporal_window),
                min(num_nodes, i + self.temporal_window + 1),
            ):
                if i != j and [i, j] not in edge_indices:
                    edge_indices.append([i, j])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()

        batch = torch.zeros(num_nodes, dtype=torch.long)
        return edge_index, batch

    def _load_frame(self, train_dir, file_name):
        """Load frame with caching"""
        cache_key = f"{train_dir.name}_{file_name}"

        if self.cache_frames and cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        frame_path = train_dir / file_name

        if not frame_path.exists():
            frame = np.zeros(
                (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
            )
        else:
            try:
                image = cv2.imread(str(frame_path))
                if image is None:
                    frame = np.zeros(
                        (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                    )
                else:
                    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                frame = np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                )

        if self.cache_frames and len(self.frame_cache) < 1000:
            self.frame_cache[cache_key] = frame

        return frame

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames_data = sequence["frames"]
        train_dir = sequence["train_dir"]

        frames = []
        labels = []

        for frame_data in frames_data:
            frame = self._load_frame(train_dir, frame_data["file_name"])
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

            label_idx = self.label_to_idx.get(frame_data["action_label"], 0)
            labels.append(label_idx)

        frames_tensor = torch.stack(frames)
        edge_index, batch = self._create_temporal_graph(frames_data)

        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        sequence_label = max(label_counts.keys(), key=lambda x: label_counts[x])

        return {
            "frames": frames_tensor,
            "edge_index": edge_index,
            "batch": batch,
            "label": sequence_label,
            "sequence_id": idx,
        }


class ResultsSaver:
    """Comprehensive results saver for weighted loss experiments"""

    def __init__(self, save_dir="results", experiment_name=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"weighted_efficientnet_experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)

        print(f"Results will be saved to: {self.experiment_dir}")

        # Initialize results storage
        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "val_accuracy": [],
            "test_accuracy": [],
            "val_f1_macro": [],
            "test_f1_macro": [],
            "val_f1_micro": [],
            "test_f1_micro": [],
            "val_f1_weighted": [],
            "test_f1_weighted": [],
            "val_precision_macro": [],
            "test_precision_macro": [],
            "val_precision_micro": [],
            "test_precision_micro": [],
            "val_precision_weighted": [],
            "test_precision_weighted": [],
            "val_recall_macro": [],
            "test_recall_macro": [],
            "val_recall_micro": [],
            "test_recall_micro": [],
            "val_recall_weighted": [],
            "test_recall_weighted": [],
            "val_cohen_kappa": [],
            "test_cohen_kappa": [],
            "val_matthews_corrcoef": [],
            "test_matthews_corrcoef": [],
            "epoch_time": [],
            "learning_rate": [],
        }

    def save_epoch_results(
        self, epoch, train_loss, val_metrics, test_metrics, epoch_time, learning_rate
    ):
        """Save results for each epoch including test metrics"""
        self.training_history["epoch"].append(epoch)
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_metrics["loss"])
        self.training_history["test_loss"].append(test_metrics["loss"])
        self.training_history["val_accuracy"].append(val_metrics["accuracy"])
        self.training_history["test_accuracy"].append(test_metrics["accuracy"])
        self.training_history["val_f1_macro"].append(val_metrics["f1_macro"])
        self.training_history["test_f1_macro"].append(test_metrics["f1_macro"])
        self.training_history["val_f1_micro"].append(val_metrics["f1_micro"])
        self.training_history["test_f1_micro"].append(test_metrics["f1_micro"])
        self.training_history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        self.training_history["test_f1_weighted"].append(test_metrics["f1_weighted"])
        self.training_history["val_precision_macro"].append(
            val_metrics["precision_macro"]
        )
        self.training_history["test_precision_macro"].append(
            test_metrics["precision_macro"]
        )
        self.training_history["val_precision_micro"].append(
            val_metrics["precision_micro"]
        )
        self.training_history["test_precision_micro"].append(
            test_metrics["precision_micro"]
        )
        self.training_history["val_precision_weighted"].append(
            val_metrics["precision_weighted"]
        )
        self.training_history["test_precision_weighted"].append(
            test_metrics["precision_weighted"]
        )
        self.training_history["val_recall_macro"].append(val_metrics["recall_macro"])
        self.training_history["test_recall_macro"].append(test_metrics["recall_macro"])
        self.training_history["val_recall_micro"].append(val_metrics["recall_micro"])
        self.training_history["test_recall_micro"].append(test_metrics["recall_micro"])
        self.training_history["val_recall_weighted"].append(
            val_metrics["recall_weighted"]
        )
        self.training_history["test_recall_weighted"].append(
            test_metrics["recall_weighted"]
        )
        self.training_history["val_cohen_kappa"].append(
            val_metrics.get("cohen_kappa", 0.0)
        )
        self.training_history["test_cohen_kappa"].append(
            test_metrics.get("cohen_kappa", 0.0)
        )
        self.training_history["val_matthews_corrcoef"].append(
            val_metrics.get("matthews_corrcoef", 0.0)
        )
        self.training_history["test_matthews_corrcoef"].append(
            test_metrics.get("matthews_corrcoef", 0.0)
        )
        self.training_history["epoch_time"].append(epoch_time)
        self.training_history["learning_rate"].append(learning_rate)

        # Save training history after each epoch
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.experiment_dir / "data" / "training_history.csv", index=False)

    def save_final_results(
        self,
        final_metrics,
        class_names,
        y_true,
        y_pred,
        y_proba=None,
        split_name="test",
    ):
        """Save final comprehensive results"""
        # Save final metrics
        with open(
            self.experiment_dir / "data" / f"final_metrics_{split_name}.json", "w"
        ) as f:
            json.dump(final_metrics, f, indent=2)

        # Per-class results
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class_df = pd.DataFrame(
            {
                "class_name": class_names,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

        per_class_df.to_csv(
            self.experiment_dir / "data" / f"per_class_metrics_{split_name}.csv",
            index=False,
        )

        # Save predictions for analysis
        predictions_df = pd.DataFrame(
            {
                "true_label": y_true,
                "predicted_label": y_pred,
                "true_class": [class_names[i] for i in y_true],
                "predicted_class": [class_names[i] for i in y_pred],
            }
        )

        if y_proba is not None:
            for i, class_name in enumerate(class_names):
                predictions_df[f"prob_{class_name}"] = y_proba[:, i]

        predictions_df.to_csv(
            self.experiment_dir / "data" / f"predictions_{split_name}.csv", index=False
        )

        print(f"Final {split_name} results saved to {self.experiment_dir / 'data'}")

    def save_experiment_config(self, config, model_params, class_weights_info):
        """Save experiment configuration including weighted loss details"""
        experiment_info = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "model_parameters": model_params,
            "class_weights_info": class_weights_info,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        }

        if torch.cuda.is_available():
            experiment_info["cuda_device"] = torch.cuda.get_device_name()
            experiment_info["cuda_memory"] = torch.cuda.get_device_properties(
                0
            ).total_memory

        with open(self.experiment_dir / "experiment_config.json", "w") as f:
            json.dump(experiment_info, f, indent=2)

        print(f"Experiment configuration saved to {self.experiment_dir}")

    def create_weighted_loss_plots(self, class_weights, class_names, class_counts):
        """Create plots specific to weighted loss analysis"""
        plots_dir = self.experiment_dir / "plots"

        # Class weights visualization
        plt.figure(figsize=(14, 8))

        weights_cpu = class_weights.cpu().numpy()
        counts = [class_counts.get(i, 0) for i in range(len(class_names))]

        # Create bar plot
        x_pos = np.arange(len(class_names))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Top plot: Class counts
        bars1 = ax1.bar(x_pos, counts, color="skyblue", alpha=0.7)
        ax1.set_xlabel("Classes")
        ax1.set_ylabel("Sample Count")
        ax1.set_title("Class Distribution (Sample Counts)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars1, counts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Bottom plot: Class weights
        bars2 = ax2.bar(x_pos, weights_cpu, color="coral", alpha=0.7)
        ax2.set_xlabel("Classes")
        ax2.set_ylabel("Loss Weight")
        ax2.set_title("Class Weights for Weighted CrossEntropyLoss")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Add weight labels on bars
        for bar, weight in zip(bars2, weights_cpu):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(weights_cpu) * 0.01,
                f"{weight:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "class_weights_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "class_weights_analysis.pdf", bbox_inches="tight")
        plt.close()

        # Weight vs Count correlation plot
        plt.figure(figsize=(10, 8))
        plt.scatter(counts, weights_cpu, alpha=0.7, s=100)

        for i, class_name in enumerate(class_names):
            plt.annotate(
                class_name,
                (counts[i], weights_cpu[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        plt.xlabel("Sample Count")
        plt.ylabel("Loss Weight")
        plt.title("Class Weight vs Sample Count Correlation")
        plt.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(counts, weights_cpu, 1)
        p = np.poly1d(z)
        plt.plot(
            counts,
            p(counts),
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}",
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            plots_dir / "weight_count_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "weight_count_correlation.pdf", bbox_inches="tight")
        plt.close()

        print(f"✅ Weighted loss analysis plots saved to {plots_dir}")


class WeightedTrainer:
    """Trainer with weighted CrossEntropyLoss for handling class imbalance"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_names,
        class_weights,
        learning_rate=1e-3,
        weight_decay=1e-4,
        results_saver=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_weights = class_weights
        self.results_saver = results_saver

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=50,
        )

        # Weighted CrossEntropyLoss with enhanced penalties for rare classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.scaler = torch.cuda.amp.GradScaler()

        print("🎯 Using Weighted CrossEntropyLoss for class imbalance handling")
        print(f"Weight range: {class_weights.min():.4f} - {class_weights.max():.4f}")
        print(f"Max penalty ratio: {(class_weights.max() / class_weights.min()):.2f}x")
        print("Using mixed precision training for speed optimization")
        print(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
        )

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None, loss=0.0):
        """Calculate comprehensive metrics"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "loss": loss,
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_micro": 0.0,
                "f1_weighted": 0.0,
                "precision_macro": 0.0,
                "precision_micro": 0.0,
                "precision_weighted": 0.0,
                "recall_macro": 0.0,
                "recall_micro": 0.0,
                "recall_weighted": 0.0,
                "cohen_kappa": 0.0,
                "matthews_corrcoef": 0.0,
            }

        metrics = {
            "loss": loss,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "precision_micro": precision_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall_micro": recall_score(
                y_true, y_pred, average="micro", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "cohen_kappa": cohen_kappa_score(y_true, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        }

        return metrics

    def train_epoch(self):
        """Training epoch with weighted loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training (Weighted Loss)")

        for batch_idx, batch in enumerate(pbar):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            edge_indices = [
                ei.to(device, non_blocking=True) for ei in batch["edge_indices"]
            ]
            batch_indices = [
                bi.to(device, non_blocking=True) for bi in batch["batch_indices"]
            ]

            self.optimizer.zero_grad()

            try:
                with torch.cuda.amp.autocast():
                    logits = self.model(frames, edge_indices, batch_indices)
                    # Weighted loss automatically applies higher penalties to rare classes
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                elapsed = time.time() - start_time
                it_per_sec = (batch_idx + 1) / elapsed

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "it/s": f"{it_per_sec:.2f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    }
                )

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, data_loader, split_name="val"):
        """Evaluate on validation or test set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")

            for batch in pbar:
                frames = batch["frames"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                edge_indices = [
                    ei.to(device, non_blocking=True) for ei in batch["edge_indices"]
                ]
                batch_indices = [
                    bi.to(device, non_blocking=True) for bi in batch["batch_indices"]
                ]

                try:
                    with torch.cuda.amp.autocast():
                        logits = self.model(frames, edge_indices, batch_indices)
                        loss = self.criterion(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1

                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                except Exception as e:
                    print(f"Error in {split_name} evaluation: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)

        if all_labels and all_predictions:
            metrics = self.calculate_comprehensive_metrics(
                all_labels,
                all_predictions,
                np.array(all_probabilities) if all_probabilities else None,
                avg_loss,
            )
        else:
            metrics = {"loss": avg_loss, "accuracy": 0.0, "f1_macro": 0.0}

        return metrics, all_predictions, all_labels, all_probabilities

    def train(self, num_epochs=30, save_path="weighted_model.pth"):
        """Training loop with weighted loss"""
        best_f1 = 0

        print(f"Starting weighted loss training for {num_epochs} epochs...")
        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics, val_predictions, val_labels, val_probabilities = self.evaluate(
                self.val_loader, "validation"
            )

            # Test
            test_metrics, test_predictions, test_labels, test_probabilities = (
                self.evaluate(self.test_loader, "test")
            )

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss (Weighted): {train_loss:.4f}")
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_macro']:.4f}"
            )
            print(
                f"Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_macro']:.4f}"
            )

            # Save epoch results
            if self.results_saver:
                self.results_saver.save_epoch_results(
                    epoch + 1,
                    train_loss,
                    val_metrics,
                    test_metrics,
                    epoch_time,
                    current_lr,
                )

            # Save best model based on validation F1
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]

                if self.results_saver:
                    model_save_path = (
                        self.results_saver.experiment_dir
                        / "models"
                        / "best_weighted_model.pth"
                    )
                else:
                    model_save_path = save_path

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "class_weights": self.class_weights,
                        "val_f1_score": val_metrics["f1_macro"],
                        "test_f1_score": test_metrics["f1_macro"],
                        "val_accuracy": val_metrics["accuracy"],
                        "test_accuracy": test_metrics["accuracy"],
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                    },
                    model_save_path,
                )

                print(
                    f"✓ Best weighted model saved (Val F1: {best_f1:.4f}, Test F1: {test_metrics['f1_macro']:.4f})"
                )

                # Save results for best model
                if self.results_saver:
                    self.results_saver.save_final_results(
                        val_metrics,
                        self.class_names,
                        val_labels,
                        val_predictions,
                        np.array(val_probabilities) if val_probabilities else None,
                        "validation",
                    )
                    self.results_saver.save_final_results(
                        test_metrics,
                        self.class_names,
                        test_labels,
                        test_predictions,
                        np.array(test_probabilities) if test_probabilities else None,
                        "test",
                    )

        total_time = time.time() - total_start_time
        print(f"\nWeighted loss training completed in {total_time:.1f} seconds")
        print(f"Best Validation F1 Score: {best_f1:.4f}")

        return best_f1, val_metrics, test_metrics


def main():
    """Main function for weighted loss EfficientNet-B0 training"""

    print("🎯 WEIGHTED LOSS EFFICIENTNET-B0 CNN-GNN Video Classification")
    print("=" * 80)

    # Set seeds for reproducibility
    set_seeds(42)

    # Configuration
    config = {
        "data_dir": "./data",
        "sequence_length": 8,
        "image_size": (224, 224),
        "temporal_window": 2,
        "batch_size": 8,
        "num_epochs": 30,
        "learning_rate": 1e-3,
        "cnn_feature_dim": 256,
        "gnn_hidden_dim": 128,
        "gnn_layers": 2,
        "num_heads": 4,
        "max_sequences": None,
        "seed": 42,
        # Weighted loss specific parameters
        "weighting_strategy": "inverse_sqrt",  # 'inverse', 'inverse_sqrt', 'balanced', 'log_balanced'
        "rare_class_boost": 3.0,  # Additional multiplier for rare classes
    }

    print(f"Configuration: {config}")
    print(
        f"🎯 Using {config['weighting_strategy']} weighting with {config['rare_class_boost']}x rare class boost"
    )

    try:
        # Create dataset
        print("\n📂 Creating dataset...")
        dataset = StratifiedVideoDataset(
            data_dir=config["data_dir"],
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            temporal_window=config["temporal_window"],
            train=True,
            max_sequences=config["max_sequences"],
            cache_frames=True,
        )

        if len(dataset) == 0:
            print("❌ No sequences found! Check your data directory.")
            return

        # Create stratified splits
        print("\n🎯 Creating stratified splits...")
        splitter = StratifiedVideoDatasetSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            min_samples_per_split=1,
            seed=config["seed"],
        )

        train_indices, val_indices, test_indices, class_splits, class_counts = (
            splitter.create_stratified_split(dataset)
        )

        # Calculate class weights for weighted loss
        print("\n⚖️  Calculating class weights for weighted loss...")
        weight_calculator = WeightedLossCalculator(
            weighting_strategy=config["weighting_strategy"],
            rare_class_boost=config["rare_class_boost"],
        )

        class_weights = weight_calculator.calculate_class_weights(
            class_counts, dataset.num_classes
        )

        weight_calculator.print_weight_analysis(
            class_weights, list(dataset.label_to_idx.keys()), class_counts
        )

        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"efficientnet_b0_weighted_{config['weighting_strategy']}_boost{config['rare_class_boost']}_seq{config['sequence_length']}_bs{config['batch_size']}_lr{config['learning_rate']}_{timestamp}"
        results_saver = ResultsSaver(
            save_dir="results", experiment_name=experiment_name
        )

        # Create data subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

        print(
            f"Dataset splits - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        # Create weighted model
        print(f"\n🏗️  Creating EfficientNet-B0 CNN-GNN model...")
        model = WeightedEfficientNetCNNGNN(
            num_classes=dataset.num_classes,
            cnn_feature_dim=config["cnn_feature_dim"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            gnn_layers=config["gnn_layers"],
            num_heads=config["num_heads"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Save experiment configuration
        model_params = {
            "backbone_name": "efficientnet_b0",
            "backbone_description": "EfficientNet-B0 with Weighted Loss for Class Imbalance",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": dataset.num_classes,
            "class_names": list(dataset.label_to_idx.keys()),
        }

        class_weights_info = {
            "weighting_strategy": config["weighting_strategy"],
            "rare_class_boost": config["rare_class_boost"],
            "class_weights": class_weights.cpu().tolist(),
            "weight_range": {
                "min": float(class_weights.min()),
                "max": float(class_weights.max()),
                "mean": float(class_weights.mean()),
                "std": float(class_weights.std()),
            },
            "max_penalty_ratio": float(class_weights.max() / class_weights.min()),
        }

        results_saver.save_experiment_config(config, model_params, class_weights_info)

        # Create weighted loss visualization
        results_saver.create_weighted_loss_plots(
            class_weights, list(dataset.label_to_idx.keys()), class_counts
        )

        # Create trainer with weighted loss
        trainer = WeightedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=dataset.num_classes,
            class_names=list(dataset.label_to_idx.keys()),
            class_weights=class_weights,
            learning_rate=config["learning_rate"],
            results_saver=results_saver,
        )

        # Train model
        print(f"\n🏃‍♂️ Starting weighted loss training...")
        start_time = time.time()

        best_f1, final_val_metrics, final_test_metrics = trainer.train(
            num_epochs=config["num_epochs"]
        )

        training_time = time.time() - start_time

        print(f"\n🎉 Weighted loss training completed!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Final Validation Accuracy: {final_val_metrics['accuracy']:.4f}")
        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Training Time: {training_time:.1f} seconds")
        print(f"Results saved to: {results_saver.experiment_dir}")

        # Print final class-wise performance analysis
        print(f"\n{'='*80}")
        print("📊 WEIGHTED LOSS EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Backbone: EfficientNet-B0")
        print(f"Weighting Strategy: {config['weighting_strategy']}")
        print(f"Rare Class Boost: {config['rare_class_boost']}x")
        print(f"Max Penalty Ratio: {class_weights_info['max_penalty_ratio']:.2f}x")
        print(f"Best Validation F1: {best_f1:.4f}")
        print(f"Final Test F1: {final_test_metrics['f1_macro']:.4f}")
        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error in main: {e}")
        import traceback

        traceback.print_exc()
        print("💡 Try running with smaller batch_size or max_sequences for debugging")


if __name__ == "__main__":
    main()
