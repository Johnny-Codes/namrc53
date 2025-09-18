import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import torchvision.transforms as transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    swin_t,
    Swin_T_Weights,
)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(
        f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

# CNN backbone configurations
CNN_BACKBONES = {
    "efficientnet_b0": {
        "model_fn": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "feature_dim": 1280,
        "description": "Original EfficientNet-B0 - Baseline efficient CNN",
    },
    "efficientnet_v2_s": {
        "model_fn": efficientnet_v2_s,
        "weights": EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "feature_dim": 1280,
        "description": "EfficientNetV2-S - Faster training & better scaling",
    },
    "convnext_tiny": {
        "model_fn": convnext_tiny,
        "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "feature_dim": 768,
        "description": "ConvNeXt-Tiny - Modernized CNN with transformer-like performance",
    },
    "swin_tiny": {
        "model_fn": swin_t,
        "weights": Swin_T_Weights.IMAGENET1K_V1,
        "feature_dim": 768,
        "description": "Swin Transformer-Tiny - Hierarchical vision transformer",
    },
}


class StratifiedVideoDatasetSplitter:
    """
    Creates stratified train/val/test splits ensuring each class has proper representation
    """

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
            # For very small classes, ensure at least 1 sample per split if possible
            if class_count == 1:
                return 1, 0, 0  # All to train
            elif class_count == 2:
                return 1, 1, 0  # train=1, val=1, test=0
            else:  # class_count == 3
                return 1, 1, 1  # train=1, val=1, test=1

        # For larger classes, use proportional split but ensure minimums
        train_size = max(
            self.min_samples_per_split, int(class_count * self.train_ratio)
        )
        val_size = max(self.min_samples_per_split, int(class_count * self.val_ratio))
        test_size = max(self.min_samples_per_split, int(class_count * self.test_ratio))

        # Adjust if total exceeds class_count
        total_assigned = train_size + val_size + test_size
        if total_assigned > class_count:
            # Reduce proportionally, but maintain minimums
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

        # Assign remaining samples to train
        remaining = class_count - (train_size + val_size + test_size)
        train_size += remaining

        return train_size, val_size, test_size

    def create_stratified_split(self, dataset):
        """Create stratified train/val/test splits"""
        # Get all labels
        all_labels = []
        for i in range(len(dataset)):
            sequence = dataset.sequences[i]
            frames_data = sequence["frames"]
            labels = [item["action_label"] for item in frames_data]
            majority_label = max(set(labels), key=labels.count)
            all_labels.append(majority_label)

        # Count samples per class
        class_counts = Counter(all_labels)
        print(f"\nClass distribution analysis:")
        print(f"{'Class':<25} {'Count':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-" * 65)

        # Calculate splits for each class
        class_splits = {}
        total_train, total_val, total_test = 0, 0, 0

        for class_name, count in sorted(class_counts.items()):
            train_size, val_size, test_size = self.calculate_split_sizes(count)
            class_splits[class_name] = (train_size, val_size, test_size)
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
        print(
            f"{'RATIOS':<25} {'100%':<8} {total_train/len(dataset)*100:.1f}%{'':<4} {total_val/len(dataset)*100:.1f}%{'':<4} {total_test/len(dataset)*100:.1f}%"
        )

        # Create indices for each class
        class_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            class_indices[label].append(idx)

        # Set random seed for reproducible splits
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Split each class
        train_indices, val_indices, test_indices = [], [], []

        for class_name, indices in class_indices.items():
            train_size, val_size, test_size = class_splits[class_name]

            # Shuffle indices for this class
            class_indices_shuffled = indices.copy()
            random.shuffle(class_indices_shuffled)

            # Split indices
            train_end = train_size
            val_end = train_end + val_size

            train_indices.extend(class_indices_shuffled[:train_end])
            val_indices.extend(class_indices_shuffled[train_end:val_end])
            test_indices.extend(class_indices_shuffled[val_end : val_end + test_size])

        # Shuffle the final splits
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        print(f"\nFinal split sizes:")
        print(f"Train: {len(train_indices)} samples")
        print(f"Val: {len(val_indices)} samples")
        print(f"Test: {len(test_indices)} samples")
        print(
            f"Total: {len(train_indices) + len(val_indices) + len(test_indices)} samples"
        )

        return train_indices, val_indices, test_indices, class_splits


class AdaptiveSpatialCNN(nn.Module):
    """Adaptive CNN backbone that can use different architectures"""

    def __init__(
        self, backbone_name="efficientnet_b0", feature_dim=256, pretrained=True
    ):
        super(AdaptiveSpatialCNN, self).__init__()

        self.backbone_name = backbone_name
        backbone_config = CNN_BACKBONES[backbone_name]

        if pretrained:
            self.backbone = backbone_config["model_fn"](
                weights=backbone_config["weights"]
            )
        else:
            self.backbone = backbone_config["model_fn"](weights=None)

        self.backbone_dim = backbone_config["feature_dim"]

        # Remove classifier and adapt for different architectures
        if "efficientnet" in backbone_name:
            self.backbone.classifier = nn.Identity()
            self.feature_extractor = self._efficientnet_features
        elif "convnext" in backbone_name:
            self.backbone.classifier = nn.Identity()
            self.feature_extractor = self._convnext_features
        elif "swin" in backbone_name:
            self.backbone.head = nn.Identity()
            self.feature_extractor = self._swin_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        print(
            f"Initialized {backbone_name} backbone with {self.backbone_dim} -> {feature_dim} features"
        )

    def _efficientnet_features(self, x):
        features = self.backbone.features(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        return pooled_features.view(pooled_features.size(0), -1)

    def _convnext_features(self, x):
        features = self.backbone.features(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        return pooled_features.view(pooled_features.size(0), -1)

    def _swin_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)
        pooled_features = F.adaptive_avg_pool1d(x, 1)
        return pooled_features.view(pooled_features.size(0), -1)

    def forward(self, x):
        pooled_features = self.feature_extractor(x)
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


class AdaptiveCNNGNN(nn.Module):
    """CNN-GNN model that can use different CNN backbones"""

    def __init__(
        self,
        num_classes,
        backbone_name="efficientnet_b0",
        cnn_feature_dim=256,
        gnn_hidden_dim=128,
        gnn_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super(AdaptiveCNNGNN, self).__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.cnn_feature_dim = cnn_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        self.spatial_cnn = AdaptiveSpatialCNN(
            backbone_name=backbone_name, feature_dim=cnn_feature_dim
        )

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
    """
    Video dataset with proper stratified splitting capabilities
    """

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
    """Comprehensive results saver for journal publication"""

    def __init__(self, save_dir="results", experiment_name=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"cnn_gnn_experiment_{timestamp}"

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

        self.final_results = {}
        self.per_class_results = {}

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

        # Detailed classification report
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        with open(
            self.experiment_dir / "data" / f"classification_report_{split_name}.json",
            "w",
        ) as f:
            json.dump(report_dict, f, indent=2)

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

    def save_split_info(self, class_splits, train_indices, val_indices, test_indices):
        """Save information about the dataset splits"""
        split_info = {
            "class_splits": class_splits,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "split_summary": {
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "test_size": len(test_indices),
                "total_size": len(train_indices) + len(val_indices) + len(test_indices),
            },
        }

        with open(self.experiment_dir / "data" / "dataset_splits.json", "w") as f:
            json.dump(split_info, f, indent=2)

        print(
            f"Dataset split information saved to {self.experiment_dir / 'data' / 'dataset_splits.json'}"
        )

    def create_publication_plots(
        self, class_names, val_y_true, val_y_pred, test_y_true, test_y_pred
    ):
        """Create publication-quality plots with robust error handling"""

        # Safeguard: Check for empty data
        if (
            not val_y_true
            or not val_y_pred
            or not test_y_true
            or not test_y_pred
            or len(val_y_true) == 0
            or len(val_y_pred) == 0
            or len(test_y_true) == 0
            or len(test_y_pred) == 0
        ):
            print("⚠️  Warning: Empty prediction data, skipping plot creation")
            return

        # Safeguard: Check training history has data
        if (
            not self.training_history.get("epoch")
            or len(self.training_history["epoch"]) == 0
        ):
            print("⚠️  Warning: No training history available, skipping plot creation")
            return

        print(
            f"📊 Creating plots with {len(val_y_true)} validation and {len(test_y_true)} test samples"
        )

        # Use matplotlib backend handling
        try:
            plt.style.use("seaborn-v0_8")
        except:
            try:
                plt.style.use("seaborn")
            except:
                plt.style.use("default")
                print("Using default matplotlib style")

        # 1. Training curves (2x3 subplot) - comparing val and test
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Loss curves
        axes[0, 0].plot(
            self.training_history["epoch"],
            self.training_history["train_loss"],
            label="Training Loss",
            linewidth=2,
            color="blue",
        )
        axes[0, 0].plot(
            self.training_history["epoch"],
            self.training_history["val_loss"],
            label="Validation Loss",
            linewidth=2,
            color="red",
        )
        axes[0, 0].plot(
            self.training_history["epoch"],
            self.training_history["test_loss"],
            label="Test Loss",
            linewidth=2,
            color="green",
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training/Validation/Test Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(
            self.training_history["epoch"],
            self.training_history["val_accuracy"],
            label="Validation Accuracy",
            linewidth=2,
            color="red",
        )
        axes[0, 1].plot(
            self.training_history["epoch"],
            self.training_history["test_accuracy"],
            label="Test Accuracy",
            linewidth=2,
            color="green",
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Validation/Test Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 scores
        axes[0, 2].plot(
            self.training_history["epoch"],
            self.training_history["val_f1_macro"],
            label="Val F1 Macro",
            linewidth=2,
            color="purple",
        )
        axes[0, 2].plot(
            self.training_history["epoch"],
            self.training_history["test_f1_macro"],
            label="Test F1 Macro",
            linewidth=2,
            color="orange",
        )
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("F1 Score")
        axes[0, 2].set_title("F1 Scores (Macro)")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Precision
        axes[1, 0].plot(
            self.training_history["epoch"],
            self.training_history["val_precision_macro"],
            label="Val Precision",
            linewidth=2,
            color="brown",
        )
        axes[1, 0].plot(
            self.training_history["epoch"],
            self.training_history["test_precision_macro"],
            label="Test Precision",
            linewidth=2,
            color="pink",
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision (Macro)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Recall
        axes[1, 1].plot(
            self.training_history["epoch"],
            self.training_history["val_recall_macro"],
            label="Val Recall",
            linewidth=2,
            color="cyan",
        )
        axes[1, 1].plot(
            self.training_history["epoch"],
            self.training_history["test_recall_macro"],
            label="Test Recall",
            linewidth=2,
            color="magenta",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].set_title("Recall (Macro)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Cohen's Kappa
        axes[1, 2].plot(
            self.training_history["epoch"],
            self.training_history["val_cohen_kappa"],
            label="Val Kappa",
            linewidth=2,
            color="olive",
        )
        axes[1, 2].plot(
            self.training_history["epoch"],
            self.training_history["test_cohen_kappa"],
            label="Test Kappa",
            linewidth=2,
            color="navy",
        )
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Cohen's Kappa")
        axes[1, 2].set_title("Cohen's Kappa Score")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with error handling
        try:
            plt.savefig(
                plots_dir / "training_curves.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.savefig(
                plots_dir / "training_curves.pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"✅ Training curves saved to {plots_dir}")
        except Exception as e:
            print(f"❌ Error saving training curves: {e}")
        finally:
            plt.close()

    def _create_confusion_matrices(
        self, plots_dir, class_names, val_y_true, val_y_pred, test_y_true, test_y_pred
    ):
        """Create confusion matrices with error handling"""

        # Validation confusion matrix
        try:
            cm_val = confusion_matrix(val_y_true, val_y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm_val,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
            )
            plt.title("Validation Confusion Matrix", fontsize=16)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            plt.savefig(
                plots_dir / "confusion_matrix_val.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.savefig(
                plots_dir / "confusion_matrix_val.pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"✅ Validation confusion matrix saved")
            plt.close()
        except Exception as e:
            print(f"❌ Error creating validation confusion matrix: {e}")

        # Test confusion matrix
        try:
            cm_test = confusion_matrix(test_y_true, test_y_pred)
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm_test,
                annot=True,
                fmt="d",
                cmap="Reds",
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={"label": "Count"},
            )
            plt.title("Test Confusion Matrix", fontsize=16)
            plt.xlabel("Predicted Label", fontsize=14)
            plt.ylabel("True Label", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            plt.savefig(
                plots_dir / "confusion_matrix_test.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.savefig(
                plots_dir / "confusion_matrix_test.pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"✅ Test confusion matrix saved")
            plt.close()
        except Exception as e:
            print(f"❌ Error creating test confusion matrix: {e}")

    def save_experiment_config(self, config, model_params):
        """Save experiment configuration"""
        experiment_info = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "model_parameters": model_params,
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


class OptimizedTrainer:
    """Optimized trainer with train/val/test evaluation"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_names,
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

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()

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

        if y_proba is not None and self.num_classes > 2:
            try:
                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
            except:
                metrics["roc_auc_macro"] = 0.0
                metrics["roc_auc_weighted"] = 0.0

        return metrics

    def train_epoch(self):
        """Training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training")

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

    def train(self, num_epochs=30, save_path="optimized_model.pth"):
        """Training loop with train/val/test evaluation"""
        best_f1 = 0

        print(f"Starting training for {num_epochs} epochs...")
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
            print(f"Train Loss: {train_loss:.4f}")
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
                        self.results_saver.experiment_dir / "models" / "best_model.pth"
                    )
                else:
                    model_save_path = save_path

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
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
                    f"✓ Best model saved (Val F1: {best_f1:.4f}, Test F1: {test_metrics['f1_macro']:.4f})"
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
        print(f"\nTraining completed in {total_time:.1f} seconds")
        print(f"Best Validation F1 Score: {best_f1:.4f}")

        # Create publication plots
        if self.results_saver and val_labels and test_labels:
            self.results_saver.create_publication_plots(
                self.class_names,
                val_labels,
                val_predictions,
                test_labels,
                test_predictions,
            )

        return best_f1, val_metrics, test_metrics


def train_single_backbone(
    backbone_name, config, dataset, train_indices, val_indices, test_indices
):
    """Train a single backbone with proper stratified splits"""
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {backbone_name.upper()}")
    print(f"Description: {CNN_BACKBONES[backbone_name]['description']}")
    print(f"{'='*80}")

    # Set seeds for reproducibility
    set_seeds(config["seed"])

    # Fix: Add timestamp to prevent overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{backbone_name}_stratified_seq{config['sequence_length']}_bs{config['batch_size']}_lr{config['learning_rate']}_{timestamp}"
    results_saver = ResultsSaver(save_dir="results", experiment_name=experiment_name)

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

    # Create model with specified backbone
    print(f"\nCreating model with {backbone_name} backbone...")
    model = AdaptiveCNNGNN(
        num_classes=dataset.num_classes,
        backbone_name=backbone_name,
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
        "backbone_name": backbone_name,
        "backbone_description": CNN_BACKBONES[backbone_name]["description"],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_classes": dataset.num_classes,
        "class_names": list(dataset.label_to_idx.keys()),
    }
    results_saver.save_experiment_config(config, model_params)

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=dataset.num_classes,
        class_names=list(dataset.label_to_idx.keys()),
        learning_rate=config["learning_rate"],
        results_saver=results_saver,
    )

    # Train model
    print(f"\n🏃‍♂️ Starting training for {backbone_name}...")
    start_time = time.time()

    best_f1, final_val_metrics, final_test_metrics = trainer.train(
        num_epochs=config["num_epochs"], save_path=f"{backbone_name}_model.pth"
    )

    training_time = time.time() - start_time

    print(f"\n🎉 {backbone_name} training completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Final Validation Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"Training Time: {training_time:.1f} seconds")
    print(f"Results saved to: {results_saver.experiment_dir}")

    # Clear GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return {
        "backbone_name": backbone_name,
        "best_f1": best_f1,
        "final_val_accuracy": final_val_metrics["accuracy"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "training_time": training_time,
        "total_parameters": total_params,
        "results_dir": results_saver.experiment_dir,
    }


def create_comparison_report(all_results, save_dir="results"):
    """Create a comprehensive comparison report across all backbones"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create comparison dataframe
    comparison_data = []
    for result in all_results:
        comparison_data.append(
            {
                "Backbone": result["backbone_name"],
                "Description": CNN_BACKBONES[result["backbone_name"]]["description"],
                "Best F1 Score": result["best_f1"],
                "Final Validation Accuracy": result["final_val_accuracy"],
                "Final Test Accuracy": result["final_test_accuracy"],
                "Training Time (s)": result["training_time"],
                "Total Parameters": result["total_parameters"],
                "Results Directory": result["results_dir"].name,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Best F1 Score", ascending=False)

    # Save comparison table
    comparison_df.to_csv(save_dir / "backbone_comparison.csv", index=False)

    # Create comparison plots with error handling
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # F1 Score comparison
        bars1 = ax1.bar(
            comparison_df["Backbone"],
            comparison_df["Best F1 Score"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        ax1.set_title("F1 Score Comparison Across Backbones", fontsize=14)
        ax1.set_ylabel("F1 Score")
        ax1.tick_params(axis="x", rotation=45)
        for bar, score in zip(bars1, comparison_df["Best F1 Score"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # Validation Accuracy comparison
        bars2 = ax2.bar(
            comparison_df["Backbone"],
            comparison_df["Final Validation Accuracy"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        ax2.set_title("Validation Accuracy Comparison Across Backbones", fontsize=14)
        ax2.set_ylabel("Accuracy")
        ax2.tick_params(axis="x", rotation=45)
        for bar, acc in zip(bars2, comparison_df["Final Validation Accuracy"]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # Test Accuracy comparison
        bars3 = ax3.bar(
            comparison_df["Backbone"],
            comparison_df["Final Test Accuracy"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        ax3.set_title("Test Accuracy Comparison Across Backbones", fontsize=14)
        ax3.set_ylabel("Accuracy")
        ax3.tick_params(axis="x", rotation=45)
        for bar, acc in zip(bars3, comparison_df["Final Test Accuracy"]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # Training time comparison
        bars4 = ax4.bar(
            comparison_df["Backbone"],
            comparison_df["Training Time (s)"],
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        ax4.set_title("Training Time Comparison", fontsize=14)
        ax4.set_ylabel("Training Time (seconds)")
        ax4.tick_params(axis="x", rotation=45)
        for bar, time in zip(bars4, comparison_df["Training Time (s)"]):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{time:.0f}s",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            save_dir / "backbone_comparison.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.savefig(
            save_dir / "backbone_comparison.pdf", bbox_inches="tight", facecolor="white"
        )
        plt.close()

        print(f"✓ Backbone comparison plots saved to {save_dir}")

    except Exception as e:
        print(f"Error creating comparison plots: {e}")
        import traceback

        traceback.print_exc()

    # Print summary
    print(f"\n{'='*80}")
    print("🏆 BACKBONE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    print(
        f"\n🥇 Best performing backbone: {comparison_df.iloc[0]['Backbone']} (F1: {comparison_df.iloc[0]['Best F1 Score']:.4f})"
    )
    print(f"📊 Comparison plots saved to: {save_dir / 'backbone_comparison.png'}")
    print(f"📄 Comparison table saved to: {save_dir / 'backbone_comparison.csv'}")

    return comparison_df


def check_environment():
    """Check and setup the environment for optimal performance"""
    print("🔍 Environment Check:")

    # Check matplotlib backend
    print(f"   - Matplotlib backend: {plt.get_backend()}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"   - CUDA available: {torch.cuda.get_device_name()}")
        print(
            f"   - CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        print("   - CUDA: Not available (using CPU)")

    # Check disk space in results directory
    import shutil

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    total, used, free = shutil.disk_usage(results_dir)
    print(f"   - Available disk space: {free / 1e9:.1f} GB")

    if free < 5e9:  # Less than 5GB
        print("⚠️  Warning: Low disk space! Consider freeing up space.")

    print("✅ Environment check complete\n")


def main():
    """Main function with comprehensive error handling"""

    print("🚀 MULTI-BACKBONE CNN-GNN Video Classification")
    print("=" * 80)

    # Check environment first
    check_environment()

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
    }

    print(f"Configuration: {config}")
    print(f"Backbones to train: {list(CNN_BACKBONES.keys())}")

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

        train_indices, val_indices, test_indices, class_splits = (
            splitter.create_stratified_split(dataset)
        )

        # Save split information
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_saver_temp = ResultsSaver(
            save_dir="results", experiment_name=f"dataset_splits_info_{timestamp}"
        )
        results_saver_temp.save_split_info(
            class_splits, train_indices, val_indices, test_indices
        )

        # Train each backbone
        all_results = []
        total_start_time = time.time()
        successful_backbones = 0

        for i, backbone_name in enumerate(CNN_BACKBONES.keys(), 1):
            print(
                f"\n{'='*20} BACKBONE {i}/{len(CNN_BACKBONES)}: {backbone_name.upper()} {'='*20}"
            )

            try:
                result = train_single_backbone(
                    backbone_name,
                    config,
                    dataset,
                    train_indices,
                    val_indices,
                    test_indices,
                )
                all_results.append(result)
                successful_backbones += 1
                print(f"✅ {backbone_name} completed successfully")

            except Exception as e:
                print(f"❌ Error training {backbone_name}: {e}")
                import traceback

                traceback.print_exc()
                print(f"⏭️  Continuing with next backbone...")
                continue

        total_time = time.time() - total_start_time

        # Create comparison report
        if all_results:
            print(f"\n🎯 Creating comparison report...")
            try:
                comparison_df = create_comparison_report(all_results)
                print("✅ Comparison report created successfully")
            except Exception as e:
                print(f"❌ Error creating comparison report: {e}")

            print(f"\n{'='*80}")
            print("🎉 TRAINING COMPLETED!")
            print(f"{'='*80}")
            print(
                f"✅ Successfully trained: {successful_backbones}/{len(CNN_BACKBONES)} backbones"
            )
            print(
                f"⏱️  Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)"
            )
            print(f"📁 Results saved in subdirectories under 'results/'")
        else:
            print("❌ No backbones trained successfully!")
            print("🔍 Check your data directory and configuration.")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error in main: {e}")
        import traceback

        traceback.print_exc()
        print("💡 Try running with smaller batch_size or max_sequences for debugging")


def collate_fn(batch):
    """Optimized collate function"""
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


if __name__ == "__main__":
    main()
