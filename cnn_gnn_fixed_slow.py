import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
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
)
import seaborn as sns
from tqdm import tqdm
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SpatialCNN(nn.Module):
    """
    Advanced CNN backbone for spatial feature extraction from video frames
    Uses EfficientNet-B0 as base with custom modifications for temporal data
    """

    def __init__(self, feature_dim=512, pretrained=True):
        super(SpatialCNN, self).__init__()

        # Load pretrained EfficientNet-B0
        if pretrained:
            self.backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Remove the final classifier
        self.backbone.classifier = nn.Identity()

        # Get the feature dimension from EfficientNet-B0 (1280)
        backbone_dim = 1280

        # Add custom feature projection layers
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(backbone_dim, backbone_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_dim // 16, backbone_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, 3, height, width)

        # Extract features using backbone up to the last conv layer
        features = self.backbone.features(x)  # Shape: (batch_size, 1280, H', W')

        # Apply spatial attention
        attention = self.spatial_attention(features)
        features = features * attention

        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Project to desired feature dimension
        projected_features = self.feature_projector(pooled_features)

        return projected_features


class TemporalGNN(nn.Module):
    """
    Advanced Graph Neural Network for modeling temporal relationships
    Uses Graph Attention Networks with multiple layers and residual connections
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=3, num_heads=8):
        super(TemporalGNN, self).__init__()

        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Graph Attention layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=0.2,
                    concat=True,
                )
            )
            # Residual projection if needed
            self.residual_projections.append(nn.Linear(hidden_dim, hidden_dim))

        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Temporal pooling
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x, edge_index, batch):
        # x shape: (num_nodes, feature_dim)
        # edge_index shape: (2, num_edges)
        # batch: batch assignment for each node

        # Input projection
        h = self.input_proj(x)

        # Apply GAT layers with residual connections
        for i, (gat, residual_proj, layer_norm) in enumerate(
            zip(self.gat_layers, self.residual_projections, self.layer_norms)
        ):
            h_residual = residual_proj(h)
            h = gat(h, edge_index)
            h = h + h_residual  # Residual connection
            h = layer_norm(h)
            h = F.relu(h)

        # Global pooling across the graph
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max], dim=1)

        # Temporal pooling
        temporal_features = self.temporal_pool(h_combined)

        return temporal_features


class CNNGNN(nn.Module):
    """
    Full CNN-GNN model for video behavior classification
    """

    def __init__(
        self,
        num_classes,
        cnn_feature_dim=512,
        gnn_hidden_dim=256,
        gnn_layers=3,
        num_heads=8,
        dropout=0.3,
    ):
        super(CNNGNN, self).__init__()

        self.num_classes = num_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        # Spatial CNN for frame-level feature extraction
        self.spatial_cnn = SpatialCNN(feature_dim=cnn_feature_dim)

        # Temporal GNN for sequence modeling
        self.temporal_gnn = TemporalGNN(
            feature_dim=cnn_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
        )

        # Enhanced feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
        )

        # Self-attention for temporal features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=gnn_hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, gnn_hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(gnn_hidden_dim // 4, num_classes),
        )

    def forward(self, frames, edge_indices, batch_indices, verbose=False):
        """
        Forward pass
        Args:
            frames: (batch_size * sequence_length, 3, H, W) - video frames
            edge_indices: list of edge_index tensors for each sequence
            batch_indices: list of batch assignment tensors
            verbose: whether to print debug info
        """
        batch_size = len(edge_indices)

        if verbose:
            print(f"Input frames shape: {frames.shape}")
            print(f"Batch size: {batch_size}")

        # Extract spatial features from all frames
        spatial_features = self.spatial_cnn(frames)

        if verbose:
            print(f"Spatial features shape: {spatial_features.shape}")

        # Calculate sequence length
        total_frames = spatial_features.size(0)
        seq_len = total_frames // batch_size

        # Process each sequence through GNN
        gnn_outputs = []

        for i in range(batch_size):
            # Get features for this sequence
            start_idx = i * seq_len
            end_idx = (i + 1) * seq_len
            seq_features = spatial_features[start_idx:end_idx]

            # Apply GNN
            try:
                gnn_output = self.temporal_gnn(
                    seq_features, edge_indices[i], batch_indices[i]
                )
                gnn_outputs.append(gnn_output)
            except Exception as e:
                if verbose:
                    print(f"Error in GNN for sequence {i}: {e}")
                # Fallback: use mean pooling
                fallback_output = torch.mean(seq_features, dim=0, keepdim=True)
                fallback_output = self.temporal_gnn.temporal_pool(
                    torch.cat([fallback_output, fallback_output], dim=1)
                )
                gnn_outputs.append(fallback_output)

        # Stack GNN outputs
        if gnn_outputs:
            temporal_features = torch.stack(
                [out.squeeze(0) if out.dim() > 1 else out for out in gnn_outputs]
            )
        else:
            temporal_features = torch.zeros(
                batch_size, self.gnn_hidden_dim, device=frames.device
            )

        # Enhanced feature processing
        enhanced_features = self.feature_fusion(temporal_features)

        # Apply self-attention
        attn_input = enhanced_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attn_features, _ = self.self_attention(attn_input, attn_input, attn_input)
        attn_features = attn_features.squeeze(1)  # Remove sequence dimension

        # Final classification
        logits = self.classifier(attn_features)

        return logits


class VideoDataset(Dataset):
    """
    Full dataset class for loading video sequences with temporal relationships
    """

    def __init__(
        self,
        data_dir,
        sequence_length=16,
        image_size=(224, 224),
        temporal_window=3,
        train=True,
        max_sequences=None,  # No limit by default
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.temporal_window = temporal_window
        self.train = train
        self.max_sequences = max_sequences

        # Load metadata and create sequences
        self.sequences = self._create_sequences()

        # Create label mapping
        all_labels = set()
        for seq in self.sequences:
            all_labels.update([item["action_label"] for item in seq["frames"]])

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        print(f"Found {len(self.sequences)} sequences with {self.num_classes} classes")
        print(f"Classes: {list(self.label_to_idx.keys())}")

        # Print class distribution
        class_counts = defaultdict(int)
        for seq in self.sequences:
            labels = [item["action_label"] for item in seq["frames"]]
            majority_label = max(set(labels), key=labels.count)
            class_counts[majority_label] += 1

        print("\nClass distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} sequences")

        # Data transforms
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.RandomHorizontalFlip(p=0.3),
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
        """Create temporal sequences from the dataset"""
        sequences = []

        # Process all training directories
        train_dirs = sorted(self.data_dir.glob("train-*"))

        print(f"Processing {len(train_dirs)} training directories...")

        for train_dir in tqdm(train_dirs, desc="Loading directories"):
            metadata_file = train_dir / "metadata.jsonl"

            if not metadata_file.exists():
                continue

            # Load metadata
            frames_data = []
            with open(metadata_file, "r") as f:
                for line in f:
                    try:
                        frame_data = json.loads(line.strip())

                        # Check if required fields exist
                        if all(
                            field in frame_data
                            for field in ["file_name", "action_label", "action_number"]
                        ):
                            frame_data["train_dir"] = train_dir.name
                            frames_data.append(frame_data)
                    except:
                        continue

            # Sort by action number
            frames_data.sort(key=lambda x: x.get("action_number", 0))

            # Create sequences with fixed length
            step_size = self.sequence_length // 2  # Overlap sequences for more data
            for i in range(0, len(frames_data) - self.sequence_length + 1, step_size):
                sequence_frames = frames_data[i : i + self.sequence_length]

                if len(sequence_frames) == self.sequence_length:  # Exact length only
                    sequences.append(
                        {"frames": sequence_frames, "train_dir": train_dir}
                    )

                    if self.max_sequences and len(sequences) >= self.max_sequences:
                        return sequences

        return sequences

    def _create_temporal_graph(self, sequence_frames):
        """Create temporal graph edges for the sequence"""
        num_nodes = len(sequence_frames)
        edge_indices = []

        # Sequential connections (temporal order)
        for i in range(num_nodes - 1):
            edge_indices.append([i, i + 1])
            edge_indices.append([i + 1, i])  # Bidirectional

        # Window-based connections (temporal proximity)
        for i in range(num_nodes):
            for j in range(
                max(0, i - self.temporal_window),
                min(num_nodes, i + self.temporal_window + 1),
            ):
                if i != j and [i, j] not in edge_indices:
                    edge_indices.append([i, j])

        # Convert to tensor
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # Fallback: create at least one edge
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()

        # Batch indices (all nodes belong to the same graph)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        return edge_index, batch

    def _load_frame(self, train_dir, file_name):
        """Load and preprocess a single frame"""
        frame_path = train_dir / file_name

        if not frame_path.exists():
            # Return black frame if file doesn't exist
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        try:
            # Load image
            image = cv2.imread(str(frame_path))
            if image is None:
                return np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                )

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except:
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames_data = sequence["frames"]
        train_dir = sequence["train_dir"]

        # Load frames - ensure exact sequence length
        frames = []
        labels = []

        for frame_data in frames_data[: self.sequence_length]:  # Ensure exact length
            # Load frame
            frame = self._load_frame(train_dir, frame_data["file_name"])
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

            # Get label
            label_idx = self.label_to_idx.get(frame_data["action_label"], 0)
            labels.append(label_idx)

        # Stack frames
        frames_tensor = torch.stack(frames)  # (seq_len, 3, H, W)

        # Create temporal graph
        edge_index, batch = self._create_temporal_graph(frames_data)

        # Use majority vote for sequence label
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


def collate_fn(batch):
    """Custom collate function for batching graph data"""
    frames = torch.stack([item["frames"] for item in batch])
    edge_indices = [item["edge_index"] for item in batch]
    batch_indices = [item["batch"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Reshape frames for CNN processing
    batch_size, seq_len, channels, height, width = frames.shape
    frames = frames.view(batch_size * seq_len, channels, height, width)

    return {
        "frames": frames,
        "edge_indices": edge_indices,
        "batch_indices": batch_indices,
        "labels": labels,
    }


class MetricsCalculator:
    """Comprehensive metrics calculation for multi-class classification"""

    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names

    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive classification metrics"""

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["precision_micro"] = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics["precision_weighted"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["recall_micro"] = recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Agreement metrics
        metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
        metrics["matthews_corrcoef"] = matthews_corrcoef(y_true, y_pred)

        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f"f1_{class_name}"] = (
                per_class_f1[i] if i < len(per_class_f1) else 0.0
            )
            metrics[f"precision_{class_name}"] = (
                per_class_precision[i] if i < len(per_class_precision) else 0.0
            )
            metrics[f"recall_{class_name}"] = (
                per_class_recall[i] if i < len(per_class_recall) else 0.0
            )

        # ROC AUC and AP if probabilities are provided
        if y_proba is not None and self.num_classes > 2:
            try:
                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
                metrics["avg_precision_macro"] = average_precision_score(
                    np.eye(self.num_classes)[y_true], y_proba, average="macro"
                )
            except:
                metrics["roc_auc_macro"] = 0.0
                metrics["roc_auc_weighted"] = 0.0
                metrics["avg_precision_macro"] = 0.0

        return metrics

    def print_metrics_summary(self, metrics):
        """Print a formatted summary of metrics"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("=" * 60)

        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")

        if "roc_auc_macro" in metrics:
            print(f"ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
            print(f"Average Precision (Macro): {metrics['avg_precision_macro']:.4f}")

        print("\nPer-Class Metrics:")
        print("-" * 40)
        for class_name in self.class_names:
            f1 = metrics.get(f"f1_{class_name}", 0.0)
            precision = metrics.get(f"precision_{class_name}", 0.0)
            recall = metrics.get(f"recall_{class_name}", 0.0)
            print(f"{class_name:15s}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        return cm


class Trainer:
    """Enhanced training and evaluation manager with comprehensive metrics"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        # Metrics calculator
        self.metrics_calc = MetricsCalculator(num_classes, class_names)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            # Forward pass
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device)
            edge_indices = [ei.to(device) for ei in batch["edge_indices"]]
            batch_indices = [bi.to(device) for bi in batch["batch_indices"]]

            try:
                logits = self.model(frames, edge_indices, batch_indices, verbose=False)
                loss = self.criterion(logits, labels)

                # Check for NaN
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Validate the model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch_idx, batch in enumerate(pbar):
                frames = batch["frames"].to(device)
                labels = batch["labels"].to(device)
                edge_indices = [ei.to(device) for ei in batch["edge_indices"]]
                batch_indices = [bi.to(device) for bi in batch["batch_indices"]]

                try:
                    logits = self.model(
                        frames, edge_indices, batch_indices, verbose=False
                    )
                    loss = self.criterion(logits, labels)

                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1

                        # Get predictions and probabilities
                        probabilities = F.softmax(logits, dim=1)
                        predictions = torch.argmax(logits, dim=1)

                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probabilities.extend(probabilities.cpu().numpy())

                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue

        # Calculate comprehensive metrics
        avg_loss = total_loss / max(num_batches, 1)

        if all_labels and all_predictions:
            metrics = self.metrics_calc.calculate_all_metrics(
                all_labels,
                all_predictions,
                np.array(all_probabilities) if all_probabilities else None,
            )
        else:
            metrics = {"accuracy": 0.0, "f1_macro": 0.0}

        self.val_losses.append(avg_loss)
        self.val_metrics.append(metrics)

        return avg_loss, metrics, all_predictions, all_labels, all_probabilities

    def train(self, num_epochs=50, save_path="best_model.pth"):
        """Full training loop with comprehensive evaluation"""
        best_f1_score = 0
        best_metrics = None

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, metrics, predictions, labels, probabilities = self.validate()

            # Update scheduler
            self.scheduler.step()

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {metrics['accuracy']:.4f}")
            print(f"Val F1 (Macro): {metrics['f1_macro']:.4f}")
            print(f"Val F1 (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model based on F1 score
            if metrics["f1_macro"] > best_f1_score:
                best_f1_score = metrics["f1_macro"]
                best_metrics = metrics.copy()

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": metrics,
                        "loss": val_loss,
                    },
                    save_path,
                )
                print(f"✓ New best model saved with F1 score: {best_f1_score:.4f}")

        print(f"\nTraining completed! Best F1 score: {best_f1_score:.4f}")

        # Print final comprehensive metrics
        if best_metrics:
            self.metrics_calc.print_metrics_summary(best_metrics)

        return best_f1_score, best_metrics

    def evaluate_final_model(self, save_plots=True):
        """Final evaluation with comprehensive metrics and plots"""
        print("\nPerforming final model evaluation...")

        val_loss, metrics, predictions, labels, probabilities = self.validate()

        # Print comprehensive metrics
        self.metrics_calc.print_metrics_summary(metrics)

        # Generate classification report
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(
            classification_report(
                labels, predictions, target_names=self.class_names, zero_division=0
            )
        )

        if save_plots:
            # Plot confusion matrix
            self.metrics_calc.plot_confusion_matrix(
                labels, predictions, "final_confusion_matrix.png"
            )

            # Plot training history
            self.plot_training_history()

        return metrics

    def plot_training_history(self):
        """Plot comprehensive training curves"""
        if not self.train_losses or not self.val_losses:
            print("No training history to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Loss curves
        ax1.plot(self.train_losses, label="Train Loss", linewidth=2)
        ax1.plot(self.val_losses, label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        if self.val_metrics:
            accuracies = [m["accuracy"] for m in self.val_metrics]
            ax2.plot(accuracies, label="Val Accuracy", linewidth=2, color="green")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Validation Accuracy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # F1 scores
        if self.val_metrics:
            f1_macro = [m["f1_macro"] for m in self.val_metrics]
            f1_weighted = [m["f1_weighted"] for m in self.val_metrics]
            ax3.plot(f1_macro, label="F1 Macro", linewidth=2, color="blue")
            ax3.plot(f1_weighted, label="F1 Weighted", linewidth=2, color="orange")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("F1 Score")
            ax3.set_title("F1 Scores")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Other metrics
        if self.val_metrics:
            precision = [m["precision_macro"] for m in self.val_metrics]
            recall = [m["recall_macro"] for m in self.val_metrics]
            ax4.plot(precision, label="Precision (Macro)", linewidth=2, color="red")
            ax4.plot(recall, label="Recall (Macro)", linewidth=2, color="purple")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Score")
            ax4.set_title("Precision and Recall")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("comprehensive_training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Save metrics to CSV
        if self.val_metrics:
            metrics_df = pd.DataFrame(self.val_metrics)
            metrics_df.to_csv("training_metrics.csv", index=False)
            print("Training metrics saved to training_metrics.csv")


def main():
    """Main function to run the full CNN-GNN training with comprehensive metrics"""

    # Configuration
    config = {
        "data_dir": "./data",
        "sequence_length": 16,
        "image_size": (224, 224),
        "temporal_window": 3,
        "batch_size": 4,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "cnn_feature_dim": 512,
        "gnn_hidden_dim": 256,
        "gnn_layers": 3,
        "num_heads": 8,
        "max_sequences": None,  # No limit - use full dataset
    }

    print("Initializing FULL CNN-GNN Video Classification Model")
    print("=" * 60)
    print(f"Configuration: {config}")

    # Create datasets
    print("\nCreating datasets...")
    try:
        train_dataset = VideoDataset(
            data_dir=config["data_dir"],
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            temporal_window=config["temporal_window"],
            train=True,
            max_sequences=config["max_sequences"],
        )

        if len(train_dataset) == 0:
            print("No valid sequences found!")
            return

        # Split dataset
        train_size = max(1, int(0.8 * len(train_dataset)))
        val_size = len(train_dataset) - train_size

        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        print(f"Train samples: {len(train_subset)}")
        print(f"Val samples: {len(val_subset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Create model
        print(f"\nCreating CNN-GNN model with {train_dataset.num_classes} classes...")
        model = CNNGNN(
            num_classes=train_dataset.num_classes,
            cnn_feature_dim=config["cnn_feature_dim"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            gnn_layers=config["gnn_layers"],
            num_heads=config["num_heads"],
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=train_dataset.num_classes,
            class_names=list(train_dataset.label_to_idx.keys()),
            learning_rate=config["learning_rate"],
        )

        # Train model
        print("\nStarting training...")
        best_f1_score, best_metrics = trainer.train(
            num_epochs=config["num_epochs"], save_path="cnn_gnn_full_model.pth"
        )

        # Final evaluation
        final_metrics = trainer.evaluate_final_model(save_plots=True)

        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best F1 Score (Macro): {best_f1_score:.4f}")
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"Model saved as: cnn_gnn_full_model.pth")
        print(f"Metrics saved as: training_metrics.csv")
        print(
            f"Plots saved as: comprehensive_training_history.png, final_confusion_matrix.png"
        )

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
