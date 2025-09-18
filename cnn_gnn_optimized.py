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
import time

warnings.filterwarnings("ignore")

# Set device and optimize CUDA settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(
        f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )


class OptimizedSpatialCNN(nn.Module):
    """
    Optimized CNN backbone with reduced complexity
    """

    def __init__(self, feature_dim=256, pretrained=True):  # Reduced from 512
        super(OptimizedSpatialCNN, self).__init__()

        # Use a lighter backbone or reduce EfficientNet complexity
        if pretrained:
            self.backbone = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Remove classifier
        self.backbone.classifier = nn.Identity()
        backbone_dim = 1280

        # Simplified feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Reduced dropout
        )

        # Simplified attention (optional - can be removed for speed)
        self.use_attention = False  # Set to False for speed
        if self.use_attention:
            self.spatial_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(backbone_dim, backbone_dim // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(backbone_dim // 16, backbone_dim, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)

        if self.use_attention:
            attention = self.spatial_attention(features)
            features = features * attention

        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Project features
        projected_features = self.feature_projector(pooled_features)

        return projected_features


class OptimizedTemporalGNN(nn.Module):
    """
    Simplified and optimized GNN module
    """

    def __init__(
        self, feature_dim=256, hidden_dim=128, num_layers=2, num_heads=4
    ):  # Reduced complexity
        super(OptimizedTemporalGNN, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Simplified GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=0.1,
                    concat=True,
                )  # Reduced dropout
            )

        # Simplified pooling
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x, edge_index, batch):
        # Input projection
        h = self.input_proj(x)

        # Apply GAT layers (no residual connections for speed)
        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.relu(h)

        # Global pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_combined = torch.cat([h_mean, h_max], dim=1)

        # Final pooling
        temporal_features = self.temporal_pool(h_combined)

        return temporal_features


class OptimizedCNNGNN(nn.Module):
    """
    Optimized CNN-GNN model for faster training
    """

    def __init__(
        self,
        num_classes,
        cnn_feature_dim=256,
        gnn_hidden_dim=128,
        gnn_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super(OptimizedCNNGNN, self).__init__()

        self.num_classes = num_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        # Optimized components
        self.spatial_cnn = OptimizedSpatialCNN(feature_dim=cnn_feature_dim)
        self.temporal_gnn = OptimizedTemporalGNN(
            feature_dim=cnn_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
        )

        # Simplified classifier (no self-attention for speed)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, num_classes),
        )

    def forward(self, frames, edge_indices, batch_indices, verbose=False):
        batch_size = len(edge_indices)

        # Extract spatial features with mixed precision
        with torch.cuda.amp.autocast():
            spatial_features = self.spatial_cnn(frames)

        # Calculate sequence length
        total_frames = spatial_features.size(0)
        seq_len = total_frames // batch_size

        # Process sequences through GNN
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
                # Fallback
                fallback_output = torch.mean(seq_features, dim=0, keepdim=True)
                fallback_output = self.temporal_gnn.temporal_pool(
                    torch.cat([fallback_output, fallback_output], dim=1)
                )
                gnn_outputs.append(fallback_output)

        # Stack and classify
        if gnn_outputs:
            temporal_features = torch.stack(
                [out.squeeze(0) if out.dim() > 1 else out for out in gnn_outputs]
            )
        else:
            temporal_features = torch.zeros(
                batch_size, self.gnn_hidden_dim, device=frames.device
            )

        # Final classification
        logits = self.classifier(temporal_features)

        return logits


class OptimizedVideoDataset(Dataset):
    """
    Optimized dataset with caching and faster loading
    """

    def __init__(
        self,
        data_dir,
        sequence_length=8,
        image_size=(224, 224),  # Reduced sequence length
        temporal_window=2,
        train=True,
        max_sequences=None,  # Reduced temporal window
        cache_frames=True,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.temporal_window = temporal_window
        self.train = train
        self.max_sequences = max_sequences
        self.cache_frames = cache_frames
        self.frame_cache = {}  # Cache for frequently accessed frames

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

        # Optimized transforms
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1
                    ),  # Reduced augmentation
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
        """Optimized sequence creation with early stopping"""
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

            # Create sequences with larger step size for speed
            step_size = self.sequence_length  # No overlap for speed
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
        """Simplified graph creation"""
        num_nodes = len(sequence_frames)
        edge_indices = []

        # Only sequential connections for speed
        for i in range(num_nodes - 1):
            edge_indices.append([i, i + 1])
            edge_indices.append([i + 1, i])

        # Reduced window connections
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
        """Optimized frame loading with caching"""
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

        # Cache if enabled and cache isn't too large
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

        # Majority vote for sequence label
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


class OptimizedTrainer:
    """
    Optimized trainer with mixed precision and faster training
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ):  # Higher LR for faster convergence
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names

        # Optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Faster scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=50,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []

        print("Using mixed precision training for speed optimization")

    def train_epoch(self):
        """Optimized training epoch with mixed precision"""
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
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    logits = self.model(frames, edge_indices, batch_indices)
                    loss = self.criterion(logits, labels)

                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                # Calculate speed
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
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Fast validation"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

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

                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
        f1_macro = (
            f1_score(all_labels, all_predictions, average="macro", zero_division=0)
            if all_labels
            else 0.0
        )

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1_macro)

        return avg_loss, accuracy, f1_macro

    def train(self, num_epochs=30, save_path="optimized_model.pth"):  # Reduced epochs
        """Fast training loop"""
        best_f1 = 0

        print(f"Starting optimized training for {num_epochs} epochs...")
        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, accuracy, f1_macro = self.validate()

            epoch_time = time.time() - epoch_start_time

            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {accuracy:.4f}")
            print(f"Val F1: {f1_macro:.4f}")

            # Save best model
            if f1_macro > best_f1:
                best_f1 = f1_macro
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "f1_score": f1_macro,
                        "accuracy": accuracy,
                    },
                    save_path,
                )
                print(f"✓ Best model saved (F1: {best_f1:.4f})")

        total_time = time.time() - total_start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        print(f"Best F1 Score: {best_f1:.4f}")

        return best_f1


def main():
    """Optimized main function"""

    # Optimized configuration for speed
    config = {
        "data_dir": "./data",
        "sequence_length": 8,  # Reduced from 16
        "image_size": (224, 224),
        "temporal_window": 2,  # Reduced from 3
        "batch_size": 8,  # Increased from 4
        "num_epochs": 30,  # Reduced from 50
        "learning_rate": 1e-3,  # Increased from 1e-4
        "cnn_feature_dim": 256,  # Reduced from 512
        "gnn_hidden_dim": 128,  # Reduced from 256
        "gnn_layers": 2,  # Reduced from 3
        "num_heads": 4,  # Reduced from 8
        "max_sequences": None,
    }

    print("🚀 OPTIMIZED CNN-GNN Video Classification")
    print("=" * 50)
    print(f"Configuration: {config}")

    try:
        # Create optimized dataset
        print("\nCreating optimized dataset...")
        dataset = OptimizedVideoDataset(
            data_dir=config["data_dir"],
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            temporal_window=config["temporal_window"],
            train=True,
            max_sequences=config["max_sequences"],
            cache_frames=True,
        )

        if len(dataset) == 0:
            print("No sequences found!")
            return

        # Split dataset
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

        # Optimized data loaders
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

        # Create optimized model
        print(f"\nCreating optimized model ({dataset.num_classes} classes)...")
        model = OptimizedCNNGNN(
            num_classes=dataset.num_classes,
            cnn_feature_dim=config["cnn_feature_dim"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            gnn_layers=config["gnn_layers"],
            num_heads=config["num_heads"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Create optimized trainer
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=dataset.num_classes,
            class_names=list(dataset.label_to_idx.keys()),
            learning_rate=config["learning_rate"],
        )

        # Train with optimizations
        print("\n🏃‍♂️ Starting speed-optimized training...")
        best_f1 = trainer.train(
            num_epochs=config["num_epochs"], save_path="optimized_cnn_gnn_model.pth"
        )

        print(f"\n🎉 Training completed!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"Model saved as: optimized_cnn_gnn_model.pth")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


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
