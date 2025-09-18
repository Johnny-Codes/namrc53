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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import warnings

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
            if i == 0:
                self.gat_layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=0.2,
                        concat=True,
                    )
                )
            else:
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
    State-of-the-art CNN-GNN model for video behavior classification
    Combines spatial CNN features with temporal GNN modeling
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

        # Multi-scale temporal features
        self.temporal_conv1d = nn.Sequential(
            nn.Conv1d(gnn_hidden_dim, gnn_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(gnn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(gnn_hidden_dim, gnn_hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(gnn_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Final classifier with attention
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 2, gnn_hidden_dim),  # *2 for multi-scale
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout // 2),
            nn.Linear(gnn_hidden_dim // 2, num_classes),
        )

        # Self-attention for final features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=gnn_hidden_dim, num_heads=8, dropout=dropout
        )

    def forward(self, frames, edge_indices, batch_indices):
        """
        Forward pass
        Args:
            frames: (batch_size * sequence_length, 3, H, W) - video frames
            edge_indices: list of edge_index tensors for each sequence
            batch_indices: list of batch assignment tensors
        """
        batch_size = len(edge_indices)

        # Extract spatial features from all frames
        spatial_features = self.spatial_cnn(
            frames
        )  # (batch_size * seq_len, feature_dim)

        # Process each sequence through GNN
        gnn_outputs = []

        for i in range(batch_size):
            # Get features for this sequence
            start_idx = i * spatial_features.size(0) // batch_size
            end_idx = (i + 1) * spatial_features.size(0) // batch_size
            seq_features = spatial_features[start_idx:end_idx]

            # Apply GNN
            gnn_output = self.temporal_gnn(
                seq_features, edge_indices[i], batch_indices[i]
            )
            gnn_outputs.append(gnn_output)

        # Stack GNN outputs
        temporal_features = torch.stack(gnn_outputs)  # (batch_size, gnn_hidden_dim)

        # Apply 1D convolution for multi-scale temporal modeling
        # Reshape for conv1d: (batch_size, channels, sequence_length)
        conv_input = temporal_features.unsqueeze(-1)  # Add sequence dimension
        conv_features = self.temporal_conv1d(conv_input.transpose(1, 2))
        conv_features = conv_features.transpose(1, 2).squeeze(-1)

        # Combine original and conv features
        combined_features = torch.cat([temporal_features, conv_features], dim=1)

        # Apply self-attention
        attn_features, _ = self.self_attention(
            combined_features.unsqueeze(1),  # Add sequence dim
            combined_features.unsqueeze(1),
            combined_features.unsqueeze(1),
        )
        attn_features = attn_features.squeeze(1)

        # Final classification
        logits = self.classifier(attn_features)

        return logits


class VideoDataset(Dataset):
    """
    Dataset class for loading video sequences with temporal relationships
    """

    def __init__(
        self,
        data_dir,
        sequence_length=16,
        image_size=(224, 224),
        temporal_window=5,
        train=True,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.temporal_window = temporal_window
        self.train = train

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

        # Process each training directory
        train_dirs = sorted(self.data_dir.glob("train-*"))

        for train_dir in train_dirs[:10]:  # Limit for testing
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

            # Create sequences
            for i in range(0, len(frames_data), self.sequence_length):
                sequence_frames = frames_data[i : i + self.sequence_length]

                if (
                    len(sequence_frames) >= self.sequence_length // 2
                ):  # At least half sequence
                    sequences.append(
                        {"frames": sequence_frames, "train_dir": train_dir}
                    )

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
                if i != j:
                    edge_indices.append([i, j])

        # Convert to tensor
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

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

        # Load frames
        frames = []
        labels = []

        for frame_data in frames_data:
            # Load frame
            frame = self._load_frame(train_dir, frame_data["file_name"])
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

            # Get label
            label_idx = self.label_to_idx.get(frame_data["action_label"], 0)
            labels.append(label_idx)

        # Pad sequence if needed
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))
            labels.append(0)  # Padding label

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


class Trainer:
    """Training and evaluation manager"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_classes,
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            self.optimizer.zero_grad()

            # Forward pass
            frames = batch["frames"].to(device)
            labels = batch["labels"].to(device)
            edge_indices = [ei.to(device) for ei in batch["edge_indices"]]
            batch_indices = [bi.to(device) for bi in batch["batch_indices"]]

            try:
                logits = self.model(frames, edge_indices, batch_indices)
                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")

            for batch in pbar:
                frames = batch["frames"].to(device)
                labels = batch["labels"].to(device)
                edge_indices = [ei.to(device) for ei in batch["edge_indices"]]
                batch_indices = [bi.to(device) for bi in batch["batch_indices"]]

                try:
                    logits = self.model(frames, edge_indices, batch_indices)
                    loss = self.criterion(logits, labels)

                    total_loss += loss.item()
                    num_batches += 1

                    # Get predictions
                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = accuracy_score(all_labels, all_predictions)

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self, num_epochs=50, save_path="best_model.pth"):
        """Full training loop"""
        best_accuracy = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_accuracy, predictions, labels = self.validate()

            # Update scheduler
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "accuracy": val_accuracy,
                        "loss": val_loss,
                    },
                    save_path,
                )
                print(f"New best model saved with accuracy: {val_accuracy:.4f}")

        print(f"\nTraining completed! Best accuracy: {best_accuracy:.4f}")
        return best_accuracy

    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curve
        ax2.plot(self.val_accuracies, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """Main function to run the CNN-GNN training"""

    # Configuration
    config = {
        "data_dir": "./data",
        "sequence_length": 16,
        "image_size": (224, 224),
        "temporal_window": 3,
        "batch_size": 4,  # Small batch size due to memory constraints
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "cnn_feature_dim": 512,
        "gnn_hidden_dim": 256,
        "gnn_layers": 3,
        "num_heads": 8,
    }

    print("Initializing CNN-GNN Video Classification Model")
    print("=" * 50)
    print(f"Configuration: {config}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = VideoDataset(
        data_dir=config["data_dir"],
        sequence_length=config["sequence_length"],
        image_size=config["image_size"],
        temporal_window=config["temporal_window"],
        train=True,
    )

    # Split dataset (simple split for demo)
    train_size = int(0.8 * len(train_dataset))
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
        num_workers=0,  # Set to 0 for Windows compatibility
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
        learning_rate=config["learning_rate"],
    )

    # Train model
    print("\nStarting training...")
    best_accuracy = trainer.train(
        num_epochs=config["num_epochs"], save_path="cnn_gnn_best_model.pth"
    )

    # Plot training history
    trainer.plot_training_history()

    # Print final results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Model saved as: cnn_gnn_best_model.pth")
    print(f"Training history plot saved as: training_history.png")

    # Test inference on a sample
    print("\nTesting inference...")
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        frames = sample_batch["frames"].to(device)
        labels = sample_batch["labels"].to(device)
        edge_indices = [ei.to(device) for ei in sample_batch["edge_indices"]]
        batch_indices = [bi.to(device) for bi in sample_batch["batch_indices"]]

        try:
            logits = model(frames, edge_indices, batch_indices)
            predictions = torch.argmax(logits, dim=1)

            print(f"Sample predictions: {predictions.cpu().numpy()}")
            print(f"True labels: {labels.cpu().numpy()}")
            print(
                f"Predicted classes: {[train_dataset.idx_to_label[p.item()] for p in predictions]}"
            )
            print(
                f"True classes: {[train_dataset.idx_to_label[l.item()] for l in labels]}"
            )

        except Exception as e:
            print(f"Error in inference: {e}")


if __name__ == "__main__":
    main()
