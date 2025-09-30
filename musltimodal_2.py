import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import cv2
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import random
from datetime import datetime
import time

# Import your utility functions
from utils.data_utils import StratifiedVideoDatasetSplitter
from utils.loss_utils import WeightedLossCalculator, FocalLoss, LabelSmoothingLoss
from utils.metrics import calculate_comprehensive_metrics, calculate_per_class_metrics
from utils.visualization import create_class_weights_plots, create_training_curves
from utils.results_saver import ResultsSaver

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VideoFeatureExtractor(nn.Module):
    """
    Placeholder for a powerful feature extractor (e.g., frozen Diffusion Model or ViT).
    For this implementation, uses a simple CNN-based feature extractor.
    """

    def __init__(self, in_channels=3, feature_dim=512):
        super(VideoFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim

        # Simple CNN backbone (placeholder for more sophisticated models)
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(256, feature_dim), nn.ReLU(inplace=True), nn.Dropout(0.2)
        )

    def forward(self, video_frames):
        """
        Process video frames to extract features.

        Input: video_frames (Batch, Num_Frames, Channels, Height, Width)
        Output: (Batch, Num_Frames, feature_dim)
        """
        batch_size, num_frames, channels, height, width = video_frames.shape

        # Reshape to process all frames together: (B*T, C, H, W)
        frames_flat = video_frames.view(
            batch_size * num_frames, channels, height, width
        )

        # Extract features for all frames
        features = self.backbone(frames_flat)  # (B*T, 256, 1, 1)
        features = features.view(batch_size * num_frames, -1)  # (B*T, 256)

        # Project to desired feature dimension
        projected_features = self.feature_projector(features)  # (B*T, feature_dim)

        # Reshape back to sequence format: (B, T, feature_dim)
        output_features = projected_features.view(
            batch_size, num_frames, self.feature_dim
        )

        return output_features


class VideoTransformerEncoder(nn.Module):
    """
    Transformer encoder to model temporal relationships between video frame features.
    """

    def __init__(
        self, feature_dim=512, n_heads=8, n_layers=4, dim_feedforward=2048, dropout=0.1
    ):
        super(VideoTransformerEncoder, self).__init__()

        # Positional encoding for temporal information
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1000, feature_dim) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, video_features):
        """
        Model temporal relationships in video features.

        Input: video_features (Batch, Num_Frames, feature_dim)
        Output: (Batch, Num_Frames, feature_dim)
        """
        batch_size, num_frames, feature_dim = video_features.shape

        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :num_frames, :].expand(
            batch_size, -1, -1
        )
        video_features = video_features + pos_encoding

        # Apply transformer encoder
        encoded_features = self.transformer_encoder(video_features)

        # Layer normalization
        output = self.layer_norm(encoded_features)

        return output


class HyperSAEncoder(nn.Module):
    """
    Conceptual placeholder for Hypergraph Self-Attention model.
    Implemented as a standard Transformer Encoder for skeleton feature processing.
    """

    def __init__(
        self, input_feature_dim, model_dim=512, n_heads=8, n_layers=4, dropout=0.1
    ):
        super(HyperSAEncoder, self).__init__()

        self.model_dim = model_dim

        # Project input features to model dimension
        self.input_projection = nn.Linear(input_feature_dim, model_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, model_dim) * 0.02)

        # Transformer encoder for skeleton sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, skeleton_data):
        """
        Learn temporal dependencies in skeleton sequence.

        Input: skeleton_data (Batch, Num_Frames, Num_Joints * Joint_Features)
        Output: (Batch, Num_Frames, model_dim)
        """
        batch_size, num_frames, input_dim = skeleton_data.shape

        # Project to model dimension
        projected_features = self.input_projection(skeleton_data)  # (B, T, model_dim)

        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :num_frames, :].expand(
            batch_size, -1, -1
        )
        skeleton_features = projected_features + pos_encoding

        # Apply transformer encoder
        encoded_features = self.transformer_encoder(skeleton_features)

        # Layer normalization
        output = self.layer_norm(encoded_features)

        return output


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where skeleton features query video features.
    """

    def __init__(self, query_dim=512, kv_dim=512, n_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.query_dim = query_dim
        self.kv_dim = kv_dim

        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization and residual connection
        self.layer_norm1 = nn.LayerNorm(query_dim)
        self.layer_norm2 = nn.LayerNorm(query_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, skeleton_features, video_features):
        """
        Perform cross-attention fusion.

        Input:
            skeleton_features (query): (B, T, query_dim)
            video_features (key/value): (B, T, kv_dim)
        Output: (Batch, query_dim)
        """
        # Cross-attention: skeleton queries video
        attended_features, attention_weights = self.cross_attention(
            query=skeleton_features, key=video_features, value=video_features
        )

        # Residual connection and layer norm
        attended_features = self.layer_norm1(attended_features + skeleton_features)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(attended_features)
        fused_features = self.layer_norm2(ffn_output + attended_features)

        # Global average pooling over time dimension
        pooled_features = torch.mean(fused_features, dim=1)  # (B, query_dim)

        return pooled_features


class ActionRecognitionModel(nn.Module):
    """
    Main two-stream model integrating video and skeleton processing with cross-attention fusion.
    """

    def __init__(
        self,
        num_classes,
        video_feature_dim=512,
        skeleton_model_dim=512,
        skeleton_input_dim=96,  # Updated to match actual skeleton data (32 joints * 3 coordinates)
        video_transformer_layers=4,
        skeleton_transformer_layers=4,
        n_heads=8,
        dropout=0.1,
    ):
        super(ActionRecognitionModel, self).__init__()

        self.num_classes = num_classes

        # Video stream components
        self.video_feature_extractor = VideoFeatureExtractor(
            in_channels=3, feature_dim=video_feature_dim
        )

        self.video_transformer = VideoTransformerEncoder(
            feature_dim=video_feature_dim,
            n_heads=n_heads,
            n_layers=video_transformer_layers,
            dropout=dropout,
        )

        # Skeleton stream components
        self.skeleton_encoder = HyperSAEncoder(
            input_feature_dim=skeleton_input_dim,
            model_dim=skeleton_model_dim,
            n_heads=n_heads,
            n_layers=skeleton_transformer_layers,
            dropout=dropout,
        )

        # Cross-attention fusion
        self.cross_attention_fusion = CrossAttentionFusion(
            query_dim=skeleton_model_dim,
            kv_dim=video_feature_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(skeleton_model_dim, skeleton_model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(skeleton_model_dim // 2, skeleton_model_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(skeleton_model_dim // 4, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, video_frames, skeleton_data):
        """
        Forward pass through the two-stream model.

        Input:
            video_frames: (Batch, Num_Frames, Channels, Height, Width)
            skeleton_data: (Batch, Num_Frames, Num_Joints * Joint_Features)
        Output: (Batch, num_classes)
        """
        # Video stream processing
        video_features = self.video_feature_extractor(
            video_frames
        )  # (B, T, video_feature_dim)
        video_encoded = self.video_transformer(
            video_features
        )  # (B, T, video_feature_dim)

        # Skeleton stream processing
        skeleton_encoded = self.skeleton_encoder(
            skeleton_data
        )  # (B, T, skeleton_model_dim)

        # Cross-attention fusion (skeleton queries video)
        fused_features = self.cross_attention_fusion(
            skeleton_features=skeleton_encoded, video_features=video_encoded
        )  # (B, skeleton_model_dim)

        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)

        return logits


class MultimodalActionDataset(Dataset):
    """
    Dataset class for loading multimodal action recognition data.
    Processes both video frames and skeleton data from the specified directory structure.
    """

    def __init__(
        self,
        data_dir,
        sequence_length=16,
        image_size=(224, 224),
        train=True,
        max_sequences=None,  # Limit sequences for testing
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.train = train
        self.max_sequences = max_sequences

        # Load sequences
        self.sequences = self._load_sequences()

        # Create label mappings
        all_labels = set()
        for seq in self.sequences:
            all_labels.update([item["action_label"] for item in seq["frames"]])

        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        print(f"Loaded {len(self.sequences)} sequences with {self.num_classes} classes")
        print(f"Classes: {list(self.label_to_idx.keys())}")

        # Count skeleton joints from first sequence
        if len(self.sequences) > 0:
            first_frame = self.sequences[0]["frames"][0]
            if "skeleton" in first_frame:
                self.num_joints = len(first_frame["skeleton"])
                self.joint_features = len(first_frame["skeleton"][0])
                print(
                    f"Skeleton data: {self.num_joints} joints with {self.joint_features} features each"
                )
            else:
                self.num_joints = 32
                self.joint_features = 3
                print("No skeleton data found, using synthetic data")

        # Transforms for video frames
        if train:
            self.video_transform = transforms.Compose(
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
            self.video_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _load_sequences(self):
        """Load sequences from the data directory structure"""
        sequences = []
        train_dirs = sorted(self.data_dir.glob("train-*"))

        print(f"Processing {len(train_dirs)} training directories...")

        for train_dir in tqdm(train_dirs, desc="Loading sequences"):
            metadata_file = train_dir / "metadata.jsonl"
            if not metadata_file.exists():
                continue

            # Load metadata
            frames_data = []
            with open(metadata_file, "r") as f:
                for line in f:
                    try:
                        frame_data = json.loads(line.strip())
                        if all(
                            field in frame_data
                            for field in ["file_name", "action_label", "action_number"]
                        ):
                            frame_data["train_dir"] = train_dir
                            frames_data.append(frame_data)
                    except Exception as e:
                        print(f"Error parsing line in {metadata_file}: {e}")
                        continue

            # Sort by action number
            frames_data.sort(key=lambda x: x.get("action_number", 0))

            # Create sequences
            step_size = max(
                1, self.sequence_length // 4
            )  # More overlap for better sampling
            for i in range(0, len(frames_data) - self.sequence_length + 1, step_size):
                sequence_frames = frames_data[i : i + self.sequence_length]

                if len(sequence_frames) == self.sequence_length:
                    sequences.append(
                        {"frames": sequence_frames, "train_dir": train_dir}
                    )

                    # Limit sequences if specified
                    if self.max_sequences and len(sequences) >= self.max_sequences:
                        return sequences

        return sequences

    def _load_video_frame(self, train_dir, file_name):
        """Load and process a video frame"""
        frame_path = train_dir / file_name

        if not frame_path.exists():
            # Return a black frame if file doesn't exist
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
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                frame = np.zeros(
                    (self.image_size[0], self.image_size[1], 3), dtype=np.uint8
                )

        return frame

    def _load_skeleton_data(self, frame_data):
        """Load skeleton data from metadata"""
        try:
            if "skeleton" in frame_data:
                # Convert skeleton data from strings to floats
                skeleton = frame_data["skeleton"]
                skeleton_array = []

                for joint in skeleton:
                    joint_coords = [float(coord) for coord in joint]
                    skeleton_array.extend(joint_coords)

                return np.array(skeleton_array, dtype=np.float32)
            else:
                # Fallback to synthetic data if no skeleton data
                return np.random.randn(32 * 3).astype(np.float32)

        except Exception as e:
            print(f"Error loading skeleton data: {e}")
            # Fallback to synthetic data
            return np.random.randn(32 * 3).astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        frames_data = sequence["frames"]
        train_dir = sequence["train_dir"]

        # Load video frames
        video_frames = []
        skeleton_frames = []
        labels = []

        for frame_data in frames_data:
            # Load video frame
            frame = self._load_video_frame(train_dir, frame_data["file_name"])
            frame_tensor = self.video_transform(frame)
            video_frames.append(frame_tensor)

            # Load skeleton data from metadata
            skeleton_data = self._load_skeleton_data(frame_data)
            skeleton_frames.append(skeleton_data)

            # Get label
            label_idx = self.label_to_idx.get(frame_data["action_label"], 0)
            labels.append(label_idx)

        # Stack frames
        video_tensor = torch.stack(video_frames)  # (T, C, H, W)
        skeleton_tensor = torch.stack(
            [torch.from_numpy(s) for s in skeleton_frames]
        )  # (T, skeleton_dim)

        # Get sequence label (majority vote)
        label_counts = Counter(labels)
        sequence_label = max(label_counts.keys(), key=lambda x: label_counts[x])

        return {
            "video_frames": video_tensor,
            "skeleton_data": skeleton_tensor,
            "label": sequence_label,
            "sequence_id": idx,
        }

    @property
    def sequence_labels(self):
        """Get sequence labels for stratification (used by StratifiedVideoDatasetSplitter)"""
        labels = []
        for sequence in self.sequences:
            frames_data = sequence["frames"]
            frame_labels = [item["action_label"] for item in frames_data]
            majority_label = max(set(frame_labels), key=frame_labels.count)
            labels.append(majority_label)
        return labels


def collate_multimodal_fn(batch):
    """Custom collate function for multimodal data"""
    video_frames = torch.stack([item["video_frames"] for item in batch])
    skeleton_data = torch.stack([item["skeleton_data"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    sequence_ids = [item["sequence_id"] for item in batch]

    return {
        "video_frames": video_frames,
        "skeleton_data": skeleton_data,
        "labels": labels,
        "sequence_ids": sequence_ids,
    }


class MultimodalTrainer:
    """Enhanced trainer with comprehensive results saving and metrics tracking"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_names,
        class_counts,
        learning_rate=1e-4,
        weight_decay=1e-4,
        results_saver=None,
        device=device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.class_counts = class_counts
        self.device = device
        self.results_saver = results_saver

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=30,
        )

        # Setup loss function with class weights
        weight_calculator = WeightedLossCalculator(
            weighting_strategy="inverse_sqrt", rare_class_boost=2.0
        )
        self.class_weights = weight_calculator.calculate_class_weights(
            class_counts, num_classes, device
        )
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Print weight analysis
        weight_calculator.print_weight_analysis(
            self.class_weights, class_names, class_counts
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"✅ Trainer initialized with weighted CrossEntropy loss")
        print(
            f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
        )

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            video_frames = batch["video_frames"].to(self.device, non_blocking=True)
            skeleton_data = batch["skeleton_data"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(video_frames, skeleton_data)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            elapsed = time.time() - start_time
            it_per_sec = (batch_idx + 1) / elapsed
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{100.*correct/total:.2f}%",
                    "it/s": f"{it_per_sec:.2f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def evaluate(self, data_loader, split_name="val"):
        """Evaluate on validation or test set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")
            for batch in pbar:
                video_frames = batch["video_frames"].to(self.device, non_blocking=True)
                skeleton_data = batch["skeleton_data"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = self.model(video_frames, skeleton_data)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            all_labels,
            all_predictions,
            np.array(all_probabilities) if all_probabilities else None,
            avg_loss,
        )

        return metrics, all_predictions, all_labels, all_probabilities

    def train(self, num_epochs=30):
        """Full training loop with comprehensive logging"""
        best_f1 = 0
        print(f"\n🎯 Starting multimodal training for {num_epochs} epochs...")
        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Evaluate
            val_metrics, val_predictions, val_labels, val_probabilities = self.evaluate(
                self.val_loader, "validation"
            )
            test_metrics, test_predictions, test_labels, test_probabilities = (
                self.evaluate(self.test_loader, "test")
            )

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log results
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
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

            # Save best model
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]

                if self.results_saver:
                    model_save_path = (
                        self.results_saver.experiment_dir
                        / "models"
                        / "best_multimodal_model.pth"
                    )
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
                            "class_weights": self.class_weights,
                        },
                        model_save_path,
                    )

                    print(
                        f"✓ Best model saved (Val F1: {best_f1:.4f}, Test F1: {test_metrics['f1_macro']:.4f})"
                    )

                    # Save final results for best model
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
        print(f"\n🎉 Training completed in {total_time:.1f} seconds")
        print(f"Best Validation F1 Score: {best_f1:.4f}")

        return best_f1, val_metrics, test_metrics


def run_multimodal_training():
    """Main function to run multimodal training with comprehensive results saving"""

    print("🎯 MULTIMODAL TWO-STREAM ACTION RECOGNITION TRAINING")
    print("=" * 80)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Configuration
    config = {
        "data_dir": "./data",
        "sequence_length": 16,
        "image_size": (224, 224),
        "batch_size": 8,
        "num_epochs": 25,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "video_feature_dim": 512,
        "skeleton_model_dim": 512,
        "video_transformer_layers": 4,
        "skeleton_transformer_layers": 4,
        "n_heads": 8,
        "dropout": 0.1,
        "seed": 42,
    }

    print(f"Configuration: {config}")

    try:
        # Create dataset to determine dimensions
        print("\n📂 Creating dataset...")
        dataset = MultimodalActionDataset(
            data_dir=config["data_dir"],
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            train=True,
            max_sequences=None,  # Load full dataset
        )

        if len(dataset) == 0:
            print("❌ No sequences found in dataset")
            return

        # Get actual dimensions from dataset
        NUM_CLASSES = dataset.num_classes
        NUM_JOINTS = dataset.num_joints
        JOINT_FEATURES = dataset.joint_features
        SKELETON_INPUT_DIM = NUM_JOINTS * JOINT_FEATURES

        print(f"✅ Dataset loaded: {len(dataset)} sequences")
        print(f"📊 Classes: {NUM_CLASSES}")
        print(
            f"🦴 Skeleton: {NUM_JOINTS} joints × {JOINT_FEATURES} features = {SKELETON_INPUT_DIM}D"
        )

        # Count class distribution
        sequence_labels = dataset.sequence_labels
        class_counts = Counter(sequence_labels)
        class_names = sorted(list(class_counts.keys()))
        class_counts_array = [class_counts[name] for name in class_names]

        print(f"📈 Class distribution:")
        for name, count in zip(class_names, class_counts_array):
            print(f"  {name}: {count} sequences")

        # Create stratified splits
        print("\n🎯 Creating stratified train/val/test splits...")
        splitter = StratifiedVideoDatasetSplitter(
            dataset=dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        train_indices, val_indices, test_indices = splitter.split()

        print(
            f"✅ Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
        )

        # Create subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_multimodal_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_multimodal_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_multimodal_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # Create model
        print(f"\n🏗️ Creating multimodal model...")
        model = ActionRecognitionModel(
            num_classes=NUM_CLASSES,
            video_feature_dim=config["video_feature_dim"],
            skeleton_model_dim=config["skeleton_model_dim"],
            skeleton_input_dim=SKELETON_INPUT_DIM,
            video_transformer_layers=config["video_transformer_layers"],
            skeleton_transformer_layers=config["skeleton_transformer_layers"],
            n_heads=config["n_heads"],
            dropout=config["dropout"],
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")

        # Initialize results saver
        experiment_name = (
            f"multimodal_2stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        results_saver = ResultsSaver(
            experiment_name=experiment_name,
            config=config,
            model_architecture=str(model),
        )

        print(f"📁 Results will be saved to: {results_saver.experiment_dir}")

        # Create class weights visualization
        try:
            create_class_weights_plots(
                class_weights=None,  # Will be calculated in trainer
                class_names=class_names,
                class_counts=class_counts_array,
                save_dir=results_saver.experiment_dir / "figures",
            )
            print("✅ Class distribution plots created")
        except Exception as e:
            print(f"⚠️ Could not create class plots: {e}")

        # Initialize trainer
        print(f"\n🏋️ Initializing multimodal trainer...")
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=NUM_CLASSES,
            class_names=class_names,
            class_counts=class_counts_array,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            results_saver=results_saver,
            device=device,
        )

        # Start training
        print(f"\n🚀 Starting training...")
        best_f1, final_val_metrics, final_test_metrics = trainer.train(
            num_epochs=config["num_epochs"]
        )

        # Create training curves
        try:
            create_training_curves(
                results_file=results_saver.experiment_dir / "training_log.csv",
                save_dir=results_saver.experiment_dir / "figures",
            )
            print("✅ Training curves created")
        except Exception as e:
            print(f"⚠️ Could not create training curves: {e}")

        # Final summary
        print(f"\n🎯 TRAINING COMPLETE!")
        print(f"{'='*50}")
        print(f"Best Validation F1 Score: {best_f1:.4f}")
        print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
        print(f"Final Test F1 Score: {final_test_metrics['f1_macro']:.4f}")
        print(f"Results saved to: {results_saver.experiment_dir}")
        print(f"{'='*50}")

        return {
            "best_val_f1": best_f1,
            "final_test_metrics": final_test_metrics,
            "experiment_dir": results_saver.experiment_dir,
        }

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete multimodal training pipeline
    results = run_multimodal_training()

    if results:
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Best Validation F1: {results['best_val_f1']:.4f}")
        print(f"🎯 Test F1: {results['final_test_metrics']['f1_macro']:.4f}")
        print(f"📁 Results: {results['experiment_dir']}")
    else:
        print(f"\n❌ Training failed!")
