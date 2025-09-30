import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import cv2
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import random

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


# Training utilities
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        video_frames = batch["video_frames"].to(device)
        skeleton_data = batch["skeleton_data"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(video_frames, skeleton_data)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
        )

    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            video_frames = batch["video_frames"].to(device)
            skeleton_data = batch["skeleton_data"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(video_frames, skeleton_data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Test with actual data if available
    data_dir = "./data"
    if Path(data_dir).exists():
        print(f"=== Testing with Real Data from {data_dir} ===")

        try:
            # First, create a small dataset to determine the actual number of classes and skeleton dimensions
            print("Creating dataset to determine data dimensions...")
            dataset = MultimodalActionDataset(
                data_dir=data_dir,
                sequence_length=16,
                image_size=(224, 224),
                train=True,
                # max_sequences=100,  # Limit for initial testing
            )

            if len(dataset) == 0:
                print("No sequences found in dataset")
                exit(1)

            # Get actual dimensions from dataset
            NUM_CLASSES = dataset.num_classes
            NUM_JOINTS = dataset.num_joints
            JOINT_FEATURES = dataset.joint_features
            SKELETON_INPUT_DIM = NUM_JOINTS * JOINT_FEATURES

            print(f"\n=== Dataset Information ===")
            print(f"Number of classes: {NUM_CLASSES}")
            print(f"Number of joints: {NUM_JOINTS}")
            print(f"Joint features: {JOINT_FEATURES}")
            print(f"Skeleton input dimension: {SKELETON_INPUT_DIM}")

            # Hyperparameters
            BATCH_SIZE = 4
            NUM_FRAMES = 16
            IMAGE_HEIGHT = 224
            IMAGE_WIDTH = 224
            CHANNELS = 3

            VIDEO_FEATURE_DIM = 512
            SKELETON_MODEL_DIM = 512

            print(f"\n=== Model Configuration ===")
            print(f"Batch size: {BATCH_SIZE}")
            print(f"Number of frames per sequence: {NUM_FRAMES}")
            print(f"Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{CHANNELS}")

            # Instantiate the main model with correct number of classes
            model = ActionRecognitionModel(
                num_classes=NUM_CLASSES,
                video_feature_dim=VIDEO_FEATURE_DIM,
                skeleton_model_dim=SKELETON_MODEL_DIM,
                skeleton_input_dim=SKELETON_INPUT_DIM,
                video_transformer_layers=4,
                skeleton_transformer_layers=4,
                n_heads=8,
                dropout=0.1,
            ).to(device)

            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"\nModel Parameters:")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

            # Create dummy input tensors with correct shapes
            print(f"\n=== Testing Forward Pass with Dummy Data ===")

            dummy_video_frames = torch.randn(
                BATCH_SIZE, NUM_FRAMES, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH
            ).to(device)

            dummy_skeleton_data = torch.randn(
                BATCH_SIZE, NUM_FRAMES, SKELETON_INPUT_DIM
            ).to(device)

            print(f"Video frames shape: {dummy_video_frames.shape}")
            print(f"Skeleton data shape: {dummy_skeleton_data.shape}")

            # Perform forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_video_frames, dummy_skeleton_data)

            print(f"Output shape: {output.shape}")
            print(f"Expected shape: ({BATCH_SIZE}, {NUM_CLASSES})")

            # Verify output shape
            assert output.shape == (
                BATCH_SIZE,
                NUM_CLASSES,
            ), f"Expected shape ({BATCH_SIZE}, {NUM_CLASSES}), got {output.shape}"
            print("✓ Output shape is correct!")

            # Test with real data
            print(f"\n=== Testing with Real Data ===")

            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_multimodal_fn,
                num_workers=0,
            )

            print(f"Dataset loaded: {len(dataset)} sequences")
            print(f"Number of classes: {dataset.num_classes}")
            print(f"Classes: {list(dataset.label_to_idx.keys())}")

            # Test one batch
            for batch in dataloader:
                video_frames = batch["video_frames"].to(device)
                skeleton_data = batch["skeleton_data"].to(device)
                labels = batch["labels"].to(device)

                print(f"\nReal data batch shapes:")
                print(f"Video frames: {video_frames.shape}")
                print(f"Skeleton data: {skeleton_data.shape}")
                print(f"Labels: {labels.shape}")
                print(f"Label values: {labels}")

                # Forward pass
                with torch.no_grad():
                    real_output = model(video_frames, skeleton_data)

                print(f"Real output shape: {real_output.shape}")
                print(f"Output logits: {real_output[0]}")

                # Calculate softmax probabilities
                probabilities = F.softmax(real_output, dim=1)
                print(f"Probabilities for first sample: {probabilities[0]}")

                break

            # Demonstrate training setup
            print(f"\n=== Demonstrating Training Setup ===")

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-4
            )

            print("Loss function: CrossEntropyLoss")
            print("Optimizer: AdamW")
            print("Learning rate: 1e-4")

            # Calculate loss for one batch
            model.train()
            try:
                loss = criterion(real_output, labels)
                print(f"Sample loss: {loss.item():.4f}")
                print("✓ Training setup successful!")
            except Exception as e:
                print(f"Error in training setup: {e}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(f"Data directory {data_dir} not found.")

    print(f"\n=== Model Architecture Summary ===")
    print("✓ VideoFeatureExtractor: CNN-based frame feature extraction")
    print("✓ VideoTransformerEncoder: Temporal modeling for video features")
    print("✓ HyperSAEncoder: Self-attention for skeleton sequences")
    print("✓ CrossAttentionFusion: Skeleton queries video features")
    print("✓ Classification Head: MLP for final predictions")

    # Full training setup
    print("\n🏋️ Starting Full Training...")

    # Create full dataset
    full_dataset = MultimodalActionDataset(
        data_dir=data_dir,
        sequence_length=16,
        image_size=(224, 224),
        train=True,
        max_sequences=None,
    )

    # Split dataset
    from torch.utils.data import random_split

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_multimodal_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_multimodal_fn
    )

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
