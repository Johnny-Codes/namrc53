import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
import random
import math
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import logging
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Configuration ---
class Config:
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34
    SEQUENCE_LENGTH = 100  # Reduced for memory efficiency
    BATCH_SIZE = 8  # Reduced for memory efficiency
    NUM_EPOCHS = 30  # Reduced for faster iteration
    LEARNING_RATE = 0.001
    PATIENCE = 8
    NUM_JOINTS = 32
    NUM_COORDS = 3
    INPUT_FEATURES = NUM_JOINTS * NUM_COORDS
    MIN_SEQUENCE_FRAMES = 10  # Minimum frames required per sequence
    MAX_SEQUENCE_FRAMES = 300  # Maximum frames to prevent memory issues

    ACTION_LABELS = [
        "using_control_panel",
        "using_flexpendant_mounted",
        "using_flexpendant_mobile",
        "inspecting_buildplate",
        "preparing_buildplate",
        "refit_buildplate",
        "grinding_buildplate",
        "toggle_lights",
        "open_door",
        "close_door",
        "turning_gas_knobs",
        "adjusting_tool",
        "wiring",
        "donning_ppe",
        "doffing_ppe",
        "observing",
        "walking",
    ]

    @property
    def label_to_idx(self):
        return {label: i for i, label in enumerate(self.ACTION_LABELS)}

    @property
    def num_classes(self):
        return len(self.ACTION_LABELS)


config = Config()


# --- Skeleton Graph Structure (for GCN) ---
class SkeletonGraph:
    EDGES = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 26),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (7, 10),
        (2, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (14, 17),
        (0, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (0, 22),
        (22, 23),
        (23, 24),
        (24, 25),
        (26, 27),
        (27, 28),
        (28, 29),
        (27, 30),
        (30, 31),
    ]
    PARENT_MAP = {
        1: 0,
        2: 1,
        3: 2,
        4: 2,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 7,
        11: 2,
        12: 11,
        13: 12,
        14: 13,
        15: 14,
        16: 15,
        17: 14,
        18: 0,
        19: 18,
        20: 19,
        21: 20,
        22: 0,
        23: 22,
        24: 23,
        25: 24,
        26: 3,
        27: 26,
        28: 27,
        29: 28,
        30: 27,
        31: 30,
    }

    @classmethod
    def get_adjacency_matrix(cls):
        A = np.zeros((config.NUM_JOINTS, config.NUM_JOINTS))
        for i, j in cls.EDGES:
            A[i, j], A[j, i] = 1, 1
        I = np.identity(config.NUM_JOINTS)
        A_hat = A + I
        D_hat_diag = np.sum(A_hat, axis=1)
        D_hat_inv_sqrt = np.power(D_hat_diag, -0.5)
        D_hat_inv_sqrt[np.isinf(D_hat_inv_sqrt)] = 0.0
        normalized_adj = np.diag(D_hat_inv_sqrt) @ A_hat @ np.diag(D_hat_inv_sqrt)
        return torch.tensor(normalized_adj, dtype=torch.float32)


ADJACENCY_MATRIX = SkeletonGraph.get_adjacency_matrix()


# --- Data Processing ---
class SkeletonProcessor:
    @staticmethod
    def normalize_skeleton(
        sequence: np.ndarray, reference_joint: int = 0
    ) -> np.ndarray:
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(-1, config.NUM_JOINTS, config.NUM_COORDS)
        reference_coords = sequence[:, reference_joint : reference_joint + 1, :]
        return sequence - reference_coords

    @staticmethod
    def compute_bone_vectors(joint_sequence: np.ndarray) -> np.ndarray:
        bone_sequence = np.zeros_like(joint_sequence)
        for child, parent in SkeletonGraph.PARENT_MAP.items():
            bone_sequence[:, child, :] = (
                joint_sequence[:, child, :] - joint_sequence[:, parent, :]
            )
        return bone_sequence


# --- MODEL ARCHITECTURES ---
class ActionRecognitionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = config.num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden states properly
        h0 = torch.zeros(
            self.num_layers * 2,
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        c0 = torch.zeros(
            self.num_layers * 2,
            batch_size,
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        x = self.linear(x)
        return torch.einsum("ij,btjf->btif", adj, x)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                ((kernel_size - 1) // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        res = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.gcn(x, adj).permute(0, 3, 1, 2)
        x = self.tcn(x).permute(0, 2, 3, 1)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = config.num_classes

        self.data_bn = nn.BatchNorm1d(config.NUM_JOINTS * config.NUM_COORDS)
        self.layers = nn.ModuleList(
            [
                STGCNBlock(config.NUM_COORDS, 64),
                STGCNBlock(64, 64),
                STGCNBlock(64, 128, stride=2),
                STGCNBlock(128, 128),
                STGCNBlock(128, 256, stride=2),
                STGCNBlock(256, 256),
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        B, T, V, C = x.shape
        x_flat = x.reshape(B, T, V * C).permute(0, 2, 1)
        x_flat = self.data_bn(x_flat).permute(0, 2, 1)
        x = x_flat.reshape(B, T, V, C)

        adj = ADJACENCY_MATRIX.to(x.device)
        for layer in self.layers:
            x = layer(x, adj)

        x = self.global_pool(x.permute(0, 3, 1, 2)).view(B, -1)
        x = self.dropout(x)
        return self.fc(x)


class TwoStreamSTGCN(nn.Module):
    def __init__(self, num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = config.num_classes
        self.joint_stream = STGCN(num_classes)
        self.bone_stream = STGCN(num_classes)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_joint, x_bone = x
        out_joint = self.joint_stream(x_joint)
        out_bone = self.bone_stream(x_bone)
        alpha = torch.sigmoid(self.fusion_weight)
        return alpha * out_joint + (1 - alpha) * out_bone


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ActionRecognitionTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        model_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_classes: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = config.num_classes

        self.input_fc = nn.Sequential(
            nn.Linear(input_features, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer_encoder(x)
        return self.classifier(x[:, 0, :])


# --- Dataset Classes ---
class BaseSkeletonDataset(Dataset):
    def __init__(self, sequences_data: List, labels_data: List):
        self.sequences = sequences_data
        self.labels = labels_data
        self.processor = SkeletonProcessor()

    def __len__(self):
        return len(self.sequences)

    def _pad(self, seq, length):
        if len(seq) >= length:
            return seq[:length]
        pad = np.zeros((length - len(seq), *seq.shape[1:]), dtype=np.float32)
        return np.vstack([seq, pad])


class LSTMTransformerDataset(BaseSkeletonDataset):
    def __getitem__(self, idx):
        sequence_np = np.array(self.sequences[idx], dtype=np.float32)
        normalized = self.processor.normalize_skeleton(sequence_np).reshape(
            -1, config.INPUT_FEATURES
        )
        padded = self._pad(normalized, config.SEQUENCE_LENGTH)
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class GCNDataset(BaseSkeletonDataset):
    def __getitem__(self, idx):
        sequence_np = np.array(self.sequences[idx], dtype=np.float32)
        joint_sequence = sequence_np.reshape(-1, config.NUM_JOINTS, config.NUM_COORDS)
        normalized_joints = self.processor.normalize_skeleton(joint_sequence)
        bone_sequence = self.processor.compute_bone_vectors(joint_sequence)
        padded_joints = self._pad(normalized_joints, config.SEQUENCE_LENGTH)
        padded_bones = self._pad(bone_sequence, config.SEQUENCE_LENGTH)
        return (
            torch.tensor(padded_joints, dtype=torch.float32),
            torch.tensor(padded_bones, dtype=torch.float32),
        ), torch.tensor(self.labels[idx], dtype=torch.long)


# --- Training & Evaluation ---
class ModelTrainer:
    def __init__(self, model_name: str, model: nn.Module, device: torch.device):
        self.model_name = model_name
        self.model = model
        self.device = device
        self.best_accuracy = 0.0
        self.patience_counter = 0

    def train_and_evaluate(
        self, train_loader, val_loader, test_loader, class_weights, view_name=""
    ):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.AdamW(
            self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS
        )

        model_save_name = (
            f"best_{self.model_name.lower()}_{view_name.lower()}_model.pth"
        )
        logger.info(f"Training {self.model_name} model for {view_name} view...")

        # Training metrics tracking
        train_losses = []
        val_accuracies = []

        for epoch in range(config.NUM_EPOCHS):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Progress tracking
            if epoch % 10 == 0:
                logger.info(f"Starting epoch {epoch+1}/{config.NUM_EPOCHS}")

            for batch_idx, (data, labels) in enumerate(train_loader):
                try:
                    if self.model_name == "GCN":
                        data = [d.to(self.device, non_blocking=True) for d in data]
                    else:
                        data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(
                            f"NaN loss detected at epoch {epoch+1}, batch {batch_idx}"
                        )
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    # Clear cache periodically for memory management
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(
                            f"GPU out of memory at epoch {epoch+1}, batch {batch_idx}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            if num_batches == 0:
                logger.error(f"No valid batches in epoch {epoch+1}")
                break

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # Validation phase
            val_accuracy, _, _ = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)

            # Logging every 5 epochs or if significant improvement
            if epoch % 5 == 0 or val_accuracy > self.best_accuracy:
                logger.info(
                    f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
                )

            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), model_save_name)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            scheduler.step()

            # Memory cleanup
            torch.cuda.empty_cache()

        # Load best model and evaluate
        if os.path.exists(model_save_name):
            self.model.load_state_dict(torch.load(model_save_name))
            logger.info(
                f"Loaded best model with validation accuracy: {self.best_accuracy:.2f}%"
            )

        _, test_preds, test_labels = self.evaluate(test_loader)

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(test_labels, test_preds) * 100,
            "Precision": precision_score(
                test_labels, test_preds, average="weighted", zero_division=0
            )
            * 100,
            "Recall": recall_score(
                test_labels, test_preds, average="weighted", zero_division=0
            )
            * 100,
            "F1-Score": f1_score(
                test_labels, test_preds, average="weighted", zero_division=0
            )
            * 100,
        }

        # Get unique labels for classification report
        unique_labels = sorted(list(set(test_labels)))
        present_action_labels = [config.ACTION_LABELS[i] for i in unique_labels]

        # Log classification report
        logger.info(f"\n{self.model_name} ({view_name}) Classification Report:")
        try:
            report = classification_report(
                test_labels,
                test_preds,
                labels=unique_labels,
                target_names=present_action_labels,
                zero_division=0,
            )
            logger.info(f"\n{report}")
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")

        return metrics

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data, labels in loader:
                if self.model_name == "GCN":
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds) * 100
        return accuracy, all_preds, all_labels


# --- Data Loading ---
class DataManager:
    @staticmethod
    def load_action_sequences(data_prefix: str, num_folders: int) -> Dict:
        logger.info("Loading action sequences...")
        action_sequences = defaultdict(list)

        for i in range(num_folders):
            folder_name = f"train-{i:03d}"
            metadata_path = os.path.join(data_prefix, folder_name, "metadata.jsonl")
            if not os.path.exists(metadata_path):
                continue

            with open(metadata_path, "r") as f:
                for line in f:
                    try:
                        frame = json.loads(line.strip())
                        if frame.get("action_number") is not None:
                            action_sequences[frame["action_number"]].append(frame)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {metadata_path}: {e}")
                        continue

        logger.info(f"Loaded {len(action_sequences)} total action sequences")
        return action_sequences

    @staticmethod
    def separate_views(action_sequences: Dict) -> Tuple[Dict, Dict]:
        inner = {
            sid: f
            for sid, f in action_sequences.items()
            if f and "inner" in f[0].get("file_name", "").lower()
        }
        outer = {
            sid: f
            for sid, f in action_sequences.items()
            if f and "outer" in f[0].get("file_name", "").lower()
        }
        logger.info(f"Inner sequences: {len(inner)}, Outer sequences: {len(outer)}")
        return inner, outer

    @staticmethod
    def prepare_split_data(
        sequences_dict: Dict, sequence_ids: List[int]
    ) -> Tuple[List, List]:
        data, labels = [], []
        skipped_sequences = 0
        action_counts = Counter()

        for seq_id in sequence_ids:
            seq_id = int(seq_id) if isinstance(seq_id, str) else seq_id
            frames = sequences_dict.get(seq_id)
            if not frames:
                skipped_sequences += 1
                continue

            frames.sort(key=lambda x: x["frame"])
            label = frames[0].get("action_label")

            if label not in config.label_to_idx:
                logger.warning(f"Unknown action label '{label}' in sequence {seq_id}")
                skipped_sequences += 1
                continue

            # Extract skeleton data with better validation
            skeletons = []
            valid_frames = 0

            for f in frames:
                try:
                    skeleton = f.get("skeleton", [])
                    if not skeleton:
                        continue

                    skeleton_array = np.array(skeleton, dtype=np.float32)

                    # Check if skeleton has correct dimensions
                    if skeleton_array.size != config.INPUT_FEATURES:
                        logger.warning(
                            f"Skeleton size mismatch in sequence {seq_id}: expected {config.INPUT_FEATURES}, got {skeleton_array.size}"
                        )
                        continue

                    # Check for invalid values
                    if np.any(np.isnan(skeleton_array)) or np.any(
                        np.isinf(skeleton_array)
                    ):
                        continue

                    skeletons.append(skeleton_array.flatten())
                    valid_frames += 1

                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid skeleton data in sequence {seq_id}: {e}")
                    continue

            # Only include sequences with sufficient valid frames
            if valid_frames >= 10:  # Minimum frame threshold
                data.append(skeletons)
                labels.append(config.label_to_idx[label])
                action_counts[label] += 1
            else:
                skipped_sequences += 1
                logger.warning(
                    f"Sequence {seq_id} has too few valid frames ({valid_frames})"
                )

        logger.info(f"Action distribution: {dict(action_counts)}")
        if skipped_sequences > 0:
            logger.info(
                f"Skipped {skipped_sequences} sequences due to missing/invalid data"
            )

        return data, labels

    @staticmethod
    def load_split_config(view_name: str) -> Optional[Dict]:
        """Load split configuration for a specific view"""
        config_file = f"split_config_{view_name.lower()}.json"
        try:
            with open(config_file, "r") as f:
                split_config = json.load(f)
            logger.info(f"Loaded split config from {config_file}")
            return split_config
        except FileNotFoundError:
            logger.error(f"Split config file {config_file} not found.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing {config_file}: {e}")
            return None


# --- Main Execution ---
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    data_manager = DataManager()
    action_sequences = data_manager.load_action_sequences(
        config.DATA_PREFIX, config.NUM_FOLDERS
    )
    inner_sequences, outer_sequences = data_manager.separate_views(action_sequences)

    results = []

    # Process each view with its own split configuration
    views_data = [("Inner", inner_sequences), ("Outer", outer_sequences)]

    for view_name, sequences_dict in views_data:
        if not sequences_dict:
            logger.warning(f"No sequences for {view_name} view.")
            continue

        # Load view-specific split configuration
        split_config = data_manager.load_split_config(view_name)
        if split_config is None:
            logger.error(f"Cannot proceed with {view_name} view without split config.")
            continue

        logger.info(f"\nProcessing {view_name} view...")

        # Get sequence IDs for each split
        view_train_ids = [
            sid
            for sid in split_config.get("train_sequences", [])
            if sid in sequences_dict
        ]
        view_val_ids = [
            sid
            for sid in split_config.get("validation_sequences", [])
            if sid in sequences_dict
        ]
        view_test_ids = [
            sid
            for sid in split_config.get("test_sequences", [])
            if sid in sequences_dict
        ]

        # Prepare data for each split
        train_data, train_labels = data_manager.prepare_split_data(
            sequences_dict, view_train_ids
        )
        val_data, val_labels = data_manager.prepare_split_data(
            sequences_dict, view_val_ids
        )
        test_data, test_labels = data_manager.prepare_split_data(
            sequences_dict, view_test_ids
        )

        if not train_data:
            logger.warning(f"No training data for {view_name} view after filtering.")
            continue

        logger.info(
            f"{view_name} view - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )

        # Calculate class weights for balanced training
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = torch.FloatTensor(
            [
                total_samples / max(class_counts.get(i, 1), 1)
                for i in range(config.num_classes)
            ]
        )

        # Log class distribution
        logger.info(f"Class distribution: {dict(class_counts)}")

        # Log detailed dataset statistics
        def log_dataset_stats(data, labels, split_name, view_name):
            """Log detailed dataset statistics"""
            logger.info(f"\n{view_name} {split_name} Dataset Statistics:")
            logger.info(f"  Total sequences: {len(data)}")
            if data:
                sequence_lengths = [len(seq) for seq in data]
                logger.info(f"  Avg sequence length: {np.mean(sequence_lengths):.1f}")
                logger.info(
                    f"  Min/Max sequence length: {min(sequence_lengths)}/{max(sequence_lengths)}"
                )

            label_counts = Counter(labels)
            logger.info(f"  Class distribution: {dict(sorted(label_counts.items()))}")

        log_dataset_stats(train_data, train_labels, "Train", view_name)
        log_dataset_stats(val_data, val_labels, "Validation", view_name)
        log_dataset_stats(test_data, test_labels, "Test", view_name)

        # Create datasets
        datasets = {
            "lstm_train": LSTMTransformerDataset(train_data, train_labels),
            "lstm_val": LSTMTransformerDataset(val_data, val_labels),
            "lstm_test": LSTMTransformerDataset(test_data, test_labels),
            "gcn_train": GCNDataset(train_data, train_labels),
            "gcn_val": GCNDataset(val_data, val_labels),
            "gcn_test": GCNDataset(test_data, test_labels),
        }

        # Create data loaders
        dataloaders = {
            name: DataLoader(
                ds,
                batch_size=config.BATCH_SIZE,
                shuffle="train" in name,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            for name, ds in datasets.items()
        }

        # Initialize models
        models = {
            "LSTM": ActionRecognitionLSTM(config.INPUT_FEATURES).to(device),
            "GCN": TwoStreamSTGCN().to(device),
            "Transformer": ActionRecognitionTransformer(config.INPUT_FEATURES).to(
                device
            ),
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_name} model for {view_name} view")
            logger.info(f"{'='*60}")

            # Select appropriate data loaders
            train_dl_key = "gcn_train" if model_name == "GCN" else "lstm_train"
            val_dl_key = "gcn_val" if model_name == "GCN" else "lstm_val"
            test_dl_key = "gcn_test" if model_name == "GCN" else "lstm_test"

            trainer = ModelTrainer(model_name, model, device)

            try:
                metrics = trainer.train_and_evaluate(
                    dataloaders[train_dl_key],
                    dataloaders[val_dl_key],
                    dataloaders[test_dl_key],
                    class_weights,
                    view_name,
                )

                metrics["Model"] = model_name
                metrics["View"] = view_name
                results.append(metrics)

                logger.info(
                    f"{model_name} Results - Accuracy: {metrics['Accuracy']:.2f}%, "
                    f"F1: {metrics['F1-Score']:.2f}%"
                )

            except Exception as e:
                logger.error(
                    f"Error training {model_name} for {view_name} view: {e}",
                    exc_info=True,
                )
                continue

    # Generate and save final report
    if results:
        logger.info(f"\n{'='*80}")
        logger.info(f"{'FINAL MODEL COMPARISON REPORT':^80}")
        logger.info(f"{'='*80}")

        report_df = pd.DataFrame(results)
        report_df = report_df[
            ["Model", "View", "Accuracy", "F1-Score", "Precision", "Recall"]
        ].round(2)

        print("\n" + report_df.to_string(index=False))

        # Save report
        report_df.to_csv("model_comparison_report.csv", index=False)
        logger.info(f"\n{'='*80}")
        logger.info("Report saved to model_comparison_report.csv")

        # Find and report best performing model
        best_model = report_df.loc[report_df["Accuracy"].idxmax()]
        logger.info(
            f"Best performing model: {best_model['Model']} ({best_model['View']} view) "
            f"- Accuracy: {best_model['Accuracy']:.2f}%"
        )

        # Additional analysis
        logger.info("\nModel Performance Summary:")
        for view in report_df["View"].unique():
            view_results = report_df[report_df["View"] == view]
            logger.info(f"\n{view} View:")
            for _, row in view_results.iterrows():
                logger.info(
                    f"  {row['Model']}: {row['Accuracy']:.2f}% accuracy, {row['F1-Score']:.2f}% F1"
                )
    else:
        logger.error("No models were successfully trained!")


if __name__ == "__main__":
    main()
