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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pandas as pd
import time
import random
import os

# Import our utility functions
from utils.seeds import set_seeds
from utils.data_utils import collate_fn, StratifiedVideoDatasetSplitter
from utils.loss_utils import WeightedLossCalculator, FocalLoss, LabelSmoothingLoss
from utils.metrics import calculate_comprehensive_metrics, calculate_per_class_metrics
from utils.visualization import create_class_weights_plots, create_training_curves
from utils.results_saver import ResultsSaver

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(
        f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )


# Additional advanced loss functions
class CBLoss(nn.Module):
    """Class-Balanced Loss Based on Effective Number of Samples"""

    def __init__(
        self, samples_per_cls, num_classes, loss_type="focal", beta=0.9999, gamma=2.0
    ):
        super(CBLoss, self).__init__()

        # Ensure samples_per_cls has the right length and no zeros
        if len(samples_per_cls) != num_classes:
            print(
                f"Warning: samples_per_cls length {len(samples_per_cls)} != num_classes {num_classes}"
            )
            samples_per_cls = samples_per_cls[:num_classes] + [1] * (
                num_classes - len(samples_per_cls)
            )

        # Avoid zeros in samples_per_cls
        samples_per_cls = [max(1, count) for count in samples_per_cls]

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes

        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.gamma = gamma

        print(f"CB Loss initialized with weights: {self.weights}")

    def forward(self, logits, labels):
        try:
            if self.loss_type == "focal":
                # Use standard cross entropy with class weights for simplicity
                ce_loss = F.cross_entropy(
                    logits, labels, weight=self.weights, reduction="none"
                )
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
            else:
                # Fallback to weighted cross entropy
                return F.cross_entropy(logits, labels, weight=self.weights)
        except Exception as e:
            print(f"Error in CB Loss forward: {e}")
            # Fallback to standard cross entropy
            return F.cross_entropy(logits, labels)


class LDAM_Loss(nn.Module):
    """Label-Distribution-Aware Margin Loss"""

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAM_Loss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class EQLv2Loss(nn.Module):
    """Equalization Loss v2"""

    def __init__(self, num_classes, gamma=12, mu=0.8, alpha=4.0):
        super(EQLv2Loss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        self.register_buffer("pos_grad", torch.zeros(num_classes))
        self.register_buffer("neg_grad", torch.zeros(num_classes))
        # protection for a class without positive sample
        self.register_buffer("pos_neg", torch.ones(num_classes))

    def forward(self, logits, labels):
        self.targets = labels
        self.predictions = torch.sigmoid(logits)

        grad = self.collect_grad()

        pos_w, neg_w = self.get_weight(grad)

        weight = pos_w * labels + neg_w * (1 - labels)

        cls_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        cls_loss = torch.sum(cls_loss * weight) / self.num_classes

        return cls_loss

    def collect_grad(self):
        prob = torch.sigmoid(self.predictions)
        grad = torch.abs(prob - self.targets.float())
        return grad

    def get_weight(self, grad):
        pos_grad = torch.sum(grad * self.targets.float(), dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets.float()), dim=0)

        self.pos_grad += self.mu * (pos_grad - self.pos_grad)
        self.neg_grad += self.mu * (neg_grad - self.neg_grad)

        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

        pos_w = 1 + self.alpha * torch.tanh(self.gamma * self.pos_neg)
        neg_w = 1 + self.alpha * torch.tanh(self.gamma / (self.pos_neg + 1e-10))

        return pos_w, neg_w


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


class MultiLossEfficientNetCNNGNN(nn.Module):
    """Multi-loss CNN-GNN model using EfficientNet-B0"""

    def __init__(
        self,
        num_classes,
        cnn_feature_dim=256,
        gnn_hidden_dim=128,
        gnn_layers=2,
        num_heads=4,
        dropout=0.2,
    ):
        super(MultiLossEfficientNetCNNGNN, self).__init__()

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


class MultiLossTrainer:
    """Trainer with multiple state-of-the-art loss functions"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_names,
        class_counts,
        loss_config,
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
        self.class_counts = class_counts
        self.loss_config = loss_config
        self.results_saver = results_saver

        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=30,
        )

        # Initialize loss function based on config
        self.criterion = self._get_loss_function(loss_config, class_counts, num_classes)
        self.scaler = torch.cuda.amp.GradScaler()

        print(f"🎯 Using {loss_config['name']} for class imbalance handling")
        print("Using mixed precision training for speed optimization")
        print(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
        )

    def _get_loss_function(self, loss_config, class_counts, num_classes):
        """Get loss function based on configuration"""
        loss_name = loss_config["name"]

        try:
            if loss_name == "weighted_ce":
                # Standard weighted CrossEntropy
                weight_calculator = WeightedLossCalculator(
                    weighting_strategy=loss_config.get(
                        "weighting_strategy", "inverse_sqrt"
                    ),
                    rare_class_boost=loss_config.get("rare_class_boost", 3.0),
                )
                class_weights = weight_calculator.calculate_class_weights(
                    class_counts, num_classes, device
                )
                return nn.CrossEntropyLoss(weight=class_weights)

            elif loss_name == "focal":
                # Focal Loss
                weight_calculator = WeightedLossCalculator(
                    weighting_strategy=loss_config.get(
                        "weighting_strategy", "inverse_sqrt"
                    ),
                    rare_class_boost=loss_config.get("rare_class_boost", 2.0),
                )
                class_weights = weight_calculator.calculate_class_weights(
                    class_counts, num_classes, device
                )
                return FocalLoss(
                    alpha=loss_config.get("alpha", 1.0),
                    gamma=loss_config.get("gamma", 2.0),
                    weight=class_weights,
                )

            elif loss_name == "label_smoothing":
                # Label Smoothing with weights
                weight_calculator = WeightedLossCalculator(
                    weighting_strategy=loss_config.get(
                        "weighting_strategy", "inverse_sqrt"
                    ),
                    rare_class_boost=loss_config.get("rare_class_boost", 2.0),
                )
                class_weights = weight_calculator.calculate_class_weights(
                    class_counts, num_classes, device
                )
                return LabelSmoothingLoss(
                    num_classes=num_classes,
                    smoothing=loss_config.get("smoothing", 0.1),
                    weight=class_weights,
                )

            elif loss_name == "cb_focal":
                # Class-Balanced Focal Loss
                samples_per_cls = [class_counts.get(i, 1) for i in range(num_classes)]
                print(f"CB Loss samples per class: {samples_per_cls}")
                return CBLoss(
                    samples_per_cls=samples_per_cls,
                    num_classes=num_classes,
                    loss_type="focal",
                    beta=loss_config.get("beta", 0.9999),
                    gamma=loss_config.get("gamma", 2.0),
                )

            elif loss_name == "ldam":
                # Label-Distribution-Aware Margin Loss
                cls_num_list = [class_counts.get(i, 1) for i in range(num_classes)]
                weight_calculator = WeightedLossCalculator(
                    weighting_strategy="inverse_sqrt", rare_class_boost=2.0
                )
                class_weights = weight_calculator.calculate_class_weights(
                    class_counts, num_classes, device
                )
                return LDAM_Loss(
                    cls_num_list=cls_num_list,
                    max_m=loss_config.get("max_m", 0.5),
                    weight=class_weights,
                    s=loss_config.get("s", 30),
                )

            else:
                print(
                    f"Unknown loss function: {loss_name}, falling back to weighted CrossEntropy"
                )
                weight_calculator = WeightedLossCalculator()
                class_weights = weight_calculator.calculate_class_weights(
                    class_counts, num_classes, device
                )
                return nn.CrossEntropyLoss(weight=class_weights)

        except Exception as e:
            print(f"Error creating {loss_name} loss function: {e}")
            print("Falling back to standard CrossEntropy")
            return nn.CrossEntropyLoss()

    def train_epoch(self):
        """Training epoch with specified loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Training ({self.loss_config['name']})")

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
                try:
                    frames = batch["frames"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    edge_indices = [
                        ei.to(device, non_blocking=True) for ei in batch["edge_indices"]
                    ]
                    batch_indices = [
                        bi.to(device, non_blocking=True)
                        for bi in batch["batch_indices"]
                    ]

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
                    print(f"Error in {split_name} evaluation batch: {e}")
                    continue

        avg_loss = total_loss / max(num_batches, 1)

        # Always calculate metrics, even if empty
        if all_labels and all_predictions:
            metrics = calculate_comprehensive_metrics(
                all_labels,
                all_predictions,
                np.array(all_probabilities) if all_probabilities else None,
                avg_loss,
            )
        else:
            print(f"Warning: No valid predictions for {split_name} evaluation")
            # Return complete default metrics structure
            metrics = {
                "loss": avg_loss,
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

        return metrics, all_predictions, all_labels, all_probabilities

    def train(self, num_epochs=30):
        """Training loop with specified loss"""
        best_f1 = 0

        print(
            f"Starting {self.loss_config['name']} training for {num_epochs} epochs..."
        )
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
            print(f"Train Loss ({self.loss_config['name']}): {train_loss:.4f}")
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
                        / f"best_{self.loss_config['name']}_model.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss_config": self.loss_config,
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
                        f"✓ Best {self.loss_config['name']} model saved (Val F1: {best_f1:.4f}, Test F1: {test_metrics['f1_macro']:.4f})"
                    )

                    # Save results for best model
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
        print(
            f"\n{self.loss_config['name']} training completed in {total_time:.1f} seconds"
        )
        print(f"Best Validation F1 Score: {best_f1:.4f}")

        return best_f1, val_metrics, test_metrics


def run_multi_loss_experiments():
    """Main function to run multiple loss function experiments"""

    print("🎯 MULTI-LOSS EFFICIENTNET-B0 CNN-GNN Video Classification Experiments")
    print("=" * 90)

    # Set seeds for reproducibility
    set_seeds(42)

    # Configuration
    base_config = {
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

    # Define multiple loss function configurations
    loss_configs = [
        {
            "name": "weighted_ce",
            "description": "Weighted CrossEntropy with Inverse Square Root",
            "weighting_strategy": "inverse_sqrt",
            "rare_class_boost": 3.0,
        },
        {
            "name": "focal",
            "description": "Focal Loss with Class Weights",
            "weighting_strategy": "inverse_sqrt",
            "rare_class_boost": 2.0,
            "alpha": 1.0,
            "gamma": 2.0,
        },
        {
            "name": "label_smoothing",
            "description": "Label Smoothing with Class Weights",
            "weighting_strategy": "inverse_sqrt",
            "rare_class_boost": 2.0,
            "smoothing": 0.1,
        },
        {
            "name": "cb_focal",
            "description": "Class-Balanced Focal Loss",
            "beta": 0.9999,
            "gamma": 2.0,
        },
        {
            "name": "ldam",
            "description": "Label-Distribution-Aware Margin Loss",
            "max_m": 0.5,
            "s": 30,
        },
    ]

    print(f"Base Configuration: {base_config}")
    print(f"\n🧪 Running {len(loss_configs)} different loss function experiments:")
    for i, config in enumerate(loss_configs, 1):
        print(f"  {i}. {config['name']}: {config['description']}")

    experiment_results = {}

    try:
        # Create dataset (only once)
        print("\n📂 Creating dataset...")
        dataset = StratifiedVideoDataset(
            data_dir=base_config["data_dir"],
            sequence_length=base_config["sequence_length"],
            image_size=base_config["image_size"],
            temporal_window=base_config["temporal_window"],
            train=True,
            max_sequences=base_config["max_sequences"],
            cache_frames=True,
        )

        if len(dataset) == 0:
            print("❌ No sequences found! Check your data directory.")
            return

        # Create stratified splits (only once)
        print("\n🎯 Creating stratified splits...")
        splitter = StratifiedVideoDatasetSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            min_samples_per_split=1,
            seed=base_config["seed"],
        )

        train_indices, val_indices, test_indices, class_splits, class_counts = (
            splitter.create_stratified_split(dataset)
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
            batch_size=base_config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=base_config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=base_config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # Run experiments for each loss function
        for exp_idx, loss_config in enumerate(loss_configs, 1):
            print(f"\n{'='*90}")
            print(
                f"🧪 EXPERIMENT {exp_idx}/{len(loss_configs)}: {loss_config['name'].upper()}"
            )
            print(f"📝 {loss_config['description']}")
            print(f"{'='*90}")

            # Create experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"efficientnet_b0_{loss_config['name']}_seq{base_config['sequence_length']}_bs{base_config['batch_size']}_lr{base_config['learning_rate']}_{timestamp}"
            results_saver = ResultsSaver(
                save_dir="results", experiment_name=experiment_name
            )

            # Create fresh model for each experiment
            model = MultiLossEfficientNetCNNGNN(
                num_classes=dataset.num_classes,
                cnn_feature_dim=base_config["cnn_feature_dim"],
                gnn_hidden_dim=base_config["gnn_hidden_dim"],
                gnn_layers=base_config["gnn_layers"],
                num_heads=base_config["num_heads"],
            )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # Save experiment configuration
            combined_config = {**base_config, "loss_config": loss_config}
            model_params = {
                "backbone_name": "efficientnet_b0",
                "backbone_description": f"EfficientNet-B0 with {loss_config['name']} Loss",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "num_classes": dataset.num_classes,
                "class_names": list(dataset.label_to_idx.keys()),
            }

            results_saver.save_experiment_config(
                combined_config, model_params, loss_config
            )

            # Create trainer with specific loss function
            trainer = MultiLossTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_classes=dataset.num_classes,
                class_names=list(dataset.label_to_idx.keys()),
                class_counts=class_counts,
                loss_config=loss_config,
                learning_rate=base_config["learning_rate"],
                results_saver=results_saver,
            )

            # Train model
            start_time = time.time()
            best_f1, final_val_metrics, final_test_metrics = trainer.train(
                num_epochs=base_config["num_epochs"]
            )
            training_time = time.time() - start_time

            # Store results
            experiment_results[loss_config["name"]] = {
                "best_val_f1": best_f1,
                "final_test_f1": final_test_metrics["f1_macro"],
                "final_test_accuracy": final_test_metrics["accuracy"],
                "training_time": training_time,
                "experiment_dir": results_saver.experiment_dir,
                "loss_config": loss_config,
            }

            print(f"\n🎉 {loss_config['name'].upper()} experiment completed!")
            print(f"Best Validation F1: {best_f1:.4f}")
            print(f"Final Test F1: {final_test_metrics['f1_macro']:.4f}")
            print(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
            print(f"Training Time: {training_time:.1f} seconds")
            print(f"Results saved to: {results_saver.experiment_dir}")

        # Create comprehensive comparison report
        print(f"\n{'='*90}")
        print("📊 COMPREHENSIVE EXPERIMENT COMPARISON")
        print(f"{'='*90}")

        # Sort by best validation F1
        sorted_results = sorted(
            experiment_results.items(), key=lambda x: x[1]["best_val_f1"], reverse=True
        )

        print(
            f"\n{'Rank':<5} {'Loss Function':<20} {'Val F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Time (s)':<10}"
        )
        print("-" * 75)

        for rank, (loss_name, results) in enumerate(sorted_results, 1):
            print(
                f"{rank:<5} {loss_name:<20} {results['best_val_f1']:<10.4f} "
                f"{results['final_test_f1']:<10.4f} {results['final_test_accuracy']:<10.4f} "
                f"{results['training_time']:<10.1f}"
            )

        # Save comparison results
        comparison_df = pd.DataFrame(
            {
                "loss_function": [name for name, _ in sorted_results],
                "best_val_f1": [
                    results["best_val_f1"] for _, results in sorted_results
                ],
                "final_test_f1": [
                    results["final_test_f1"] for _, results in sorted_results
                ],
                "final_test_accuracy": [
                    results["final_test_accuracy"] for _, results in sorted_results
                ],
                "training_time_seconds": [
                    results["training_time"] for _, results in sorted_results
                ],
            }
        )

        comparison_path = Path("results") / "multi_loss_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n📄 Comparison results saved to: {comparison_path}")

        # Create comparison plots
        create_comparison_plots(experiment_results)

        print(f"\n🏆 BEST PERFORMING LOSS FUNCTION: {sorted_results[0][0].upper()}")
        print(f"   📈 Validation F1: {sorted_results[0][1]['best_val_f1']:.4f}")
        print(f"   🎯 Test F1: {sorted_results[0][1]['final_test_f1']:.4f}")
        print(f"   ⚡ Training Time: {sorted_results[0][1]['training_time']:.1f}s")

    except KeyboardInterrupt:
        print("\n⚠️  Experiments interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error in experiments: {e}")
        import traceback

        traceback.print_exc()


def create_comparison_plots(experiment_results):
    """Create comparison plots for all experiments"""
    plots_dir = Path("results") / "comparison_plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Extract data for plotting
    loss_names = list(experiment_results.keys())
    val_f1_scores = [experiment_results[name]["best_val_f1"] for name in loss_names]
    test_f1_scores = [experiment_results[name]["final_test_f1"] for name in loss_names]
    test_accuracies = [
        experiment_results[name]["final_test_accuracy"] for name in loss_names
    ]
    training_times = [experiment_results[name]["training_time"] for name in loss_names]

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # F1 Score comparison
    x_pos = np.arange(len(loss_names))

    axes[0, 0].bar(
        x_pos - 0.2,
        val_f1_scores,
        0.4,
        label="Validation F1",
        color="skyblue",
        alpha=0.8,
    )
    axes[0, 0].bar(
        x_pos + 0.2, test_f1_scores, 0.4, label="Test F1", color="lightcoral", alpha=0.8
    )
    axes[0, 0].set_xlabel("Loss Functions")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score Comparison Across Loss Functions")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(loss_names, rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test Accuracy comparison
    axes[0, 1].bar(x_pos, test_accuracies, color="lightgreen", alpha=0.8)
    axes[0, 1].set_xlabel("Loss Functions")
    axes[0, 1].set_ylabel("Test Accuracy")
    axes[0, 1].set_title("Test Accuracy Comparison")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(loss_names, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3)

    # Training Time comparison
    axes[1, 0].bar(x_pos, training_times, color="orange", alpha=0.8)
    axes[1, 0].set_xlabel("Loss Functions")
    axes[1, 0].set_ylabel("Training Time (seconds)")
    axes[1, 0].set_title("Training Time Comparison")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(loss_names, rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3)

    # Performance vs Time scatter plot
    axes[1, 1].scatter(
        training_times,
        test_f1_scores,
        s=100,
        alpha=0.7,
        c=range(len(loss_names)),
        cmap="viridis",
    )
    for i, name in enumerate(loss_names):
        axes[1, 1].annotate(
            name,
            (training_times[i], test_f1_scores[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1, 1].set_xlabel("Training Time (seconds)")
    axes[1, 1].set_ylabel("Test F1 Score")
    axes[1, 1].set_title("Performance vs Training Time")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        plots_dir / "loss_functions_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(plots_dir / "loss_functions_comparison.pdf", bbox_inches="tight")
    plt.close()

    print(f"✅ Comparison plots saved to {plots_dir}")


if __name__ == "__main__":
    run_multi_loss_experiments()
