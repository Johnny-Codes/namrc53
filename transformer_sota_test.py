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
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time


# --- Configuration ---
class Config:
    DATA_PREFIX = "./data"
    NUM_FOLDERS = 34
    SEQUENCE_LENGTH = 150
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    MODEL_DIM = 128
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    DROPOUT = 0.1
    SCHEDULER_STEP_SIZE = 15
    SCHEDULER_GAMMA = 0.1
    PATIENCE = 10

    # Model save paths
    MODEL_SAVE_PATH_INNER = "best_inner_camera_transformer_model.pth"
    MODEL_SAVE_PATH_OUTER = "best_outer_camera_transformer_model.pth"

    # Results save paths
    RESULTS_CSV = "transformer_detailed_results.csv"
    HYPERPARAMS_CSV = "transformer_hyperparameters.csv"
    CONFUSION_MATRIX_DIR = "confusion_matrices"


config = Config()

# --- Action Labels ---
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
label_to_idx = {label: i for i, label in enumerate(ACTION_LABELS)}
NUM_CLASSES = len(ACTION_LABELS)
INPUT_FEATURES = 32 * 3  # 96 features per frame


# --- Enhanced Metrics Tracker ---
class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_times = []
        self.best_epoch = 0
        self.total_training_time = 0

    def update(self, train_loss, val_loss, val_accuracy, epoch_time):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        self.epoch_times.append(epoch_time)

    def get_best_metrics(self):
        if not self.val_accuracies:
            return {}

        best_idx = np.argmax(self.val_accuracies)
        return {
            "best_epoch": best_idx + 1,
            "best_val_accuracy": self.val_accuracies[best_idx],
            "best_val_loss": self.val_losses[best_idx],
            "corresponding_train_loss": self.train_losses[best_idx],
            "avg_epoch_time": np.mean(self.epoch_times),
            "total_training_time": self.total_training_time,
        }


# --- 1. Enhanced Transformer Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        input_features,
        model_dim,
        nhead,
        num_encoder_layers,
        num_classes,
        dropout=0.1,
    ):
        super(ActionRecognitionTransformer, self).__init__()
        self.model_dim = model_dim

        # Input projection with layer norm
        self.input_fc = nn.Sequential(
            nn.Linear(input_features, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x


# --- 2. Enhanced Dataset ---
class SkeletonActionDataset(Dataset):
    def __init__(self, sequences_data, labels_data, augment=False):
        self.sequences = sequences_data
        self.labels = labels_data
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        sequence_np = np.array(sequence, dtype=np.float32)

        # Improved normalization
        sequence_reshaped = sequence_np.reshape(-1, 32, 3)
        pelvis_coords = sequence_reshaped[:, 0:1, :]  # Use pelvis (joint 0)
        normalized_sequence = sequence_reshaped - pelvis_coords
        normalized_sequence = normalized_sequence.reshape(-1, INPUT_FEATURES)

        # Data augmentation for training
        if self.augment and random.random() < 0.3:
            noise = np.random.normal(0, 0.01, normalized_sequence.shape)
            normalized_sequence += noise

        # Padding/truncation
        padded_sequence = np.zeros(
            (config.SEQUENCE_LENGTH, INPUT_FEATURES), dtype=np.float32
        )
        seq_len = min(len(normalized_sequence), config.SEQUENCE_LENGTH)
        padded_sequence[:seq_len] = normalized_sequence[:seq_len]

        return torch.tensor(padded_sequence, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


# --- 3. Enhanced Training and Evaluation ---
def compute_metrics(y_true, y_pred, class_names=None):
    """Compute comprehensive metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    if class_names:
        unique_labels = sorted(list(set(y_true)))
        present_classes = [class_names[i] for i in unique_labels]

        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0, labels=unique_labels
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=unique_labels
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=unique_labels
        )

        for i, class_name in enumerate(present_classes):
            metrics[f"precision_{class_name}"] = precision_per_class[i]
            metrics[f"recall_{class_name}"] = recall_per_class[i]
            metrics[f"f1_{class_name}"] = f1_per_class[i]

    return metrics


def plot_confusion_matrix(
    y_true, y_pred, class_names, save_path, title="Confusion Matrix"
):
    """Plot and save confusion matrix"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get unique labels and corresponding class names
    unique_labels = sorted(list(set(y_true)))
    present_classes = [class_names[i] for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_classes,
        yticklabels=present_classes,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model and return loss, predictions, and labels"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, all_preds, all_labels


def train_and_evaluate_model(
    train_loader, val_loader, test_loader, model_save_path, view_name, class_weights
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = ActionRecognitionTransformer(
        INPUT_FEATURES,
        config.MODEL_DIM,
        config.NHEAD,
        config.NUM_ENCODER_LAYERS,
        NUM_CLASSES,
        config.DROPOUT,
    ).to(device)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA
    )

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()

    print(f"\n{'='*60}")
    print(f"Training Transformer Model for {view_name} View")
    print(f"{'='*60}")
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    start_time = time.time()
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_loss, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion, device
        )
        val_accuracy = accuracy_score(val_labels, val_preds)

        epoch_time = time.time() - epoch_start
        metrics_tracker.update(avg_train_loss, val_loss, val_accuracy, epoch_time)

        print(
            f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"*** New best model saved: {val_accuracy:.4f} ***")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    total_training_time = time.time() - start_time
    metrics_tracker.total_training_time = total_training_time

    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print(f"Final Evaluation for {view_name} View")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion, device
    )

    # Compute comprehensive metrics
    test_metrics = compute_metrics(test_labels, test_preds, ACTION_LABELS)
    training_metrics = metrics_tracker.get_best_metrics()

    # Print classification report
    unique_labels = sorted(list(set(test_labels)))
    present_classes = [ACTION_LABELS[i] for i in unique_labels]

    print(f"\nClassification Report for {view_name} View:")
    print("=" * 50)
    print(
        classification_report(
            test_labels,
            test_preds,
            labels=unique_labels,
            target_names=present_classes,
            zero_division=0,
        )
    )

    # Save confusion matrix
    cm_path = os.path.join(
        config.CONFUSION_MATRIX_DIR,
        f"transformer_{view_name.lower()}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        test_labels,
        test_preds,
        ACTION_LABELS,
        cm_path,
        f"Transformer {view_name} View - Confusion Matrix",
    )

    # Compile all results
    results = {
        "view": view_name,
        "model": "Transformer",
        "test_loss": test_loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_minutes": total_training_time / 60,
        **test_metrics,
        **training_metrics,
    }

    # Print summary
    print(f"\n{view_name} View Results Summary:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Training Time: {total_training_time/60:.2f} minutes")

    return results


# --- 4. Data Preparation with Statistics ---
def prepare_data_with_stats(sequences_dict, sequence_ids, split_name):
    """Prepare data and return statistics"""
    data, labels = [], []
    skipped_sequences = 0
    action_counts = Counter()
    sequence_lengths = []

    for seq_id in sequence_ids:
        frames = sequences_dict.get(seq_id)
        if not frames:
            skipped_sequences += 1
            continue

        frames.sort(key=lambda x: x["frame"])
        label = frames[0].get("action_label")

        if label not in label_to_idx:
            skipped_sequences += 1
            continue

        skeletons = [
            np.array(f["skeleton"], dtype=np.float32).flatten() for f in frames
        ]

        if len(skeletons) < 10:  # Minimum sequence length
            skipped_sequences += 1
            continue

        data.append(skeletons)
        labels.append(label_to_idx[label])
        action_counts[label] += 1
        sequence_lengths.append(len(skeletons))

    stats = {
        "split": split_name,
        "total_sequences": len(data),
        "skipped_sequences": skipped_sequences,
        "avg_sequence_length": np.mean(sequence_lengths) if sequence_lengths else 0,
        "min_sequence_length": min(sequence_lengths) if sequence_lengths else 0,
        "max_sequence_length": max(sequence_lengths) if sequence_lengths else 0,
        "action_distribution": dict(action_counts),
        "num_classes": len(action_counts),
    }

    return data, labels, stats


# --- 5. Main Script with Comprehensive Reporting ---
def save_hyperparameters():
    """Save hyperparameters to CSV"""
    hyperparams = {
        "parameter": [
            "sequence_length",
            "batch_size",
            "num_epochs",
            "learning_rate",
            "model_dim",
            "num_heads",
            "num_encoder_layers",
            "dropout",
            "scheduler_step_size",
            "scheduler_gamma",
            "patience",
            "input_features",
            "num_classes",
        ],
        "value": [
            config.SEQUENCE_LENGTH,
            config.BATCH_SIZE,
            config.NUM_EPOCHS,
            config.LEARNING_RATE,
            config.MODEL_DIM,
            config.NHEAD,
            config.NUM_ENCODER_LAYERS,
            config.DROPOUT,
            config.SCHEDULER_STEP_SIZE,
            config.SCHEDULER_GAMMA,
            config.PATIENCE,
            INPUT_FEATURES,
            NUM_CLASSES,
        ],
    }

    df = pd.DataFrame(hyperparams)
    df.to_csv(config.HYPERPARAMS_CSV, index=False)
    print(f"Hyperparameters saved to {config.HYPERPARAMS_CSV}")


if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "TRANSFORMER ACTION RECOGNITION EXPERIMENT")
    print("=" * 80)
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save hyperparameters
    save_hyperparameters()

    # Create output directories
    os.makedirs(config.CONFUSION_MATRIX_DIR, exist_ok=True)

    print("\nLoading all metadata...")
    action_sequences = defaultdict(list)
    training_file_paths = []

    for i in range(config.NUM_FOLDERS):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(config.DATA_PREFIX, folder_name))

    for current_dir_path in training_file_paths:
        metadata_path = os.path.join(current_dir_path, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as f:
            for line in f:
                try:
                    frame_data = json.loads(line.strip())
                    action_num = frame_data.get("action_number")
                    if action_num is not None:
                        action_sequences[action_num].append(frame_data)
                except json.JSONDecodeError:
                    continue

    # Separate inner and outer sequences
    inner_sequences = {
        sid: frames
        for sid, frames in action_sequences.items()
        if frames and "inner_depths" in frames[0].get("file_name", "")
    }
    outer_sequences = {
        sid: frames
        for sid, frames in action_sequences.items()
        if frames and "outer_depths" in frames[0].get("file_name", "")
    }

    print(
        f"Found {len(inner_sequences)} inner sequences and {len(outer_sequences)} outer sequences."
    )

    # Load split configuration
    with open("split_config.json", "r") as f:
        split_config = json.load(f)

    all_results = []
    all_data_stats = []

    for view_name, sequences_dict, save_path in [
        ("Inner", inner_sequences, config.MODEL_SAVE_PATH_INNER),
        ("Outer", outer_sequences, config.MODEL_SAVE_PATH_OUTER),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {view_name} View")
        print(f"{'='*60}")

        # Get sequence IDs for each split
        view_train_ids = [
            sid for sid in split_config["train_sequences"] if sid in sequences_dict
        ]
        view_val_ids = [
            sid for sid in split_config["validation_sequences"] if sid in sequences_dict
        ]
        view_test_ids = [
            sid for sid in split_config["test_sequences"] if sid in sequences_dict
        ]

        # Prepare data with statistics
        train_data, train_labels, train_stats = prepare_data_with_stats(
            sequences_dict, view_train_ids, f"{view_name}_Train"
        )
        val_data, val_labels, val_stats = prepare_data_with_stats(
            sequences_dict, view_val_ids, f"{view_name}_Val"
        )
        test_data, test_labels, test_stats = prepare_data_with_stats(
            sequences_dict, view_test_ids, f"{view_name}_Test"
        )

        # Store data statistics
        for stats in [train_stats, val_stats, test_stats]:
            stats["view"] = view_name
            all_data_stats.append(stats)

        if not train_data:
            print(f"No training data for {view_name} view. Skipping.")
            continue

        # Print data statistics
        print(f"\nData Statistics for {view_name} View:")
        for split_stats in [train_stats, val_stats, test_stats]:
            print(
                f"{split_stats['split']}: {split_stats['total_sequences']} sequences, "
                f"{split_stats['num_classes']} classes, "
                f"avg length: {split_stats['avg_sequence_length']:.1f}"
            )

        # Create datasets and data loaders
        train_dataset = SkeletonActionDataset(train_data, train_labels, augment=True)
        val_dataset = SkeletonActionDataset(val_data, val_labels, augment=False)
        test_dataset = SkeletonActionDataset(test_data, test_labels, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )

        # Calculate class weights
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        weights = [
            total_samples / class_counts.get(i, 1e-9) for i in range(NUM_CLASSES)
        ]
        class_weights = torch.FloatTensor(weights)

        # Train and evaluate model
        results = train_and_evaluate_model(
            train_loader, val_loader, test_loader, save_path, view_name, class_weights
        )

        all_results.append(results)

    # Save detailed results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.round(4)
        results_df.to_csv(config.RESULTS_CSV, index=False)

        # Save data statistics
        stats_df = pd.DataFrame(all_data_stats)
        stats_df.to_csv("data_statistics.csv", index=False)

        # Print final summary table
        print("\n" + "=" * 100)
        print(" " * 35 + "FINAL RESULTS SUMMARY")
        print("=" * 100)

        # Main metrics table
        summary_cols = [
            "view",
            "model",
            "accuracy",
            "f1_weighted",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "training_time_minutes",
        ]
        if all(col in results_df.columns for col in summary_cols):
            summary_df = results_df[summary_cols].copy()
            summary_df.columns = [
                "View",
                "Model",
                "Accuracy",
                "F1-Weighted",
                "F1-Macro",
                "Precision",
                "Recall",
                "Training Time (min)",
            ]
            print(summary_df.to_string(index=False, float_format="%.4f"))

        # Hyperparameters table
        print(f"\n{' '*35}HYPERPARAMETERS")
        print("-" * 100)
        hyperparams_df = pd.read_csv(config.HYPERPARAMS_CSV)
        print(hyperparams_df.to_string(index=False))

        # Data statistics summary
        print(f"\n{' '*35}DATA STATISTICS")
        print("-" * 100)
        stats_summary = stats_df[
            [
                "view",
                "split",
                "total_sequences",
                "num_classes",
                "avg_sequence_length",
                "min_sequence_length",
                "max_sequence_length",
            ]
        ]
        stats_summary.columns = [
            "View",
            "Split",
            "Sequences",
            "Classes",
            "Avg Length",
            "Min Length",
            "Max Length",
        ]
        print(stats_summary.to_string(index=False, float_format="%.1f"))

        print("\n" + "=" * 100)
        print(
            f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Results saved to: {config.RESULTS_CSV}")
        print(f"Hyperparameters saved to: {config.HYPERPARAMS_CSV}")
        print(f"Data statistics saved to: data_statistics.csv")
        print(f"Confusion matrices saved to: {config.CONFUSION_MATRIX_DIR}/")
        print("=" * 100)
    else:
        print("No results to save - no valid data found.")
