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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- Configuration ---
DATA_PREFIX = "./data"
NUM_FOLDERS = 34
SEQUENCE_LENGTH = 150
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005

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
NUM_JOINTS = 32
NUM_COORDS = 3
INPUT_FEATURES = NUM_JOINTS * NUM_COORDS

# --- Skeleton Graph Structure (for GCN) ---
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


def get_adjacency_matrix():
    A = np.zeros((NUM_JOINTS, NUM_JOINTS))
    for i, j in EDGES:
        A[i, j] = 1
        A[j, i] = 1
    I = np.identity(NUM_JOINTS)
    A_hat = A + I
    D_hat_diag = np.sum(A_hat, axis=1)
    D_hat_inv_sqrt = np.power(D_hat_diag, -0.5)
    D_hat_inv_sqrt[np.isinf(D_hat_inv_sqrt)] = 0.0
    return torch.tensor(
        np.diag(D_hat_inv_sqrt) @ A_hat @ np.diag(D_hat_inv_sqrt), dtype=torch.float32
    )


ADJACENCY_MATRIX = get_adjacency_matrix()

# --- MODEL ARCHITECTURES ---


# 1. LSTM Model
class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActionRecognitionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), 256).to(x.device)
        c0 = torch.zeros(4, x.size(0), 256).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# 2. GCN Model (2s-AGCN style)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        return torch.einsum("ij,btjf->btif", adj, x)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        super(STGCNBlock, self).__init__()
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
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        res = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.gcn(x, adj).permute(0, 3, 1, 2)
        x = self.tcn(x).permute(0, 2, 3, 1)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, num_classes):
        super(STGCN, self).__init__()
        self.data_bn = nn.BatchNorm1d(NUM_JOINTS * NUM_COORDS)
        self.layers = nn.ModuleList(
            [
                STGCNBlock(NUM_COORDS, 64, 9),
                STGCNBlock(64, 64, 9),
                STGCNBlock(64, 128, 9),
                STGCNBlock(128, 128, 9),
                STGCNBlock(128, 256, 9),
                STGCNBlock(256, 256, 9),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, V, C = x.shape
        x = (
            self.data_bn(x.reshape(B, T, V * C).permute(0, 2, 1))
            .permute(0, 2, 1)
            .reshape(B, T, V, C)
        )
        adj = ADJACENCY_MATRIX.to(x.device)
        for layer in self.layers:
            x = layer(x, adj)
        x = self.pool(x.permute(0, 3, 1, 2)).view(B, -1)
        return self.fc(x)


class TwoStreamSTGCN(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamSTGCN, self).__init__()
        self.joint_stream = STGCN(num_classes)
        self.bone_stream = STGCN(num_classes)

    def forward(self, x):  # Modified to accept a tuple of (joints, bones)
        x_joint, x_bone = x
        out_joint = self.joint_stream(x_joint)
        out_bone = self.bone_stream(x_bone)
        return (out_joint + out_bone) / 2


# 3. Transformer Model
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
        dropout=0.5,
    ):
        super(ActionRecognitionTransformer, self).__init__()
        self.input_fc = nn.Linear(input_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.output_fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_fc(x[:, 0, :])


# --- DATASET CLASSES ---
class LstmTransformerDataset(Dataset):
    def __init__(self, sequences_data, labels_data):
        self.sequences = sequences_data
        self.labels = labels_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx], self.labels[idx]
        sequence_np = np.array(sequence)
        pelvis_coords = sequence_np[:, :3]
        normalized_sequence = sequence_np - np.tile(pelvis_coords, 32)
        padded_sequence = np.zeros((SEQUENCE_LENGTH, INPUT_FEATURES), dtype=np.float32)
        seq_len = min(len(normalized_sequence), SEQUENCE_LENGTH)
        padded_sequence[:seq_len] = normalized_sequence[:seq_len]
        return torch.tensor(padded_sequence), torch.tensor(label, dtype=torch.long)


class GcnDataset(Dataset):
    def __init__(self, sequences_data, labels_data):
        self.sequences = sequences_data
        self.labels = labels_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx], self.labels[idx]
        joint_sequence_np = np.array(sequence).reshape(-1, NUM_JOINTS, NUM_COORDS)
        pelvis_coords = joint_sequence_np[:, 0:1, :]
        normalized_joints = joint_sequence_np - pelvis_coords
        bone_sequence_np = np.zeros_like(joint_sequence_np)
        for i, parent in PARENT_MAP.items():
            bone_sequence_np[:, i, :] = (
                joint_sequence_np[:, i, :] - joint_sequence_np[:, parent, :]
            )
        padded_joints = np.zeros(
            (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS), dtype=np.float32
        )
        padded_bones = np.zeros(
            (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS), dtype=np.float32
        )
        seq_len = min(len(normalized_joints), SEQUENCE_LENGTH)
        padded_joints[:seq_len] = normalized_joints[:seq_len]
        padded_bones[:seq_len] = bone_sequence_np[:seq_len]
        return (torch.tensor(padded_joints), torch.tensor(padded_bones)), torch.tensor(
            label, dtype=torch.long
        )


# --- TRAINING & EVALUATION ---
def train_and_evaluate_model(
    model_name, train_loader, val_loader, test_loader, class_weights
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "LSTM":
        model = ActionRecognitionLSTM(INPUT_FEATURES, 256, 2, NUM_CLASSES).to(device)
    elif model_name == "GCN":
        model = TwoStreamSTGCN(NUM_CLASSES).to(device)
    elif model_name == "Transformer":
        model = ActionRecognitionTransformer(INPUT_FEATURES, 128, 4, 3, NUM_CLASSES).to(
            device
        )
    else:
        raise ValueError("Unknown model name")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print(f"\n--- Training {model_name} Model ---")
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for data, labels in train_loader:
            if model_name == "GCN":
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for data, labels in val_loader:
                if model_name == "GCN":
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Val Accuracy: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"best_{model_name.lower()}_model.pth")
        scheduler.step()

    print(f"\n--- Testing {model_name} Model ---")
    model.load_state_dict(torch.load(f"best_{model_name.lower()}_model.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            if model_name == "GCN":
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = (
        precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        * 100
    )
    recall = (
        recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    )
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("Loading and preparing data...")
    action_sequences = defaultdict(list)
    # ... (Data loading logic) ...
    training_file_paths = []
    for i in range(NUM_FOLDERS):
        folder_name = f"train-{i:03d}"
        training_file_paths.append(os.path.join(DATA_PREFIX, folder_name))
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

    with open("split_config.json", "r") as f:
        split_config = json.load(f)

    results = []
    for view_name, sequences_dict in [
        ("Inner", inner_sequences),
        ("Outer", outer_sequences),
    ]:

        view_train_ids = [
            sid for sid in split_config["train_sequences"] if sid in sequences_dict
        ]
        view_val_ids = [
            sid for sid in split_config["validation_sequences"] if sid in sequences_dict
        ]
        view_test_ids = [
            sid for sid in split_config["test_sequences"] if sid in sequences_dict
        ]

        def prepare_data(ids):
            data, labels = [], []
            for seq_id in ids:
                frames = sequences_dict.get(seq_id)
                if not frames:
                    continue
                frames.sort(key=lambda x: x["frame"])
                label = frames[0].get("action_label")
                if label not in label_to_idx:
                    continue
                skeletons = [
                    np.array(f["skeleton"], dtype=np.float32).flatten() for f in frames
                ]
                data.append(skeletons)
                labels.append(label_to_idx[label])
            return data, labels

        train_data, train_labels = prepare_data(view_train_ids)
        val_data, val_labels = prepare_data(view_val_ids)
        test_data, test_labels = prepare_data(view_test_ids)

        if not train_data:
            continue

        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        weights = torch.FloatTensor(
            [total_samples / class_counts.get(i, 1e-9) for i in range(NUM_CLASSES)]
        )

        # Create datasets
        lstm_transformer_train_ds = LstmTransformerDataset(train_data, train_labels)
        lstm_transformer_val_ds = LstmTransformerDataset(val_data, val_labels)
        lstm_transformer_test_ds = LstmTransformerDataset(test_data, test_labels)
        gcn_train_ds = GcnDataset(train_data, train_labels)
        gcn_val_ds = GcnDataset(val_data, val_labels)
        gcn_test_ds = GcnDataset(test_data, test_labels)

        # Create dataloaders
        lstm_train_dl = DataLoader(
            lstm_transformer_train_ds, batch_size=32, shuffle=True
        )
        lstm_val_dl = DataLoader(lstm_transformer_val_ds, batch_size=32, shuffle=False)
        lstm_test_dl = DataLoader(
            lstm_transformer_test_ds, batch_size=32, shuffle=False
        )
        gcn_train_dl = DataLoader(gcn_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        gcn_val_dl = DataLoader(gcn_val_ds, batch_size=BATCH_SIZE, shuffle=False)
        gcn_test_dl = DataLoader(gcn_test_ds, batch_size=BATCH_SIZE, shuffle=False)
        transformer_train_dl = DataLoader(
            lstm_transformer_train_ds, batch_size=BATCH_SIZE, shuffle=True
        )
        transformer_val_dl = DataLoader(
            lstm_transformer_val_ds, batch_size=BATCH_SIZE, shuffle=False
        )
        transformer_test_dl = DataLoader(
            lstm_transformer_test_ds, batch_size=BATCH_SIZE, shuffle=False
        )

        # Train and evaluate models
        for model_name, train_dl, val_dl, test_dl in [
            ("LSTM", lstm_train_dl, lstm_val_dl, lstm_test_dl),
            ("GCN", gcn_train_dl, gcn_val_dl, gcn_test_dl),
            (
                "Transformer",
                transformer_train_dl,
                transformer_val_dl,
                transformer_test_dl,
            ),
        ]:
            metrics = train_and_evaluate_model(
                model_name, train_dl, val_dl, test_dl, weights
            )
            metrics["Model"] = model_name
            metrics["View"] = view_name
            results.append(metrics)

    # --- Final Report ---
    print("\n\n" + "=" * 80)
    print(" " * 25 + "FINAL MODEL COMPARISON REPORT")
    print("=" * 80)
    report_df = pd.DataFrame(results)[
        ["Model", "View", "Accuracy", "F1-Score", "Precision", "Recall"]
    ]
    report_df = report_df.round(2)
    print(report_df.to_string(index=False))
    report_df.to_csv("model_comparison_report.csv", index=False)
    print("\n" + "=" * 80)
    print("Report saved to model_comparison_report.csv")
