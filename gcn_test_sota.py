import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
import random

# --- Configuration ---
DATA_PREFIX = "./data"
NUM_FOLDERS = 34
SEQUENCE_LENGTH = 150
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
MODEL_SAVE_PATH_INNER = "best_inner_camera_2s_gcn_model.pth"
MODEL_SAVE_PATH_OUTER = "best_outer_camera_2s_gcn_model.pth"

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

# --- 1. Define the Skeleton Graph Structure ---
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

# A mapping from each joint to its parent in a kinematic tree (0 is the root)
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


# --- 2. GCN Model Definition (ST-GCN) ---
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
                STGCNBlock(NUM_COORDS, 64, kernel_size=9),
                STGCNBlock(64, 64, kernel_size=9),
                STGCNBlock(64, 128, kernel_size=9),
                STGCNBlock(128, 128, kernel_size=9),
                STGCNBlock(128, 256, kernel_size=9),
                STGCNBlock(256, 256, kernel_size=9),
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


# --- 3. Two-Stream GCN Model ---
class TwoStreamSTGCN(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamSTGCN, self).__init__()
        self.joint_stream = STGCN(num_classes)
        self.bone_stream = STGCN(num_classes)

    def forward(self, x_joint, x_bone):
        out_joint = self.joint_stream(x_joint)
        out_bone = self.bone_stream(x_bone)
        # Fuse by averaging the outputs
        return (out_joint + out_bone) / 2


# --- 4. Custom PyTorch Dataset for Two Streams ---
class TwoStreamGCN_Dataset(Dataset):
    def __init__(self, sequences_data, labels_data):
        self.sequences = sequences_data
        self.labels = labels_data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # --- Joint Stream Data ---
        joint_sequence_np = np.array(sequence).reshape(-1, NUM_JOINTS, NUM_COORDS)
        pelvis_coords = joint_sequence_np[:, 0:1, :]
        normalized_joints = joint_sequence_np - pelvis_coords

        # --- Bone Stream Data ---
        bone_sequence_np = np.zeros_like(joint_sequence_np)
        for i, parent in PARENT_MAP.items():
            bone_sequence_np[:, i, :] = (
                joint_sequence_np[:, i, :] - joint_sequence_np[:, parent, :]
            )

        # --- Padding ---
        padded_joints = np.zeros(
            (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS), dtype=np.float32
        )
        padded_bones = np.zeros(
            (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS), dtype=np.float32
        )
        seq_len = min(len(normalized_joints), SEQUENCE_LENGTH)
        padded_joints[:seq_len] = normalized_joints[:seq_len]
        padded_bones[:seq_len] = bone_sequence_np[:seq_len]

        return (
            torch.tensor(padded_joints),
            torch.tensor(padded_bones),
            torch.tensor(label, dtype=torch.long),
        )


# --- 5. Training Function ---
def train_and_evaluate_model(
    train_loader, val_loader, test_loader, model_save_path, model_name, class_weights
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamSTGCN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    print(f"\n--- Starting Training for {model_name} 2s-GCN Model ---")
    best_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for joints, bones, labels in train_loader:
            joints, bones, labels = (
                joints.to(device),
                bones.to(device),
                labels.to(device),
            )
            outputs = model(joints, bones)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for joints, bones, labels in val_loader:
                joints, bones, labels = (
                    joints.to(device),
                    bones.to(device),
                    labels.to(device),
                )
                outputs = model(joints, bones)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"*** New best model saved: {best_accuracy:.2f}% ***")
        scheduler.step()

    print(f"\n--- Final Testing for {model_name} 2s-GCN Model ---")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for joints, bones, labels in test_loader:
            joints, bones, labels = (
                joints.to(device),
                bones.to(device),
                labels.to(device),
            )
            outputs = model(joints, bones)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total if total > 0 else 0
    print(f"Final Accuracy on the Test Set for {model_name}: {test_accuracy:.2f}%")


# --- 6. Main Script ---
if __name__ == "__main__":
    # Data loading and splitting logic remains the same
    print("Loading all metadata...")
    action_sequences = defaultdict(list)
    # ... (rest of data loading logic)
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

    print(
        f"\nFound {len(inner_sequences)} inner sequences and {len(outer_sequences)} outer sequences."
    )

    for view_name, sequences_dict, save_path in [
        ("Inner", inner_sequences, MODEL_SAVE_PATH_INNER),
        ("Outer", outer_sequences, MODEL_SAVE_PATH_OUTER),
    ]:

        with open("split_config.json", "r") as f:
            split_config = json.load(f)

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
            print(f"\nNo training data for {view_name} view. Skipping.")
            continue

        train_dataset = TwoStreamGCN_Dataset(train_data, train_labels)
        val_dataset = TwoStreamGCN_Dataset(val_data, val_labels)
        test_dataset = TwoStreamGCN_Dataset(test_data, test_labels)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        weights = [
            total_samples / class_counts.get(i, 1e-9) for i in range(NUM_CLASSES)
        ]
        class_weights = torch.FloatTensor(weights)

        train_and_evaluate_model(
            train_loader, val_loader, test_loader, save_path, view_name, class_weights
        )
