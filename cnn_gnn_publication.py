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
    precision_recall_fscore_support
)
import seaborn as sns
from tqdm import tqdm
import warnings
import pandas as pd
import time
import random
import os

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Changed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All seeds set to {seed} for reproducibility")

# Set device and optimize CUDA settings
set_seeds(42)  # Call this first for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class OptimizedSpatialCNN(nn.Module):
    """
    Optimized CNN backbone with reduced complexity
    """
    def __init__(self, feature_dim=256, pretrained=True):
        super(OptimizedSpatialCNN, self).__init__()
        
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        self.backbone.classifier = nn.Identity()
        backbone_dim = 1280
        
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.use_attention = False
        if self.use_attention:
            self.spatial_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(backbone_dim, backbone_dim // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(backbone_dim // 16, backbone_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        features = self.backbone.features(x)
        
        if self.use_attention:
            attention = self.spatial_attention(features)
            features = features * attention
        
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        projected_features = self.feature_projector(pooled_features)
        
        return projected_features


class OptimizedTemporalGNN(nn.Module):
    """
    Simplified and optimized GNN module
    """
    def __init__(self, feature_dim=256, hidden_dim=128, num_layers=2, num_heads=4):
        super(OptimizedTemporalGNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                       dropout=0.1, concat=True)
            )
        
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
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


class OptimizedCNNGNN(nn.Module):
    """
    Optimized CNN-GNN model for faster training
    """
    def __init__(self, num_classes, cnn_feature_dim=256, gnn_hidden_dim=128, 
                 gnn_layers=2, num_heads=4, dropout=0.2):
        super(OptimizedCNNGNN, self).__init__()
        
        self.num_classes = num_classes
        self.cnn_feature_dim = cnn_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        
        self.spatial_cnn = OptimizedSpatialCNN(feature_dim=cnn_feature_dim)
        self.temporal_gnn = OptimizedTemporalGNN(
            feature_dim=cnn_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden_dim // 2, num_classes)
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
                gnn_output = self.temporal_gnn(seq_features, edge_indices[i], batch_indices[i])
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
            temporal_features = torch.stack([out.squeeze(0) if out.dim() > 1 else out for out in gnn_outputs])
        else:
            temporal_features = torch.zeros(batch_size, self.gnn_hidden_dim, device=frames.device)
        
        logits = self.classifier(temporal_features)
        
        return logits


class OptimizedVideoDataset(Dataset):
    """
    Optimized dataset with caching and faster loading
    """
    def __init__(self, data_dir, sequence_length=8, image_size=(224, 224),
                 temporal_window=2, train=True, max_sequences=None, cache_frames=True):
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
        
        # Print class distribution for paper
        class_counts = defaultdict(int)
        for seq in self.sequences:
            labels = [item["action_label"] for item in seq["frames"]]
            majority_label = max(set(labels), key=labels.count)
            class_counts[majority_label] += 1
        
        print("\nClass distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} sequences ({count/len(self.sequences)*100:.1f}%)")
        
        # Optimized transforms
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
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
                        if all(field in frame_data for field in ["file_name", "action_label", "action_number"]):
                            frame_data["train_dir"] = train_dir.name
                            frames_data.append(frame_data)
                    except:
                        continue
            
            frames_data.sort(key=lambda x: x.get("action_number", 0))
            
            step_size = self.sequence_length
            for i in range(0, len(frames_data) - self.sequence_length + 1, step_size):
                sequence_frames = frames_data[i:i + self.sequence_length]
                
                if len(sequence_frames) == self.sequence_length:
                    sequences.append({"frames": sequence_frames, "train_dir": train_dir})
                    
                    if self.max_sequences and len(sequences) >= self.max_sequences:
                        return sequences
        
        return sequences
    
    def _create_temporal_graph(self, sequence_frames):
        """Simplified graph creation"""
        num_nodes = len(sequence_frames)
        edge_indices = []
        
        for i in range(num_nodes - 1):
            edge_indices.append([i, i + 1])
            edge_indices.append([i + 1, i])
        
        for i in range(num_nodes):
            for j in range(max(0, i - self.temporal_window), 
                          min(num_nodes, i + self.temporal_window + 1)):
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
            frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            try:
                image = cv2.imread(str(frame_path))
                if image is None:
                    frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
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
            "sequence_id": idx
        }


class ResultsSaver:
    """
    Comprehensive results saver for journal publication
    """
    def __init__(self, save_dir="results", experiment_name=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"cnn_gnn_experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        
        print(f"Results will be saved to: {self.experiment_dir}")
        
        # Initialize results storage
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_macro': [],
            'val_f1_micro': [],
            'val_f1_weighted': [],
            'val_precision_macro': [],
            'val_precision_micro': [],
            'val_precision_weighted': [],
            'val_recall_macro': [],
            'val_recall_micro': [],
            'val_recall_weighted': [],
            'val_cohen_kappa': [],
            'val_matthews_corrcoef': [],
            'epoch_time': [],
            'learning_rate': []
        }
        
        self.final_results = {}
        self.per_class_results = {}
        
    def save_epoch_results(self, epoch, train_loss, val_metrics, epoch_time, learning_rate):
        """Save results for each epoch"""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_accuracy'].append(val_metrics['accuracy'])
        self.training_history['val_f1_macro'].append(val_metrics['f1_macro'])
        self.training_history['val_f1_micro'].append(val_metrics['f1_micro'])
        self.training_history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        self.training_history['val_precision_macro'].append(val_metrics['precision_macro'])
        self.training_history['val_precision_micro'].append(val_metrics['precision_micro'])
        self.training_history['val_precision_weighted'].append(val_metrics['precision_weighted'])
        self.training_history['val_recall_macro'].append(val_metrics['recall_macro'])
        self.training_history['val_recall_micro'].append(val_metrics['recall_micro'])
        self.training_history['val_recall_weighted'].append(val_metrics['recall_weighted'])
        self.training_history['val_cohen_kappa'].append(val_metrics.get('cohen_kappa', 0.0))
        self.training_history['val_matthews_corrcoef'].append(val_metrics.get('matthews_corrcoef', 0.0))
        self.training_history['epoch_time'].append(epoch_time)
        self.training_history['learning_rate'].append(learning_rate)
        
        # Save training history after each epoch
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.experiment_dir / "data" / "training_history.csv", index=False)
    
    def save_final_results(self, final_metrics, class_names, y_true, y_pred, y_proba=None):
        """Save final comprehensive results"""
        self.final_results = final_metrics.copy()
        
        # Save final metrics
        with open(self.experiment_dir / "data" / "final_metrics.json", "w") as f:
            json.dump(self.final_results, f, indent=2)
        
        # Per-class results
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        per_class_df = pd.DataFrame({
            'class_name': class_names,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        
        per_class_df.to_csv(self.experiment_dir / "data" / "per_class_metrics.csv", index=False)
        
        # Detailed classification report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, 
                                          output_dict=True, zero_division=0)
        
        with open(self.experiment_dir / "data" / "classification_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)
        
        # Save predictions for analysis
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_class': [class_names[i] for i in y_true],
            'predicted_class': [class_names[i] for i in y_pred]
        })
        
        if y_proba is not None:
            for i, class_name in enumerate(class_names):
                predictions_df[f'prob_{class_name}'] = y_proba[:, i]
        
        predictions_df.to_csv(self.experiment_dir / "data" / "predictions.csv", index=False)
        
        print(f"Final results saved to {self.experiment_dir / 'data'}")
    
    def create_publication_plots(self, class_names, y_true, y_pred):
        """Create publication-quality plots"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Training curves (2x2 subplot)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curve
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['train_loss'], 
                       label='Training Loss', linewidth=2, color='blue')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['val_loss'], 
                       label='Validation Loss', linewidth=2, color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['val_accuracy'], 
                       label='Validation Accuracy', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 scores
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['val_f1_macro'], 
                       label='F1 Macro', linewidth=2, color='purple')
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['val_f1_weighted'], 
                       label='F1 Weighted', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Scores')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 1].plot(self.training_history['epoch'], self.training_history['val_precision_macro'], 
                       label='Precision', linewidth=2, color='brown')
        axes[1, 1].plot(self.training_history['epoch'], self.training_history['val_recall_macro'], 
                       label='Recall', linewidth=2, color='pink')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall (Macro)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.experiment_dir / "plots" / "training_curves.pdf", bbox_inches='tight')
        plt.show()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.experiment_dir / "plots" / "confusion_matrix.pdf", bbox_inches='tight')
        plt.show()
        
        # 3. Per-class performance bar chart
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(15, 8))
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Per-Class Performance Metrics', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "per_class_performance.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.experiment_dir / "plots" / "per_class_performance.pdf", bbox_inches='tight')
        plt.show()
        
        print(f"Publication plots saved to {self.experiment_dir / 'plots'}")
    
    def save_experiment_config(self, config, model_params):
        """Save experiment configuration"""
        experiment_info = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'model_parameters': model_params,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(device)
        }
        
        if torch.cuda.is_available():
            experiment_info['cuda_device'] = torch.cuda.get_device_name()
            experiment_info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory
        
        with open(self.experiment_dir / "experiment_config.json", "w") as f:
            json.dump(experiment_info, f, indent=2)
        
        print(f"Experiment configuration saved to {self.experiment_dir}")


class OptimizedTrainer:
    """
    Optimized trainer with comprehensive results saving
    """
    def __init__(self, model, train_loader, val_loader, num_classes, class_names,
                 learning_rate=1e-3, weight_decay=1e-4, results_saver=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.results_saver = results_saver
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=learning_rate, 
            steps_per_epoch=len(train_loader), epochs=50
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        
        print("Using mixed precision training for speed optimization")
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None, loss=0.0):
        """Calculate comprehensive metrics"""
        metrics = {
            'loss': loss,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        if y_proba is not None and self.num_classes > 2:
            try:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_weighted'] = 0.0
        
        return metrics
    
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
            edge_indices = [ei.to(device, non_blocking=True) for ei in batch["edge_indices"]]
            batch_indices = [bi.to(device, non_blocking=True) for bi in batch["batch_indices"]]
            
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
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "it/s": f"{it_per_sec:.2f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self):
        """Fast validation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                frames = batch["frames"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                edge_indices = [ei.to(device, non_blocking=True) for ei in batch["edge_indices"]]
                batch_indices = [bi.to(device, non_blocking=True) for bi in batch["batch_indices"]]
                
                try:
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
                    print(f"Error in validation: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if all_labels and all_predictions:
            metrics = self.calculate_comprehensive_metrics(
                all_labels, all_predictions, 
                np.array(all_probabilities) if all_probabilities else None,
                avg_loss
            )
        else:
            metrics = {'loss': avg_loss, 'accuracy': 0.0, 'f1_macro': 0.0}
        
        return metrics, all_predictions, all_labels, all_probabilities
    
    def train(self, num_epochs=30, save_path="optimized_model.pth"):
        """Fast training loop with comprehensive logging"""
        best_f1 = 0
        
        print(f"Starting optimized training for {num_epochs} epochs...")
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics, predictions, labels, probabilities = self.validate()
            
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            
            # Save epoch results
            if self.results_saver:
                self.results_saver.save_epoch_results(
                    epoch + 1, train_loss, val_metrics, epoch_time, current_lr
                )
            
            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                
                # Save to results directory if available
                if self.results_saver:
                    model_save_path = self.results_saver.experiment_dir / "models" / "best_model.pth"
                else:
                    model_save_path = save_path
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_score': val_metrics['f1_macro'],
                    'accuracy': val_metrics['accuracy'],
                    'all_metrics': val_metrics
                }, model_save_path)
                
                print(f"✓ Best model saved (F1: {best_f1:.4f})")
                
                # Save final results for best model
                if self.results_saver:
                    self.results_saver.save_final_results(
                        val_metrics, self.class_names, labels, predictions,
                        np.array(probabilities) if probabilities else None
                    )
        
        total_time = time.time() - total_start_time
        print(f"\nTraining completed in {total_time:.1f} seconds")
        print(f"Best F1 Score: {best_f1:.4f}")
        
        # Create publication plots
        if self.results_saver and labels and predictions:
            self.results_saver.create_publication_plots(self.class_names, labels, predictions)
        
        return best_f1


def main():
    """Optimized main function with comprehensive results saving"""
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Optimized configuration for speed
    config = {
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
        "seed": 42
    }
    
    print("🚀 PUBLICATION-READY CNN-GNN Video Classification")
    print("=" * 60)
    print(f"Configuration: {config}")
    
    try:
        # Create results saver
        experiment_name = f"cnn_gnn_seq{config['sequence_length']}_bs{config['batch_size']}_lr{config['learning_rate']}"
        results_saver = ResultsSaver(experiment_name=experiment_name)
        
        # Create optimized dataset
        print("\nCreating optimized dataset...")
        dataset = OptimizedVideoDataset(
            data_dir=config["data_dir"],
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            temporal_window=config["temporal_window"],
            train=True,
            max_sequences=config["max_sequences"],
            cache_frames=True
        )
        
        if len(dataset) == 0:
            print("No sequences found!")
            return
        
        # Split dataset with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config["seed"])
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
        
        # Optimized data loaders
        train_loader = DataLoader(
            train_subset, batch_size=config["batch_size"], shuffle=True,
            collate_fn=collate_fn, num_workers=2, pin_memory=True, 
            persistent_workers=True, generator=generator
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=config["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True, persistent_workers=True
        )
        
        # Create optimized model
        print(f"\nCreating optimized model ({dataset.num_classes} classes)...")
        model = OptimizedCNNGNN(
            num_classes=dataset.num_classes,
            cnn_feature_dim=config["cnn_feature_dim"],
            gnn_hidden_dim=config["gnn_hidden_dim"],
            gnn_layers=config["gnn_layers"],
            num_heads=config["num_heads"]
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Save experiment configuration
        model_params = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': dataset.num_classes,
            'class_names': list(dataset.label_to_idx.keys())
        }
        results_saver.save_experiment_config(config, model_params)
        
        # Create optimized trainer
        trainer = OptimizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=dataset.num_classes,
            class_names=list(dataset.label_to_idx.keys()),
            learning_rate=config["learning_rate"],
            results_saver=results_saver
        )
        
        # Train with optimizations
        print("\n🏃‍♂️ Starting speed-optimized training...")
        best_f1 = trainer.train(
            num_epochs=config["num_epochs"],
            save_path="optimized_cnn_gnn_model.pth"
        )
        
        print(f"\n🎉 Training completed!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(f"All results saved to: {results_saver.experiment_dir}")
        
        # Print summary of saved files
        print("\n📁 Saved files:")
        print(f"  - Training history: {results_saver.experiment_dir}/data/training_history.csv")
        print(f"  - Final metrics: {results_saver.experiment_dir}/data/final_metrics.json")
        print(f"  - Per-class metrics: {results_saver.experiment_dir}/data/per_class_metrics.csv")
        print(f"  - Predictions: {results_saver.experiment_dir}/data/predictions.csv")
        print(f"  - Training plots: {results_saver.experiment_dir}/plots/")
        print(f"  - Best model: {results_saver.experiment_dir}/models/best_model.pth")
        print(f"  - Config: {results_saver.experiment_dir}/experiment_config.json")
        
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
        "labels": labels
    }


if __name__ == "__main__":
    main()