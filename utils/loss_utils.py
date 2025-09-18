import torch
import numpy as np


class WeightedLossCalculator:
    """Calculate class weights for imbalanced datasets with enhanced weighting for rare classes"""

    def __init__(self, weighting_strategy="inverse_sqrt", rare_class_boost=2.0):
        """
        Args:
            weighting_strategy: 'inverse', 'inverse_sqrt', 'balanced', 'log_balanced'
            rare_class_boost: Additional multiplier for classes below median frequency
        """
        self.weighting_strategy = weighting_strategy
        self.rare_class_boost = rare_class_boost

    def calculate_class_weights(self, class_counts, num_classes, device):
        """
        Calculate class weights with enhanced penalties for rare classes

        Args:
            class_counts: Counter object with class frequencies
            num_classes: Total number of classes
            device: torch device

        Returns:
            torch.Tensor: Class weights for CrossEntropyLoss
        """
        # Get class frequencies in order
        class_freqs = np.zeros(num_classes)
        for class_idx, count in class_counts.items():
            class_freqs[class_idx] = count

        # Avoid division by zero
        class_freqs = np.maximum(class_freqs, 1)

        # Calculate base weights using selected strategy
        if self.weighting_strategy == "inverse":
            weights = 1.0 / class_freqs
        elif self.weighting_strategy == "inverse_sqrt":
            weights = 1.0 / np.sqrt(class_freqs)
        elif self.weighting_strategy == "balanced":
            total_samples = np.sum(class_freqs)
            weights = total_samples / (num_classes * class_freqs)
        elif self.weighting_strategy == "log_balanced":
            total_samples = np.sum(class_freqs)
            weights = np.log(total_samples / class_freqs + 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # Apply additional boost to rare classes (below median frequency)
        median_freq = np.median(class_freqs)
        rare_class_mask = class_freqs < median_freq
        weights[rare_class_mask] *= self.rare_class_boost

        # Normalize weights to prevent extreme values
        weights = weights / np.mean(weights)

        # Convert to tensor
        class_weights = torch.FloatTensor(weights).to(device)

        return class_weights

    def print_weight_analysis(self, class_weights, class_names, class_counts):
        """Print detailed analysis of calculated weights"""
        print(f"\n{'='*80}")
        print("🎯 CLASS WEIGHT ANALYSIS")
        print(f"{'='*80}")
        print(f"Weighting Strategy: {self.weighting_strategy}")
        print(f"Rare Class Boost: {self.rare_class_boost}x")
        print(f"\n{'Class Name':<25} {'Count':<8} {'Weight':<12} {'Penalty':<10}")
        print("-" * 65)

        weights_cpu = class_weights.cpu().numpy()

        for i, (class_name, weight) in enumerate(zip(class_names, weights_cpu)):
            count = class_counts.get(i, 0)
            penalty = weight / weights_cpu.min()
            print(f"{class_name:<25} {count:<8} {weight:<12.4f} {penalty:<10.2f}x")

        print("-" * 65)
        print(f"Weight Range: {weights_cpu.min():.4f} - {weights_cpu.max():.4f}")
        print(f"Max Penalty Ratio: {(weights_cpu.max() / weights_cpu.min()):.2f}x")
        print(f"Mean Weight: {weights_cpu.mean():.4f}")
        print(f"Std Weight: {weights_cpu.std():.4f}")


# Additional loss functions
class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=1, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(torch.nn.Module):
    """Label Smoothing Loss"""

    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smooth = (
            1 - self.smoothing
        ) * targets_one_hot + self.smoothing / self.num_classes

        if self.weight is not None:
            targets_smooth = targets_smooth * self.weight.unsqueeze(0)

        loss = (-targets_smooth * log_probs).sum(dim=1)
        return loss.mean()
