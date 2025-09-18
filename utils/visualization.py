import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def create_class_weights_plots(class_weights, class_names, class_counts, save_dir):
    """Create plots specific to weighted loss analysis"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    weights_cpu = class_weights.cpu().numpy()
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]

    # Create bar plot
    x_pos = np.arange(len(class_names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Top plot: Class counts
    bars1 = ax1.bar(x_pos, counts, color="skyblue", alpha=0.7)
    ax1.set_xlabel("Classes")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Class Distribution (Sample Counts)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Bottom plot: Class weights
    bars2 = ax2.bar(x_pos, weights_cpu, color="coral", alpha=0.7)
    ax2.set_xlabel("Classes")
    ax2.set_ylabel("Loss Weight")
    ax2.set_title("Class Weights for Weighted CrossEntropyLoss")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add weight labels on bars
    for bar, weight in zip(bars2, weights_cpu):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(weights_cpu) * 0.01,
            f"{weight:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_dir / "class_weights_analysis.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "class_weights_analysis.pdf", bbox_inches="tight")
    plt.close()

    # Weight vs Count correlation plot
    plt.figure(figsize=(10, 8))
    plt.scatter(counts, weights_cpu, alpha=0.7, s=100)

    for i, class_name in enumerate(class_names):
        plt.annotate(
            class_name,
            (counts[i], weights_cpu[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    plt.xlabel("Sample Count")
    plt.ylabel("Loss Weight")
    plt.title("Class Weight vs Sample Count Correlation")
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(counts, weights_cpu, 1)
    p = np.poly1d(z)
    plt.plot(
        counts,
        p(counts),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "weight_count_correlation.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "weight_count_correlation.pdf", bbox_inches="tight")
    plt.close()

    print(f"✅ Weighted loss analysis plots saved to {save_dir}")


def create_training_curves(training_history, save_dir):
    """Create training curves plots"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    epochs = training_history["epoch"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Loss curves
    axes[0, 0].plot(
        epochs,
        training_history["train_loss"],
        label="Training Loss",
        linewidth=2,
        color="blue",
    )
    axes[0, 0].plot(
        epochs,
        training_history["val_loss"],
        label="Validation Loss",
        linewidth=2,
        color="red",
    )
    axes[0, 0].plot(
        epochs,
        training_history["test_loss"],
        label="Test Loss",
        linewidth=2,
        color="green",
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training/Validation/Test Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(
        epochs,
        training_history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
        color="red",
    )
    axes[0, 1].plot(
        epochs,
        training_history["test_accuracy"],
        label="Test Accuracy",
        linewidth=2,
        color="green",
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Validation/Test Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 scores
    axes[0, 2].plot(
        epochs,
        training_history["val_f1_macro"],
        label="Val F1 Macro",
        linewidth=2,
        color="purple",
    )
    axes[0, 2].plot(
        epochs,
        training_history["test_f1_macro"],
        label="Test F1 Macro",
        linewidth=2,
        color="orange",
    )
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("F1 Score")
    axes[0, 2].set_title("F1 Scores (Macro)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Precision
    axes[1, 0].plot(
        epochs,
        training_history["val_precision_macro"],
        label="Val Precision",
        linewidth=2,
        color="brown",
    )
    axes[1, 0].plot(
        epochs,
        training_history["test_precision_macro"],
        label="Test Precision",
        linewidth=2,
        color="pink",
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision (Macro)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall
    axes[1, 1].plot(
        epochs,
        training_history["val_recall_macro"],
        label="Val Recall",
        linewidth=2,
        color="cyan",
    )
    axes[1, 1].plot(
        epochs,
        training_history["test_recall_macro"],
        label="Test Recall",
        linewidth=2,
        color="magenta",
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall (Macro)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Cohen's Kappa
    axes[1, 2].plot(
        epochs,
        training_history["val_cohen_kappa"],
        label="Val Kappa",
        linewidth=2,
        color="olive",
    )
    axes[1, 2].plot(
        epochs,
        training_history["test_cohen_kappa"],
        label="Test Kappa",
        linewidth=2,
        color="navy",
    )
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Cohen's Kappa")
    axes[1, 2].set_title("Cohen's Kappa Score")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_dir / "training_curves.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        save_dir / "training_curves.pdf", bbox_inches="tight", facecolor="white"
    )
    plt.close()

    print(f"✅ Training curves saved to {save_dir}")
