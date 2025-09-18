import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support


class ResultsSaver:
    """Comprehensive results saver for weighted loss experiments"""

    def __init__(self, save_dir="results", experiment_name=None, device=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"weighted_efficientnet_experiment_{timestamp}"

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
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "val_accuracy": [],
            "test_accuracy": [],
            "val_f1_macro": [],
            "test_f1_macro": [],
            "val_f1_micro": [],
            "test_f1_micro": [],
            "val_f1_weighted": [],
            "test_f1_weighted": [],
            "val_precision_macro": [],
            "test_precision_macro": [],
            "val_precision_micro": [],
            "test_precision_micro": [],
            "val_precision_weighted": [],
            "test_precision_weighted": [],
            "val_recall_macro": [],
            "test_recall_macro": [],
            "val_recall_micro": [],
            "test_recall_micro": [],
            "val_recall_weighted": [],
            "test_recall_weighted": [],
            "val_cohen_kappa": [],
            "test_cohen_kappa": [],
            "val_matthews_corrcoef": [],
            "test_matthews_corrcoef": [],
            "epoch_time": [],
            "learning_rate": [],
        }

    def save_epoch_results(
        self, epoch, train_loss, val_metrics, test_metrics, epoch_time, learning_rate
    ):
        """Save results for each epoch including test metrics"""
        self.training_history["epoch"].append(epoch)
        self.training_history["train_loss"].append(train_loss)
        self.training_history["val_loss"].append(val_metrics["loss"])
        self.training_history["test_loss"].append(test_metrics["loss"])
        self.training_history["val_accuracy"].append(val_metrics["accuracy"])
        self.training_history["test_accuracy"].append(test_metrics["accuracy"])
        self.training_history["val_f1_macro"].append(val_metrics["f1_macro"])
        self.training_history["test_f1_macro"].append(test_metrics["f1_macro"])
        self.training_history["val_f1_micro"].append(val_metrics["f1_micro"])
        self.training_history["test_f1_micro"].append(test_metrics["f1_micro"])
        self.training_history["val_f1_weighted"].append(val_metrics["f1_weighted"])
        self.training_history["test_f1_weighted"].append(test_metrics["f1_weighted"])
        self.training_history["val_precision_macro"].append(
            val_metrics["precision_macro"]
        )
        self.training_history["test_precision_macro"].append(
            test_metrics["precision_macro"]
        )
        self.training_history["val_precision_micro"].append(
            val_metrics["precision_micro"]
        )
        self.training_history["test_precision_micro"].append(
            test_metrics["precision_micro"]
        )
        self.training_history["val_precision_weighted"].append(
            val_metrics["precision_weighted"]
        )
        self.training_history["test_precision_weighted"].append(
            test_metrics["precision_weighted"]
        )
        self.training_history["val_recall_macro"].append(val_metrics["recall_macro"])
        self.training_history["test_recall_macro"].append(test_metrics["recall_macro"])
        self.training_history["val_recall_micro"].append(val_metrics["recall_micro"])
        self.training_history["test_recall_micro"].append(test_metrics["recall_micro"])
        self.training_history["val_recall_weighted"].append(
            val_metrics["recall_weighted"]
        )
        self.training_history["test_recall_weighted"].append(
            test_metrics["recall_weighted"]
        )
        self.training_history["val_cohen_kappa"].append(
            val_metrics.get("cohen_kappa", 0.0)
        )
        self.training_history["test_cohen_kappa"].append(
            test_metrics.get("cohen_kappa", 0.0)
        )
        self.training_history["val_matthews_corrcoef"].append(
            val_metrics.get("matthews_corrcoef", 0.0)
        )
        self.training_history["test_matthews_corrcoef"].append(
            test_metrics.get("matthews_corrcoef", 0.0)
        )
        self.training_history["epoch_time"].append(epoch_time)
        self.training_history["learning_rate"].append(learning_rate)

        # Save training history after each epoch
        df = pd.DataFrame(self.training_history)
        df.to_csv(self.experiment_dir / "data" / "training_history.csv", index=False)

    def save_final_results(
        self,
        final_metrics,
        class_names,
        y_true,
        y_pred,
        y_proba=None,
        split_name="test",
    ):
        """Save final comprehensive results"""
        # Save final metrics
        with open(
            self.experiment_dir / "data" / f"final_metrics_{split_name}.json", "w"
        ) as f:
            json.dump(final_metrics, f, indent=2)

        # Per-class results
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class_df = pd.DataFrame(
            {
                "class_name": class_names,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

        per_class_df.to_csv(
            self.experiment_dir / "data" / f"per_class_metrics_{split_name}.csv",
            index=False,
        )

        # Save predictions for analysis
        predictions_df = pd.DataFrame(
            {
                "true_label": y_true,
                "predicted_label": y_pred,
                "true_class": [class_names[i] for i in y_true],
                "predicted_class": [class_names[i] for i in y_pred],
            }
        )

        if y_proba is not None:
            for i, class_name in enumerate(class_names):
                predictions_df[f"prob_{class_name}"] = y_proba[:, i]

        predictions_df.to_csv(
            self.experiment_dir / "data" / f"predictions_{split_name}.csv", index=False
        )

        print(f"Final {split_name} results saved to {self.experiment_dir / 'data'}")

    def save_experiment_config(self, config, model_params, class_weights_info):
        """Save experiment configuration including weighted loss details"""
        experiment_info = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "model_parameters": model_params,
            "class_weights_info": class_weights_info,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
        }

        if torch.cuda.is_available():
            experiment_info["cuda_device"] = torch.cuda.get_device_name()
            experiment_info["cuda_memory"] = torch.cuda.get_device_properties(
                0
            ).total_memory

        with open(self.experiment_dir / "experiment_config.json", "w") as f:
            json.dump(experiment_info, f, indent=2)

        print(f"Experiment configuration saved to {self.experiment_dir}")

    def create_weighted_loss_plots(self, class_weights, class_names, class_counts):
        """Create plots specific to weighted loss analysis"""
        plots_dir = self.experiment_dir / "plots"

        # Class weights visualization
        plt.figure(figsize=(14, 8))

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
        plt.savefig(
            plots_dir / "class_weights_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "class_weights_analysis.pdf", bbox_inches="tight")
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
        plt.savefig(
            plots_dir / "weight_count_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(plots_dir / "weight_count_correlation.pdf", bbox_inches="tight")
        plt.close()

        print(f"✅ Weighted loss analysis plots saved to {plots_dir}")
