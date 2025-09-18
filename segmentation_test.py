import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


class HumanSegmentationModel:
    def __init__(self, model_type="deeplabv3", device="auto"):
        """
        Initialize human segmentation model

        Args:
            model_type: 'deeplabv3', 'fcn', 'maskrcnn', or 'detectron2'
            device: 'auto', 'cuda', or 'cpu'
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )
        self.model_type = model_type

        if model_type == "deeplabv3":
            self.model = self._load_deeplabv3()
        elif model_type == "fcn":
            self.model = self._load_fcn()
        elif model_type == "maskrcnn":
            self.model = self._load_maskrcnn()
        elif model_type == "detectron2":
            self.model = self._load_detectron2()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"Loaded {model_type} model on {self.device}")

    def _load_deeplabv3(self):
        """Load DeepLabV3 model for semantic segmentation"""
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_fcn(self):
        """Load FCN model for semantic segmentation"""
        model = models.segmentation.fcn_resnet101(pretrained=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_maskrcnn(self):
        """Load Mask R-CNN model for instance segmentation"""
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.to(self.device)
        model.eval()
        return model

    def _load_detectron2(self):
        """Load Detectron2 model (requires detectron2 installation)"""
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg

            cfg = get_cfg()
            cfg.MODEL.DEVICE = str(self.device)
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            return DefaultPredictor(cfg)
        except ImportError:
            raise ImportError(
                "Detectron2 not installed. Install with: pip install detectron2"
            )

    def preprocess_image(self, image, return_steps=False):
        """Preprocess image for semantic segmentation models with visualization steps"""
        preprocessing_steps = {}

        # Step 1: Load and convert image
        if isinstance(image, str):
            # Load from file path
            original_pil = Image.open(image).convert("RGB")
            preprocessing_steps["step_1_loaded"] = np.array(original_pil)
        elif isinstance(image, np.ndarray):
            # Convert from numpy array
            original_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            preprocessing_steps["step_1_loaded"] = np.array(original_pil)
        else:
            original_pil = image
            preprocessing_steps["step_1_loaded"] = np.array(original_pil)

        # Step 2: Convert to tensor (0-1 range)
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(original_pil)

        # Convert back to numpy for visualization (0-1 range)
        tensor_as_numpy = tensor_image.permute(1, 2, 0).numpy()
        preprocessing_steps["step_2_tensor"] = tensor_as_numpy

        # Step 3: Apply normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        normalized_tensor = normalize(tensor_image)

        # Convert normalized tensor back to numpy for visualization
        # Need to denormalize for proper visualization
        denorm_tensor = normalized_tensor.clone()
        denorm_tensor[0] = denorm_tensor[0] * 0.229 + 0.485
        denorm_tensor[1] = denorm_tensor[1] * 0.224 + 0.456
        denorm_tensor[2] = denorm_tensor[2] * 0.225 + 0.406
        denorm_numpy = denorm_tensor.permute(1, 2, 0).numpy()
        denorm_numpy = np.clip(denorm_numpy, 0, 1)
        preprocessing_steps["step_3_normalized"] = denorm_numpy

        # Final tensor for model input
        final_tensor = normalized_tensor.unsqueeze(0).to(self.device)

        if return_steps:
            return final_tensor, original_pil, preprocessing_steps
        else:
            return final_tensor, original_pil

    def visualize_preprocessing_steps(self, preprocessing_steps, save_path=None):
        """Visualize the preprocessing steps"""
        steps = list(preprocessing_steps.keys())
        n_steps = len(steps)

        fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 5))
        if n_steps == 1:
            axes = [axes]

        titles = {
            "step_1_loaded": "Original Image",
            "step_2_tensor": "After ToTensor (0-1)",
            "step_3_normalized": "After Normalization",
        }

        for i, step in enumerate(steps):
            image = preprocessing_steps[step]
            axes[i].imshow(image)
            axes[i].set_title(titles.get(step, step))
            axes[i].axis("off")

            # Add image statistics as text
            stats_text = f"Shape: {image.shape}\n"
            stats_text += f"Min: {image.min():.3f}\n"
            stats_text += f"Max: {image.max():.3f}\n"
            stats_text += f"Mean: {image.mean():.3f}"

            axes[i].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Preprocessing visualization saved to {save_path}")

        plt.show()

    def semantic_segmentation(self, image, visualize_preprocessing=False):
        """Perform semantic segmentation using DeepLabV3 or FCN"""
        if visualize_preprocessing:
            input_tensor, original_image, preprocessing_steps = self.preprocess_image(
                image, return_steps=True
            )
            self.visualize_preprocessing_steps(
                preprocessing_steps,
                save_path=f"{self.model_type}_preprocessing_steps.png",
            )
        else:
            input_tensor, original_image = self.preprocess_image(image)

        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]

        # Get human class (person = class 15 in COCO/Pascal VOC)
        human_mask = output.argmax(0) == 15  # Person class
        human_mask = human_mask.cpu().numpy().astype(np.uint8)

        # Resize to original image size
        original_size = original_image.size[::-1]  # (height, width)
        human_mask = cv2.resize(
            human_mask, original_image.size, interpolation=cv2.INTER_NEAREST
        )

        return human_mask, original_image

    def instance_segmentation(
        self, image, confidence_threshold=0.5, visualize_preprocessing=False
    ):
        """Perform instance segmentation using Mask R-CNN"""
        if isinstance(image, str):
            image_cv = cv2.imread(image)
            image_pil = Image.open(image).convert("RGB")
            preprocessing_steps = {}
            preprocessing_steps["step_1_loaded"] = np.array(image_pil)
        elif isinstance(image, np.ndarray):
            image_cv = image.copy()
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            preprocessing_steps = {}
            preprocessing_steps["step_1_loaded"] = np.array(image_pil)

        # Preprocess for Mask R-CNN (simpler preprocessing)
        transform = transforms.Compose([transforms.ToTensor()])
        tensor_image = transform(image_pil)

        # Store preprocessing step for visualization
        if visualize_preprocessing:
            tensor_as_numpy = tensor_image.permute(1, 2, 0).numpy()
            preprocessing_steps["step_2_tensor"] = tensor_as_numpy
            self.visualize_preprocessing_steps(
                preprocessing_steps,
                save_path=f"{self.model_type}_preprocessing_steps.png",
            )

        input_tensor = tensor_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(input_tensor)[0]

        # Filter for human detections (person = class 1 in COCO)
        human_indices = (predictions["labels"] == 1) & (
            predictions["scores"] > confidence_threshold
        )

        if not human_indices.any():
            return None, image_pil, []

        # Get human masks and boxes
        human_masks = predictions["masks"][human_indices].cpu().numpy()
        human_boxes = predictions["boxes"][human_indices].cpu().numpy()
        human_scores = predictions["scores"][human_indices].cpu().numpy()

        # Combine all human masks
        combined_mask = np.zeros((image_cv.shape[0], image_cv.shape[1]), dtype=np.uint8)
        for mask in human_masks:
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask_binary)

        return combined_mask, image_pil, human_boxes

    def detectron2_segmentation(self, image, visualize_preprocessing=False):
        """Perform segmentation using Detectron2"""
        if isinstance(image, str):
            image_cv = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image_cv = image.copy()
        else:
            # Convert PIL to cv2
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detectron2 uses raw BGR images, so show the preprocessing
        if visualize_preprocessing:
            preprocessing_steps = {}
            preprocessing_steps["step_1_bgr_input"] = cv2.cvtColor(
                image_cv, cv2.COLOR_BGR2RGB
            )
            self.visualize_preprocessing_steps(
                preprocessing_steps,
                save_path=f"{self.model_type}_preprocessing_steps.png",
            )

        outputs = self.model(image_cv)

        # Filter for human instances (person = class 0 in COCO)
        instances = outputs["instances"]
        human_indices = instances.pred_classes == 0

        if not human_indices.any():
            return None, image_cv

        # Get human masks
        human_masks = instances.pred_masks[human_indices].cpu().numpy()

        # Combine all human masks
        combined_mask = np.zeros((image_cv.shape[0], image_cv.shape[1]), dtype=np.uint8)
        for mask in human_masks:
            combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

        return combined_mask, image_cv

    def segment_human(
        self, image, confidence_threshold=0.5, visualize_preprocessing=False
    ):
        """Main method to segment humans based on model type"""
        if self.model_type in ["deeplabv3", "fcn"]:
            return self.semantic_segmentation(image, visualize_preprocessing)
        elif self.model_type == "maskrcnn":
            mask, original_image, boxes = self.instance_segmentation(
                image, confidence_threshold, visualize_preprocessing
            )
            return mask, original_image
        elif self.model_type == "detectron2":
            return self.detectron2_segmentation(image, visualize_preprocessing)

    def visualize_results(self, image, mask, save_path=None, show_preprocessing=False):
        """Visualize segmentation results with optional preprocessing visualization"""
        if isinstance(image, str):
            original = cv2.imread(image)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        elif hasattr(image, "size"):  # PIL Image
            original = np.array(image)
        else:
            original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask is None:
            print("No humans detected in the image")
            return

        # Create overlay
        overlay = original.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red color for human pixels

        # Blend original and overlay
        result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Human Mask")
        axes[1].axis("off")

        axes[2].imshow(result)
        axes[2].set_title("Segmentation Overlay")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Results saved to {save_path}")

        plt.show()

    def extract_human_region(self, image, mask, padding=10):
        """Extract human region from image using the mask"""
        if mask is None:
            return None

        if isinstance(image, str):
            image = cv2.imread(image)
        elif hasattr(image, "size"):  # PIL Image
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Find bounding box of human region
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add padding
        h, w = mask.shape
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(h, y_max + padding)
        x_max = min(w, x_max + padding)

        # Extract region
        human_region = image[y_min:y_max, x_min:x_max]
        human_mask_region = mask[y_min:y_max, x_min:x_max]

        return human_region, human_mask_region, (x_min, y_min, x_max, y_max)

    def depth_instance_segmentation(
        self, depth_image, rgb_image=None, confidence_threshold=0.5
    ):
        """Perform instance segmentation optimized for depth images"""
        if self.model_type != "maskrcnn":
            raise ValueError("Instance segmentation requires maskrcnn model type")

        # Use RGB if available, otherwise convert depth to 3-channel
        if rgb_image is not None:
            input_image = rgb_image
        else:
            # Convert depth to 3-channel for model compatibility
            depth_normalized = cv2.normalize(
                depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            input_image = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)

        # Use existing instance segmentation
        mask, original_image, boxes = self.instance_segmentation(
            input_image, confidence_threshold=confidence_threshold
        )

        # Post-process using depth information
        if mask is not None and depth_image is not None:
            mask = self._refine_mask_with_depth(mask, depth_image)

        return mask, original_image, boxes

    def _refine_mask_with_depth(self, mask, depth_image, depth_threshold=50):
        """Refine segmentation mask using depth discontinuities"""
        # Find depth edges
        depth_edges = cv2.Canny(depth_image.astype(np.uint8), 50, 150)

        # Use depth edges to separate connected instances
        kernel = np.ones((3, 3), np.uint8)
        depth_edges = cv2.dilate(depth_edges, kernel, iterations=1)

        # Remove mask pixels where there are strong depth discontinuities
        refined_mask = mask.copy()
        refined_mask[depth_edges > 0] = 0

        return refined_mask

    def depth_clustering_segmentation(
        self, depth_image, min_cluster_size=100, depth_tolerance=20
    ):
        """Segment independent objects using depth clustering"""
        from sklearn.cluster import DBSCAN
        from skimage.measure import label, regionprops

        # Normalize depth image
        depth_normalized = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Remove background (very far or very close objects)
        mask = (depth_normalized > 10) & (depth_normalized < 240)

        # Get coordinates of valid depth pixels
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None, depth_image

        # Create feature matrix: [x, y, depth]
        features = np.column_stack(
            [x_coords, y_coords, depth_normalized[y_coords, x_coords]]
        )

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=depth_tolerance, min_samples=min_cluster_size // 4)
        cluster_labels = clustering.fit_predict(features)

        # Create individual masks for each cluster
        object_masks = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue

            cluster_mask = np.zeros_like(depth_image, dtype=np.uint8)
            cluster_indices = cluster_labels == cluster_id
            cluster_y = y_coords[cluster_indices]
            cluster_x = x_coords[cluster_indices]
            cluster_mask[cluster_y, cluster_x] = 1

            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)

            # Filter by size
            if np.sum(cluster_mask) >= min_cluster_size:
                object_masks.append(cluster_mask)

        return object_masks, depth_image

    def visualize_depth_clustering(self, depth_image, object_masks, save_path=None):
        """Visualize depth clustering results"""
        if not object_masks:
            print("No objects detected")
            return

        fig, axes = plt.subplots(
            1, len(object_masks) + 1, figsize=(5 * (len(object_masks) + 1), 5)
        )
        if len(object_masks) == 0:
            axes = [axes]

        # Show original depth image
        axes[0].imshow(depth_image, cmap="viridis")
        axes[0].set_title("Original Depth")
        axes[0].axis("off")

        # Show each object mask
        colors = plt.cm.Set3(np.linspace(0, 1, len(object_masks)))
        combined_mask = np.zeros_like(depth_image)

        for i, mask in enumerate(object_masks):
            axes[i + 1].imshow(mask, cmap="gray")
            axes[i + 1].set_title(f"Object {i+1}")
            axes[i + 1].axis("off")

            combined_mask += mask * (i + 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Depth clustering visualization saved to {save_path}")

        plt.show()

        return combined_mask

    def watershed_depth_segmentation(self, depth_image, min_distance=20):
        """Use watershed algorithm for depth-based object segmentation"""
        from skimage.feature import peak_local_maxima
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi

        # Normalize and invert depth (closer objects become peaks)
        depth_normalized = cv2.normalize(
            depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=np.uint8
        )
        depth_inverted = 255 - depth_normalized

        # Apply Gaussian blur to reduce noise
        depth_smooth = cv2.GaussianBlur(depth_inverted, (5, 5), 0)

        # Find local maxima (object centers)
        local_maxima = peak_local_maxima(
            depth_smooth, min_distance=min_distance, threshold_abs=50
        )

        # Create markers for watershed
        markers = np.zeros_like(depth_smooth, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1

        # Apply watershed
        labels = watershed(-depth_smooth, markers, mask=depth_normalized > 10)

        # Convert to individual masks
        object_masks = []
        for label_id in np.unique(labels):
            if label_id == 0:  # Skip background
                continue
            mask = (labels == label_id).astype(np.uint8)
            if np.sum(mask) > 100:  # Filter small objects
                object_masks.append(mask)

        return object_masks, depth_image

    def segment_objects_depth(
        self, depth_image, rgb_image=None, method="clustering", **kwargs
    ):
        """Main method to segment independent objects in depth images"""
        if method == "clustering":
            return self.depth_clustering_segmentation(depth_image, **kwargs)
        elif method == "watershed":
            return self.watershed_depth_segmentation(depth_image, **kwargs)
        elif method == "instance" and self.model_type == "maskrcnn":
            return self.depth_instance_segmentation(depth_image, rgb_image, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")


# Usage example - UPDATED MAIN FUNCTION
def main():
    """Comprehensive testing of all segmentation methods"""
    print("=" * 60)
    print("COMPREHENSIVE SEGMENTATION TESTING")
    print("=" * 60)

    # Test image paths
    rgb_image_path = "./data/train-000/inner_depths/000101.png"  # Replace with RGB image if available
    depth_image_path = "./data/train-000/inner_depths/000101.png"  # Your depth image

    # Load images
    try:
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        if depth_image is None:
            print(f"Could not load depth image from {depth_image_path}")
            return
        print(f"Loaded depth image: {depth_image.shape}")

        # Try to load RGB image (optional)
        rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
        if rgb_image is not None:
            print(f"Loaded RGB image: {rgb_image.shape}")
        else:
            print("No RGB image available, using depth-only processing")

    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # ========================================
    # PART 1: Traditional RGB-based Segmentation
    # ========================================
    print("\n" + "=" * 50)
    print("PART 1: RGB-BASED HUMAN SEGMENTATION")
    print("=" * 50)

    models_to_test = ["deeplabv3", "maskrcnn"]

    for model_name in models_to_test:
        print(f"\n--- Testing {model_name.upper()} Model ---")

        try:
            segmenter = HumanSegmentationModel(model_type=model_name)

            # Use RGB image if available, otherwise convert depth
            if rgb_image is not None:
                test_image = rgb_image_path
            else:
                test_image = depth_image_path

            # Perform segmentation with preprocessing visualization
            if model_name == "maskrcnn":
                mask, original_image, boxes = segmenter.instance_segmentation(
                    test_image, visualize_preprocessing=True
                )
                print(f"Detected {len(boxes)} human instances")
            else:
                mask, original_image = segmenter.segment_human(
                    test_image, visualize_preprocessing=True
                )

            # Visualize results
            if mask is not None:
                segmenter.visualize_results(
                    original_image,
                    mask,
                    save_path=f"{model_name}_human_segmentation.png",
                )

                # Extract human region
                human_region = segmenter.extract_human_region(original_image, mask)
                if human_region is not None:
                    region, region_mask, bbox = human_region
                    print(f"Human bounding box: {bbox}")
                    cv2.imwrite(f"{model_name}_human_region.jpg", region)
            else:
                print("No humans detected")

        except Exception as e:
            print(f"Error with {model_name}: {e}")

    # ========================================
    # PART 2: Depth-based Object Segmentation
    # ========================================
    print("\n" + "=" * 50)
    print("PART 2: DEPTH-BASED OBJECT SEGMENTATION")
    print("=" * 50)

    # Initialize segmenter for depth processing
    depth_segmenter = HumanSegmentationModel(model_type="deeplabv3")

    # Test 1: Depth Clustering Segmentation
    print("\n--- Testing Depth Clustering Segmentation ---")
    try:
        object_masks, _ = depth_segmenter.depth_clustering_segmentation(
            depth_image, min_cluster_size=150, depth_tolerance=25
        )

        if object_masks:
            print(f"Clustering found {len(object_masks)} objects")
            combined_mask = depth_segmenter.visualize_depth_clustering(
                depth_image, object_masks, save_path="depth_clustering_segmentation.png"
            )

            # Analyze each object
            for i, mask in enumerate(object_masks):
                size = np.sum(mask)
                y_coords, x_coords = np.where(mask > 0)
                centroid = (np.mean(x_coords), np.mean(y_coords))
                print(f"Object {i+1}: Size={size} pixels, Centroid={centroid}")
        else:
            print("No objects detected with clustering")

    except Exception as e:
        print(f"Error with depth clustering: {e}")

    # Test 2: Watershed Segmentation
    print("\n--- Testing Watershed Segmentation ---")
    try:
        object_masks, _ = depth_segmenter.watershed_depth_segmentation(
            depth_image, min_distance=30
        )

        if object_masks:
            print(f"Watershed found {len(object_masks)} objects")
            depth_segmenter.visualize_depth_clustering(
                depth_image, object_masks, save_path="depth_watershed_segmentation.png"
            )
        else:
            print("No objects detected with watershed")

    except Exception as e:
        print(f"Error with watershed: {e}")

    # Test 3: Combined Approach (if RGB is available)
    if rgb_image is not None:
        print("\n--- Testing Combined RGB + Depth Segmentation ---")
        try:
            # Use MaskRCNN for this test
            maskrcnn_segmenter = HumanSegmentationModel(model_type="maskrcnn")

            # RGB-based detection with depth refinement
            mask, original_image, boxes = (
                maskrcnn_segmenter.depth_instance_segmentation(
                    depth_image, rgb_image=rgb_image, confidence_threshold=0.3
                )
            )

            if mask is not None:
                print(f"Combined approach detected objects")
                maskrcnn_segmenter.visualize_results(
                    original_image,
                    mask,
                    save_path="combined_rgb_depth_segmentation.png",
                )
            else:
                print("No objects detected with combined approach")

        except Exception as e:
            print(f"Error with combined approach: {e}")

    # ========================================
    # PART 3: Method Comparison
    # ========================================
    print("\n" + "=" * 50)
    print("PART 3: METHOD COMPARISON")
    print("=" * 50)

    # Test all depth methods using the unified interface
    depth_methods = ["clustering", "watershed"]

    for method in depth_methods:
        print(f"\n--- Testing {method.upper()} via unified interface ---")
        try:
            object_masks, _ = depth_segmenter.segment_objects_depth(
                depth_image,
                method=method,
                min_cluster_size=100,  # For clustering
                min_distance=25,  # For watershed
            )

            if object_masks:
                print(f"{method.capitalize()} method: {len(object_masks)} objects")
                depth_segmenter.visualize_depth_clustering(
                    depth_image,
                    object_masks,
                    save_path=f"unified_{method}_segmentation.png",
                )
            else:
                print(f"No objects detected with {method}")

        except Exception as e:
            print(f"Error with unified {method}: {e}")

    # ========================================
    # PART 4: Performance Analysis
    # ========================================
    print("\n" + "=" * 50)
    print("PART 4: PERFORMANCE ANALYSIS")
    print("=" * 50)

    import time

    methods_to_benchmark = [
        ("clustering", {"min_cluster_size": 100, "depth_tolerance": 20}),
        ("watershed", {"min_distance": 25}),
    ]

    for method_name, params in methods_to_benchmark:
        print(f"\nBenchmarking {method_name}...")
        start_time = time.time()

        try:
            object_masks, _ = depth_segmenter.segment_objects_depth(
                depth_image, method=method_name, **params
            )

            end_time = time.time()
            processing_time = end_time - start_time

            num_objects = len(object_masks) if object_masks else 0
            print(
                f"{method_name.capitalize()}: {processing_time:.3f}s, {num_objects} objects"
            )

        except Exception as e:
            print(f"Benchmark error for {method_name}: {e}")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE! Check the generated PNG files for results.")
    print("=" * 60)


# Test function for depth segmentation specifically
def test_depth_segmentation_only():
    """Focused testing of depth segmentation methods"""
    print("Testing Depth Segmentation Methods Only")
    print("=" * 50)

    # Load depth image
    depth_image_path = "./data/train-000/inner_depths/000101.png"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if depth_image is None:
        print(f"Could not load depth image from {depth_image_path}")
        return

    segmenter = HumanSegmentationModel(model_type="deeplabv3")

    # Test different parameter combinations
    test_configs = [
        ("clustering", {"min_cluster_size": 50, "depth_tolerance": 15}),
        ("clustering", {"min_cluster_size": 200, "depth_tolerance": 30}),
        ("watershed", {"min_distance": 15}),
        ("watershed", {"min_distance": 40}),
    ]

    for i, (method, params) in enumerate(test_configs):
        print(f"\nTest {i+1}: {method} with {params}")

        try:
            object_masks, _ = segmenter.segment_objects_depth(
                depth_image, method=method, **params
            )

            if object_masks:
                print(f"Found {len(object_masks)} objects")
                segmenter.visualize_depth_clustering(
                    depth_image,
                    object_masks,
                    save_path=f"test_{i+1}_{method}_segmentation.png",
                )
            else:
                print("No objects detected")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run comprehensive testing
    main()

    # Uncomment for focused depth testing
    test_depth_segmentation_only()

    # Uncomment for real-time segmentation
    # real_time_segmentation()

    # Uncomment for batch processing
    # batch_process_images("input_images/", "output_masks/", model_type='deeplabv3', show_preprocessing=True)
