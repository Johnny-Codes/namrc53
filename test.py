import json
import os
from PIL import Image, ImageDraw
from helper_definitions import JOINT_NAMES, BONE_CONNECTIONS, INTRINSICS


def create_skeleton_dict(skeleton_coords, joint_names):
    """Converts skeleton coordinates and joint names into a dictionary."""
    float_coords = [list(map(float, coord)) for coord in skeleton_coords]
    return dict(zip(joint_names, float_coords))


def project_3d_to_2d(coords_3d, intrinsics):
    """Projects 3D world coordinates (in mm) to 2D pixel coordinates."""
    x_3d, y_3d, z_3d = coords_3d

    # Avoid division by zero
    if z_3d <= 0:  # Also check for non-positive depth
        return None

    # The projection formula
    x_2d = (intrinsics["fx"] * x_3d / z_3d) + intrinsics["cx"]
    y_2d = (intrinsics["fy"] * y_3d / z_3d) + intrinsics["cy"]

    return (x_2d, y_2d)


# --- Main Program ---
file_path_prefix = "./data"
output_dir = "./output"

# Create the list of all training directories to process
training_file_paths = []
# for i in range(32, 34):  # For folders train-000 to train-032
#     # Use f-string formatting to create zero-padded folder names (e.g., train-001)
#     folder_name = f"train-{i:03d}"
#     training_file_paths.append(os.path.join(file_path_prefix, folder_name))

# Create the main output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
training_file_paths = ["./data/train-033/"]
# Loop through each training directory
for current_file_path in training_file_paths:
    print(f"\n===== Processing Directory: {current_file_path} =====")
    metadata_path = os.path.join(current_file_path, "metadata.jsonl")

    # Check if the metadata file exists before trying to open it
    if not os.path.exists(metadata_path):
        print(
            f"Warning: Metadata file not found at {metadata_path}. Skipping directory."
        )
        continue

    try:
        with open(metadata_path, "r") as f:
            # Read and process all lines in the metadata file
            lines = f.readlines()
            for i, line in enumerate(lines):
                print(
                    f"--- Processing image {i+1}/{len(lines)} in {os.path.basename(current_file_path)} ---"
                )
                json_data = json.loads(line.strip())

                # Get the action label to create a subfolder
                action_label = json_data.get("action_label", "unknown_action")

                # Get the original subfolder (e.g., "inner_depths" or "outer_depths")
                image_file_name = json_data["file_name"]
                depth_subfolder = os.path.dirname(image_file_name)

                # Create the full output path including the depth subfolder
                full_output_dir = os.path.join(
                    output_dir, action_label, depth_subfolder
                )
                os.makedirs(full_output_dir, exist_ok=True)

                image_file_path = os.path.join(current_file_path, image_file_name)
                skeleton_3d_coords = json_data["skeleton"]

                # Create a structured dictionary of the 3D skeleton data
                skeleton_3d = create_skeleton_dict(skeleton_3d_coords, JOINT_NAMES)

                image = Image.open(image_file_path)
                draw = ImageDraw.Draw(image)
                image_width, image_height = image.size

                # --- Project 3D points to 2D and store them ---
                projected_skeleton = {}
                for joint_name, coords_3d in skeleton_3d.items():
                    coords_2d = project_3d_to_2d(coords_3d, INTRINSICS)
                    if coords_2d:
                        projected_skeleton[joint_name] = coords_2d

                # --- Draw the bones (lines) ---
                for joint1_name, joint2_name in BONE_CONNECTIONS:
                    if (
                        joint1_name in projected_skeleton
                        and joint2_name in projected_skeleton
                    ):
                        p1 = projected_skeleton[joint1_name]
                        p2 = projected_skeleton[joint2_name]
                        draw.line([p1, p2], fill="lime", width=3)

                # --- Draw the joints (circles) on top of the bones ---
                for joint_name, coords_2d in projected_skeleton.items():
                    x, y = coords_2d
                    if 0 <= x < image_width and 0 <= y < image_height:
                        draw.ellipse(
                            (x - 4, y - 4, x + 4, y + 4), fill="red", outline="red"
                        )

                # Save the image into the action-specific and depth-specific subfolder
                output_filename = os.path.basename(image_file_name)
                output_path = os.path.join(
                    full_output_dir, f"skeleton_{output_filename}"
                )
                image.save(output_path)

                # Append the original metadata to the new metadata.jsonl in the action subfolder
                metadata_output_path = os.path.join(full_output_dir, "metadata.jsonl")
                with open(metadata_output_path, "a") as meta_f:
                    meta_f.write(line)

    except Exception as e:
        print(f"An error occurred while processing {current_file_path}: {e}")

print(
    "\n\nAll directories processed. Check the 'output' directory for the images and metadata."
)
