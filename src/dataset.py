import os
from glob import glob
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from . import config


def zero_pad_images(image_data: list) -> list:
    """
    Zero pads all the 3D numpy arrays in the list to have the same dimensions.

    Args:
        image_data (list): A list of 3D numpy arrays representing images with varying sizes.

    Returns:
        list: A list of zero-padded 3D numpy arrays with consistent dimensions.
    """
    # Determine the maximum dimensions across all images
    max_dims = np.max([image.shape[:2] for image in image_data], axis=0).tolist()

    # Zero pad each image in the list to have dimensions of max_dims
    padded_images = []
    for image in image_data:
        pad_x = max_dims[0] - image.shape[0]
        pad_y = max_dims[1] - image.shape[1]

        # Pad the image and add it to the list of padded images
        padded_image = np.pad(
            image, ((0, pad_x), (0, pad_y), (0, 0)), "constant", constant_values=0
        )
        padded_images.append(padded_image)

    return padded_images


patches_folder_path = "./STAN_patches_lbls"


# Load labels from CSV file
labels_folder_path = os.path.join(config.DATA_DIR, "labels")
labels_df = pd.read_csv(os.path.join(labels_folder_path, "STAN_labels.csv"))
labels_dict = dict(zip(labels_df["UID"], labels_df["MUT_STATUS"]))
labels_dict


image_data = []
labels = []

# Read image slices and stack them as channels
for subject_folder in glob(os.path.join(patches_folder_path, "patches", "SUB_*")):
    subject_id = os.path.basename(subject_folder)
    if subject_id in labels_dict:
        # Sort the files to ensure slices are in the correct order
        slice_files = sorted(glob(os.path.join(subject_folder, "*.png")))
        if len(slice_files) == 5:  # Ensure we have exactly 5 slices
            slices = [np.array(Image.open(f)) for f in slice_files]
            # Stack slices along the third dimension to form a 5-channel image
            stacked_slices = np.stack(slices, axis=-1)
            image_data.append(stacked_slices)
            labels.append(labels_dict[subject_id])

# Zero pad images to have consistent dimensions
image_data = zero_pad_images(image_data)

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Normalize the image data to [0, 1] range
image_data = image_data.astype(np.float32) / 255.0  

# Display shapes to confirm data loading
image_data.shape, labels.shape

# Split the data into training, validation, and test sets using stratified sampling
# Use 70:15:15 split for train:val:test
X_train, X_temp, y_train, y_temp = train_test_split(
    image_data, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Display the shapes of the resulting sets
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape

# Convert numpy arrays to PyTorch tensors
# Needs to be (N, 1, D, H, W)
X_train_torch = (
    torch.tensor(X_train).permute(0, 3, 1, 2).unsqueeze(1)
) 
X_val_torch = torch.tensor(X_val).permute(0, 3, 1, 2).unsqueeze(1)
X_test_torch = torch.tensor(X_test).permute(0, 3, 1, 2).unsqueeze(1)

y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_val_torch = torch.tensor(y_val, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader objects
train_data = TensorDataset(X_train_torch, y_train_torch)
val_data = TensorDataset(X_val_torch, y_val_torch)
test_data = TensorDataset(X_test_torch, y_test_torch)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_data, batch_size=32, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_data, batch_size=32, num_workers=config.NUM_WORKERS)