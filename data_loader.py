import os
import numpy as np
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CTScanDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D CT Volumes and Weight Labels.
    """
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load the 3D volume
        volume_path = self.data_paths[idx]
        volume = load_dicom_volume(volume_path)
        
        # Preprocess
        volume = preprocess_volume(volume)
        
        # Add channel dimension for PyTorch (C, D, H, W) -> (1, Depth, Height, Width)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return volume, label

def load_dicom_volume(path_to_dicom_folder):
    """
    Reads a directory of DICOM files, sorts them by slice location, 
    and returns a 3D numpy array.
    """
    if not os.path.exists(path_to_dicom_folder):
        raise FileNotFoundError(f"Directory not found: {path_to_dicom_folder}")

    # Read all dicom files
    slices = []
    for s in os.listdir(path_to_dicom_folder):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path_to_dicom_folder, s))
                slices.append(ds)
            except:
                continue
                
    # Sort slices by ImagePositionPatient Z-axis to ensure correct 3D order
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Stack into 3D array
    # Handle varying pixel spacing if necessary (resampling), but here we assume uniformity
    try:
        volume = np.stack([s.pixel_array for s in slices])
    except ValueError:
        # Fallback if slice dimensions mismatch
        print(f"Warning: Slice mismatch in {path_to_dicom_folder}. Resizing to first slice shape.")
        ref_shape = slices[0].pixel_array.shape
        volume = np.stack([cv2.resize(s.pixel_array, ref_shape) for s in slices])
        
    return volume

def preprocess_volume(volume, target_depth=64, target_size=(128, 128)):
    """
    Standard CT preprocessing:
    1. Hounsfield Unit (HU) Windowing (Soft Tissue window)
    2. Normalization to [0, 1]
    3. Resizing to standard input shape
    """
    # Soft tissue windowing (-150 to 250 HU is standard for body/torso)
    MIN_BOUND = -150.0
    MAX_BOUND = 250.0
    
    volume = (volume - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    volume[volume > 1] = 1.
    volume[volume < 0] = 0.
    
    # Resize spatial dimensions (H, W)
    resized_slices = [cv2.resize(slice, target_size) for slice in volume]
    volume = np.array(resized_slices)
    
    # Resize depth dimension (Interpolation)
    # We use simple sampling for this script to avoid heavy scipy dependencies
    current_depth = volume.shape[0]
    if current_depth != target_depth:
        indices = np.linspace(0, current_depth - 1, target_depth).astype(int)
        volume = volume[indices]
        
    return volume

def get_data_loaders(data_dir, labels_dict, batch_size=4, test_size=0.2):
    """
    Splits data and returns PyTorch DataLoaders.
    Expects labels_dict to be { 'folder_name': weight_value }
    """
    all_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    X = []
    y = []
    
    # Match folders to labels
    for folder_path in all_folders:
        folder_name = os.path.basename(folder_path)
        if folder_name in labels_dict:
            X.append(folder_path)
            y.append(labels_dict[folder_name])
            
    if not X:
        raise ValueError("No matching data found. Check your DATA_DIR and labels keys.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    train_dataset = CTScanDataset(X_train, y_train)
    test_dataset = CTScanDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader