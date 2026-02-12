# Weight-Estimation-from-CT-Scans
This repository contains the implementation of a machine learning pipeline that estimates patient body weight directly from CT scan DICOM volumes by combining tissue-aware feature engineering with modern deep learning methods. This system is designed to aid in automated dosage calculation for emergency medicine where patient weighing is not feasible.

## Project Overview

* **Objective**: Predict body weight (kg) from chest/abdomen CT scans.
* **Dataset**: 3D DICOM volumes (Hounsfield Unit normalized).
* **Models Implemented**:
    * **3D CNN**: A custom 4-layer 3D Convolutional Network using PyTorch.
    * **XGBoost / Random Forest**: Baseline regressors using flattened volumetric features.

## Repository Structure

* `data_loader.py`: Handles DICOM ingestion, slice sorting, HU windowing (-150 to 250), and resizing.
* `models.py`: Contains the PyTorch `WeightEstimationCNN` class and Scikit-Learn wrappers.
* `train.py`: Main training script with validation loops and model checkpointing.

## Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install torch torchvision numpy pandas pydicom opencv-python scikit-learn xgboost
    ```

2.  **Data Preparation**
    * Place your patient folders (containing `.dcm` files) in `dataset/ct_scans`.
    * Create a `labels.csv` file in `dataset/` with columns `PatientID` (folder name) and `Weight` (kg).

3.  **Configuration**
    * Open `train.py` and update `DATA_DIR` and `LABELS_FILE` paths if necessary.
    * Set `MODEL_TYPE` to `"3DCNN"`, `"XGB"`, or `"RF"`.

## Usage

Run the training pipeline:
```bash
python train.py
