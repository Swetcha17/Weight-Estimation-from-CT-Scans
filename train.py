import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from data_loader import get_data_loaders
from models import WeightEstimationCNN, TraditionalMLModel

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
DATA_DIR = "./dataset/ct_scans"     # Path to folder containing patient folders
LABELS_FILE = "./dataset/labels.csv" # CSV with columns: ['PatientID', 'Weight']
MODEL_TYPE = "3DCNN"                # Options: '3DCNN', 'RF' (RandomForest), 'XGB' (XGBoost)
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_labels(csv_path):
    """
    Reads the CSV and creates a dictionary {PatientID: Weight}
    """
    df = pd.read_csv(csv_path)
    # Ensure columns match your CSV structure
    return pd.Series(df.Weight.values, index=df.PatientID).to_dict()

def train_deep_learning(train_loader, test_loader):
    print(f"Initializing 3D CNN on {DEVICE}...")
    model = WeightEstimationCNN().to(DEVICE)
    criterion = nn.L1Loss() # MAE Loss is robust for weight estimation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.numpy().flatten())
        
        mae = mean_absolute_error(val_targets, val_preds)
        r2 = r2_score(val_targets, val_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val MAE: {mae:.2f} kg | R2: {r2:.4f}")
        
        # Save best model
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "best_weight_estimator.pth")
            print("--> Model Saved!")

def train_traditional_ml(train_loader, test_loader, model_type):
    # For Traditional ML, we need to load all data into memory first
    # This is fine for small datasets, but for large ones, you'd need batched incremental learning
    print("Loading all data into memory for traditional ML...")
    
    X_train, y_train = [], []
    for inputs, targets in train_loader:
        X_train.append(inputs.numpy())
        y_train.append(targets.numpy())
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_test, y_test = [], []
    for inputs, targets in test_loader:
        X_test.append(inputs.numpy())
        y_test.append(targets.numpy())
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    model = TraditionalMLModel(model_type)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Final {model_type.upper()} MAE: {mae:.2f} kg")

if __name__ == "__main__":
    # 1. Load Labels
    try:
        labels_map = load_labels(LABELS_FILE)
    except FileNotFoundError:
        print("Error: Labels file not found. Please create a dummy CSV to test.")
        exit()

    # 2. Prepare DataLoaders
    try:
        train_loader, test_loader = get_data_loaders(DATA_DIR, labels_map, BATCH_SIZE)
    except Exception as e:
        print(f"Data Loading Error: {e}")
        exit()

    # 3. Train
    if MODEL_TYPE == "3DCNN":
        train_deep_learning(train_loader, test_loader)
    else:
        train_traditional_ml(train_loader, test_loader, MODEL_TYPE)