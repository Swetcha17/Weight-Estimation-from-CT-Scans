import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Deep Learning: 3D CNN
class WeightEstimationCNN(nn.Module):
    def __init__(self, input_depth=64, input_height=128, input_width=128):
        super(WeightEstimationCNN, self).__init__()
        
        # 4-Layer 3D CNN
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: 32 x D/2 x H/2 x W/2
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: 64 x D/4 x H/4 x W/4
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: 128 x D/8 x H/8 x W/8
            
            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Output: 256 x D/16 x H/16 x W/16
        )
        
        # Calculate flat features size
        self.flat_size = 256 * (input_depth // 16) * (input_height // 16) * (input_width // 16)
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Final output: Weight (kg)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# Traditional ML Wrappers
class TraditionalMLModel:
    def __init__(self, model_type="rf"):
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                n_jobs=-1, 
                random_state=42
            )
        elif model_type == "xgb":
            self.model = xgb.XGBRegressor(
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=6, 
                objective='reg:squarederror',
                n_jobs=-1
            )
            
    def fit(self, X, y):
        # Flatten 3D input to 2D (samples, features)
        X_flat = X.reshape(X.shape[0], -1)
        print(f"Training {self.model_type.upper()} on shape: {X_flat.shape}")
        self.model.fit(X_flat, y)
        
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)