import numpy as np
from sklearn.svm import SVC
import joblib
import os

# 1. Load the processed data
try:
    X = np.load('data/processed/X_train.npy')
    y = np.load('data/processed/y_train.npy')
    
    # Check if data is empty
    if X.shape[0] == 0:
        print("Error: No data found. Please check your data/raw folders.")
    else:
        # 2. Fix the shape if it's 1D (This fixes your error)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        print(f"Training SVM with {len(X)} images...")
        model = SVC(kernel='linear')
        model.fit(X, y)

        # 3. Save the model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/svm_model.pkl')
        print("Training Complete! Model saved in models/svm_model.pkl")

except FileNotFoundError:
    print("Error: Could not find .npy files. Run preprocess.py first.")