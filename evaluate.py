import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the trained brain and the data
model = joblib.load('models/svm_model.pkl')
X = np.load('data/processed/X_train.npy')
y = np.load('data/processed/y_train.npy')

# 2. Let the AI predict everything it learned
y_pred = model.predict(X)

# 3. Print the "Report Card"
print("--- AI PERFORMANCE REPORT ---")
print(classification_report(y, y_pred, target_names=['Cat', 'Dog']))