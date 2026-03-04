import cv2
import numpy as np
import joblib

# 1. Load the trained brain (model)
model = joblib.load('models/svm_model.pkl')

# 2. Load and prepare your test image
img_path = 'data/raw/test/test_image.jpg'
img = cv2.imread(img_path)

if img is not None:
    img_resized = cv2.resize(img, (64, 64))
    img_flattened = img_resized.flatten().reshape(1, -1)

    # 3. Ask the AI for its guess
    prediction = model.predict(img_flattened)
    
    result = "DOG" if prediction[0] == 1 else "CAT"
    print(f"The AI thinks this is a: {result}")
else:
    print("Error: Could not find test_image.jpg in data/raw/test/")