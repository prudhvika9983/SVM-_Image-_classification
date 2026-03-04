import os
import cv2
import numpy as np

def load_images(folder, label):
    images = []
    labels = []
    file_count = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)
            file_count += 1
    print(f"Found {file_count} images in {folder}") # This tells us the truth!
    return images, labels

# Paths to your folders
cat_path = r'C:\SVM_Image_Classifier\data\raw\train\cats'
dog_path = r'C:\SVM_Image_Classifier\data\raw\train\dogs'

print("Processing images...")
cat_imgs, cat_labels = load_images(cat_path, 0) # 0 for cats
dog_imgs, dog_labels = load_images(dog_path, 1) # 1 for dogs

X = np.array(cat_imgs + dog_imgs)
y = np.array(cat_labels + dog_labels)

# Save the processed data
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', X)
np.save('data/processed/y_train.npy', y)
print("Done! Data saved in data/processed")