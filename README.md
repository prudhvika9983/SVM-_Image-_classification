# 🐾 Cat vs. Dog Image Classifier (SVM)

A machine learning project that uses a **Support Vector Machine (SVM)** to classify images of cats and dogs from the kaggle dataset

## 🚀 Project Overview
This project implements a complete machine learning pipeline:
* **Preprocessing**: Resizing images to 64x64 and flattening them into feature vectors.
* **Training**: Training an SVM model using 200 labeled images (100 cats / 100 dogs).
* **Prediction**: A script to test the model on new, unseen images.

## 📂 Folder Structure
* `data/raw/`: Contains the original training and test images.
* `data/processed/`: Binary `.npy` files containing processed numerical data.
* `models/`: The saved `svm_model.pkl` trained brain.
* `src/`: Python scripts for preprocessing, training, and evaluation.

## 📊 Results
The model successfully identifies pets in images, providing a classification output in the terminal.
