# 🎯 Celebrity Face Recognition
This project focuses on building an end-to-end Celebrity Face Recognition system that detects and identifies celebrities from images using a combination of Computer Vision, Machine Learning, and FastAPI for deployment.

## 🚀 Project Overview
The goal of this project is to develop a pipeline that can detect celebrity faces from images and identify them accurately, even when images contain multiple faces or are partially obstructed.

## 🧠 Steps Involved
## 1. 🧹 Data Cleaning
Images may contain multiple faces or unclear facial features.

We start by detecting faces and verifying them using eye detection.

Only retain images with at least two detected eyes.

This step helps remove noisy and low-quality data.

## 2. 🔍 Face and Eye Detection
Used OpenCV's Haar Cascade Classifiers to detect faces and eyes.

If the image contains 2 clear eyes, it is considered valid for further processing.

Ensures only high-quality, unobstructed images are used.

## 3. 🌊 Wavelet Transform (Feature Extraction)
Applied Wavelet Transformation to enhance edge features of the face.

Helps in extracting more meaningful information such as:

Eye contours

Nose bridge

Lip edges

Improves model performance by feeding it richer feature sets.

## 4. ✂️ Face Cropping
Final preprocessing pipeline:

Load image

Detect face

Detect eyes

If eyes ≥ 2 → Crop and Save the face region

## 5. 🤖 Model Training (GridSearchCV)
Tried various ML models: SVM, KNN, Random Forest, etc.

Used GridSearchCV to:

Test different model types

Fine-tune hyperparameters

Choose the best model based on accuracy and other metrics

## 6. 🌐 Deployment (FastAPI)
Final model is deployed using FastAPI as a RESTful API.

Lightweight and fast backend ready for frontend or production integration.

Accepts image input and returns the celebrity name prediction.

## ⚙️ Tech Stack
Python

OpenCV

NumPy & Pandas

Scikit-learn

PyWavelets

FastAPI

Jupyter Notebook / VS Code
