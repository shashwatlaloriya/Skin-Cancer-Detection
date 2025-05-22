# Skin Cancer Detection

## Overview
This project leverages Convolutional Neural Networks (CNNs) and advanced image processing techniques to create an automated Skin Cancer Detector. <br>The system is designed to classify skin lesion images as cancerous or non-cancerous, aiming to support early and accurate diagnosis of skin cancer, which is critical for patient outcomes.

## Motivation
Skin cancer is among the most common and potentially deadly forms of cancer. 
<br>
Early detection dramatically improves survival rates, but manual diagnosis is time-consuming and subject to human error. 
<br>
This project aims to provide a reliable, automated tool that assists dermatologists by analyzing dermoscopic images and predicting the likelihood of malignancy, thereby reducing diagnostic delays and increasing accuracy.

## Dataset
Source: Publicly available datasets such as HAM10000 and ISIC archive, containing thousands of labeled dermoscopic images of various skin lesions.
<br><br>
Classes: The dataset includes images categorized as benign (non-cancerous) and malignant (cancerous), covering multiple types of skin lesions including melanoma, basal cell carcinoma, and others.
<br><br>
Preprocessing: Images are resized, normalized, and augmented (rotation, flipping, scaling) to enhance model robustness and prevent overfitting.
<br>
## Methodology
1. Image Preprocessing: Images are cleaned, resized, and normalized. Data augmentation techniques are applied to increase dataset diversity and improve model generalization.

2. Model Architecture: A Convolutional Neural Network (CNN) is designed and trained to extract features and classify images.
Transfer learning with pre-trained models (e.g., ResNet, VGG, Inception) is optionally used to boost performance.

3. Training and Evaluation: The model is trained on the prepared dataset with categorical cross-entropy loss and accuracy as the primary metric. Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix on a held-out test set.


4. Prediction: The trained model predicts whether a given skin lesion image is benign or malignant. Results are visualized for user interpretation.

## Results
The CNN-based model achieves high accuracy in distinguishing malignant from benign lesions, with performance metrics comparable to or exceeding those reported in recent literature (e.g., 90–97% accuracy on benchmark datasets).
The system demonstrates robustness across different skin lesion types and imaging conditions.
