<h1 align="center">Facial Expression–Based Song Recommendation System</h1>
<p align="center">
  Emotion-aware music recommendation using Deep Learning and Computer Vision
</p>

---

## Overview

This project implements a **Facial Expression–Based Song Recommendation System** that analyzes a user's facial expression and recommends an appropriate set of songs based on the detected emotion.

The system takes facial images as input, identifies the underlying emotion using a trained deep learning model, and maps the recognized emotion to a curated playlist. The objective is to provide a **personalized and context-aware music recommendation experience** by leveraging computer vision and deep learning techniques.

---

## Dataset Description (`aml_notes`)

- Contains over **3,000 facial expression images**
- Covers **seven distinct emotion categories**
- Dataset sourced from **Kaggle**
- Used for training and validating the emotion classification model

---

## Model Training (`emotiondetector.ipynb`)

This notebook is responsible for training the **facial expression recognition model**.

### Model Details

- Implemented using a **Convolutional Neural Network (CNN)** with the **Keras Sequential API**
- Network architecture consists of:
  - Convolutional layers with **ReLU activation** for feature extraction
  - **MaxPooling layers** for spatial dimensionality reduction
  - **Dropout layers** to mitigate overfitting
  - **Flatten layers** to transform feature maps into vectors
  - **Dense (fully connected) layers** for final classification
- The model is trained to classify facial expressions into **seven emotion classes**
- The trained model is saved and utilized for real-time emotion prediction

---

## Real-Time Emotion Detection and Song Recommendation  
### (`RealTimeFaceDetection3.ipynb`)

This notebook enables **real-time facial expression recognition and music recommendation**.

### Key Features

- Uses **OpenCV (`cv2`)** for real-time face detection through a webcam
- Captured facial frames are preprocessed and passed to the trained CNN model
- A **Python dictionary-based mapping** is used to associate emotions with predefined song playlists
- Based on the detected emotion, a relevant set of songs is recommended in real time

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- Convolutional Neural Networks (CNN)  
- Kaggle Dataset  

---

<p align="center">
  Built to explore the application of artificial intelligence in emotion-aware music recommendation systems
</p>
