<h1 align="center">🎵 Facial Expression–Based Song Recommendation System</h1>
<p align="center">
  An emotion-aware music recommendation system using Deep Learning and Computer Vision
</p>

---

## 📌 Overview

This project presents a **Facial Expression–Based Song Recommendation System** that analyzes a user's facial expression and recommends an appropriate set of songs based on the detected emotion.

The system takes **facial images as input**, identifies the underlying emotion using a **trained deep learning model**, and maps the recognized emotion to a **curated playlist**.

The goal of this project is to create a **personalized and emotion-aware music recommendation experience** using computer vision and deep learning techniques.

---

## 📂 Dataset Description (`aml_notes`)

- 📸 Contains **3000+ facial expression images**
- 😃 Represents **7 different emotion classes**
- 🌐 Dataset sourced from **Kaggle**
- 🧠 Used as the foundation for **training and validation** of the model

---

## 🧠 Model Training (`emotiondetector.ipynb`)

This notebook is responsible for **training the facial expression recognition model**.

### 🔹 Key Highlights

- Implemented a **Convolutional Neural Network (CNN)** using the **Keras Sequential API**
- Architecture includes:
  - Convolutional layers with **ReLU activation**
  - **MaxPooling** layers to reduce spatial dimensions
  - **Dropout** layers to prevent overfitting
  - **Flatten** layers to convert feature maps into vectors
  - **Dense (fully connected)** layers for classification
- The model classifies facial expressions into **7 emotion categories**
- Trained model is saved and used for **real-time emotion prediction**

---

## 🎥 Real-Time Emotion Detection & Song Recommendation  
### (`RealTimeFaceDetection3.ipynb`)

This notebook enables **real-time emotion recognition and song recommendation**.

### 🔹 Key Features

- 📷 Uses **OpenCV (`cv2`)** for real-time face detection via webcam
- 🧠 Facial frames are processed and passed to the trained CNN model
- 🎵 Uses a **Python dictionary** to map emotions to song playlists
- ⚡ Instantly recommends songs based on detected facial expression

---

## 🛠️ Technologies Used

- 🐍 Python  
- 🧠 TensorFlow / Keras  
- 📷 OpenCV  
- 🤖 Convolutional Neural Networks (CNN)  
- 🌐 Kaggle Dataset  

---

<p align="center">
  🚀 Built to explore the intersection of emotions, AI, and music
</p>
