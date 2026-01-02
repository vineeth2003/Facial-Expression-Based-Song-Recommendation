Facial Expression–Based Song Recommendation System

This project presents a Facial Expression–Based Song Recommendation System that analyzes a user’s facial expression and recommends an appropriate set of songs based on the detected emotion. The system takes facial images as input, identifies the underlying emotion using a trained deep learning model, and maps the recognized emotion to a curated playlist.

The goal of this project is to create a personalized and emotion-aware music recommendation experience using computer vision and deep learning techniques.

Dataset Description (aml_notes)

The aml_notes file contains the facial expression image dataset used for training the model.

The dataset consists of more than 3000 images representing various facial expressions.

These images were collected from Kaggle and are categorized into 7 different emotion classes.

The dataset serves as the foundation for training and validating the emotion classification model.

Model Training (emotiondetector.ipynb)

The emotiondetector.ipynb notebook is responsible for training the facial expression recognition model.

Key Highlights:

A Convolutional Neural Network (CNN) is implemented using the Keras Sequential API.

The network architecture includes:

Convolutional layers with ReLU activation for feature extraction

MaxPooling layers to reduce spatial dimensions

Dropout layers to prevent overfitting

Flatten layers to convert feature maps into vectors

Dense (fully connected) layers for final classification

The model is trained to classify facial expressions into 7 emotion categories.

After training, the model is saved and used for real-time emotion prediction.

Real-Time Emotion Detection & Song Recommendation (RealTimeFaceDetection3.ipynb)

The RealTimeFaceDetection3.ipynb notebook enables real-time emotion recognition and song recommendation.

Key Features:

Uses OpenCV (cv2) for real-time face detection via a webcam.

Captured facial frames are processed and passed to the trained CNN model for emotion prediction.

A Python dictionary is used to map each detected emotion to a predefined list of songs.

Based on the recognized facial expression, the system recommends a suitable set of songs instantly.

Technologies Used

Python

TensorFlow / Keras

OpenCV

Convolutional Neural Networks (CNN)

Kaggle Dataset
