This is a Facial Expression Based Song Recommendation System in which it takes facial expressions as input and as output, it it recognises the facial expression and recommends a set of songs based on it.
For this project we have used images of facial expression as dataset which are more than 3000 in number. We got this dataset through kaggle.
Let's learn each file in detail.
1) aml_notes:- This file contains the dataset of images which is mentioned earlier from which we are going to train our model.
2) emotiondetector.ipynb:- In this file we are training the model. For this we have used Convolutional Neural Network (CNN) using the Sequential API from Keras.
                           The network consists of several convolutional layers followed by fully connected (dense) layers, and it is designed to classify images into 7 categories.
                           For training the model we have used the concept of ReLu, MaxPooling, Dropout, Dense layers, Flatten layers.
3) RealTimeFaceDetection3.ipynb:- This file helps to recognise and recommend the songs in real time. For this we have imported cv2 which is OpenCV library used for real-time computer vision.
                                  We have used the concept of python dictionaries for mapping each emotion to a list of songs.
