# Song Recommendation by Facial Emotion Detection

**Table of Contents**
- [Introduction](#introduction)
- [Project Description](#project-description)
- [Files in the Repository](#files-in-the-repository)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Training](#training)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is aimed at recommending songs based on facial emotion detection. It uses deep learning techniques to classify facial expressions into different emotions and recommends songs that match the detected emotion.

## Project Description

- [x] Data collection: Collecting a dataset of facial expressions labeled with corresponding emotions.
- [x] Data preprocessing: Preprocessing the data, including resizing, normalization, and label encoding.
- [x] Model: Creating a Convolutional Neural Network (CNN) for emotion detection.
- [x] Training: Training the model using the preprocessed data.
- [x] Usage: Using the trained model to detect emotions and recommend songs.

## Files in the Repository

- `final_model.ipynb`: Jupyter Notebook containing the code for data preprocessing, model creation, training, and evaluation.
- `app.py`: The Streamlit web application.
- `model.h5`: Trained model for facial emotion detection.
- `haarcascade_frontalface_default.xml`: Haar Cascade file for face detection.
- `requirements.txt`: List of required Python packages.

## Data Collection

The dataset used in this project is stored in the `drive/MyDrive/project/data` directory. It contains images of faces with labeled emotions.

## Data Preprocessing

- Data is loaded, resized, normalized, and label encoded.
- The dataset is split into training and testing sets.

## Model

The deep learning model is built using Keras. It consists of multiple convolutional and fully connected layers.

## Training

- The model is trained for multiple epochs with learning rate reduction.
- Training and validation results are monitored.

## Usage

You can use this model for your song recommendation system based on facial emotion detection. Pass an image containing a face, and the model will predict the emotion.

```python
# Example code for emotion detection
# Load and preprocess the image, and then use the model to predict the emotion
