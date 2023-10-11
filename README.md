# Song Recommendation by Facial Emotion Detection

**Table of Contents**
- [Introduction](#introduction)
- [Demo](#Demo)
- [Features](#Features)
- [Dependencies](#Dependencies)
- [Project Description](#project-description)
- [Files in the Repository](#files-in-the-repository)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Training](#training)
- [Usage](#usage)
- [Acknowledgments](#Acknowledgments)

## Introduction

This project is aimed at recommending songs based on facial emotion detection. It uses deep learning techniques to classify facial expressions into different emotions and recommends songs that match the detected emotion.

## Demo

You can try the application live at [https://song-recommendation-by-facial-emotion-detection-vp.streamlit.app/](https://song-recommendation-by-facial-emotion-detection-vp.streamlit.app/)

## Features

- **Emotion Detection:** Utilizes computer vision techniques to detect the user's facial emotions.
- **Music Recommendation:** Suggests songs from a pre-defined playlist based on the detected emotion.
- **User Interaction:** Provides a user-friendly interface for uploading images for emotion analysis.
- **Easy to Use:** Simple and intuitive design for a seamless user experience.

## Dependencies

* [![Python][python]][Python-url] : The primary programming language used for development.
* [![OpenCV][opencv]][OpenCV-url] : Used for image processing and facial emotion detection.
* [![TensorFlow][tensorflow]][TensorFlow-url] : Deep learning library for model inference.
* [![Streamlit][Streamlit.io]][Streamlit-url] : A popular Python framework for creating web applications.



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

The project uses the "Fer 2013" dataset, which contains facial expressions and emotions data. You can access the dataset from [Kaggle](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge).


## Data Preprocessing

- Data is loaded, resized.

  ```bash
  for dtype in os.listdir(folder):
    path1 = os.path.join(folder,dtype)
    for expression in os.listdir(path1):
      path2 = os.path.join(path1,expression)
      for x in os.listdir(path2):
        imagepath = os.path.join(path2,x)
        image = cv2.imread(imagepath,0)
        image = image.reshape(48,48,1)
  
        dict['pixels'].append(image)
        dict['labels'].append(expression)
        dict['type'].append(dtype)
  
  df = pd.DataFrame(dict)
  df.head()
  ```

- The dataset is label encoded and split into training, and testing sets.
  ```bash
  from sklearn.preprocessing import LabelEncoder
  le=LabelEncoder()
  
  train_data['labels'] = le.fit_transform(train_data['labels'])
  test_data['labels'] = le.transform(test_data['labels'])
  ```
  ```bash
  x_train = train_data['pixels']
  y_train = train_data['labels']
  x_test = test_data['pixels']
  y_test = test_data['labels']
  ```
- The dataset is normalized.
  ```code
  x_train = x_train/255.0
  x_test = x_test/255.0
  ```

## Model

The deep learning model is built using Keras. It consists of multiple convolutional and fully connected layers.
```code
model = Sequential()

# Conv Block 1
model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 2
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 3
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Conv Block 4
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected Block 1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected Block 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

## Training

- The model is trained for multiple epochs with learning rate reduction.
- Training and validation results are monitored.


```code
reduce_lr = ReduceLROnPlateau(monitor='val_loss' , factor=0.1, patience=2, min_lr=0.00001,model='auto')
```
```code
history = model.fit(x_train,y_train,epochs=100,callbacks= [reduce_lr],validation_data=(x_test,y_test))
```

## Usage

You can use this model for your song recommendation system based on facial emotion detection. Pass an image containing a face, and the model will predict the emotion.

1. Clone the repository to your local machine.
   
   ```bash
   git clone https://github.com/pmohanta2/Song-Recommendation-by-Facial-Emotion-Detection.git
   cd Song-Recommendation-by-Facial-Emotion-Detection
   ```
2. Install the required Python packages listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```
  
3. Run the Streamlit app using the following command:

  ```bash
  streamlit run app.py
  ```

## Acknowledgments

I would like to acknowledge and express my gratitude to the following individuals and resources that contributed to the success of this project:

- **Fer 2013 Dataset**: Our work relies on the Fer 2013 dataset for training and testing the emotion recognition model. You can find the dataset [here](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge).

- **Open Source Community**: A big shoutout to the open-source community for providing tools and libraries like Streamlit, OpenCV, TensorFlow, and many more. Our project wouldn't have been possible without the amazing work of these developers.

- **Prabin** and **Varsha**: I want to express my gratitude to my friend Km Varsha for their exceptional collaboration on this project. Their dedication and hard work significantly contributed to the successful completion of this project.

If you find this project useful or build upon it, please consider acknowledging this project in your work.





<!-- MARKDOWN LINKS & IMAGES -->
[python]:https://img.shields.io/badge/Python-blue?logo=python&logoColor=yellow
[Python-url]:https://www.python.org/
[opencv]:https://img.shields.io/badge/OpenCV-DD0031?logo=opencv&logoColor=black
[OpenCV-url]:https://opencv.org/
[tensorflow]: https://img.shields.io/badge/TensorFlow-95A5A6?style=flat_square&logo=tensorflow&logoColor=black
[TensorFlow-url]:https://www.tensorflow.org/
[Streamlit.io]:https://img.shields.io/badge/Streamlit-green?logo=streamlit&logoColor=black
[Streamlit-url]:https://streamlit.io/
