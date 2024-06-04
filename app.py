#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

def about():
    st.write("A user’s emotion or mood can be detected by his/her facial expressions. A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various human emotions or moods. Machine Learning provides various techniques through which human emotions can be detected.")
    st.write("Music is a great connector. It unites us across markets, ages, backgrounds, languages, preferences, political leanings, and income levels. People often use music as a means of mood regulation, specifically to change a bad mood, increase energy levels or reduce tension. Also, listening to the right kind of music at the right time may improve mental health. Thus, human emotions have a strong relationship with music.")
    st.write("The objective of this is to analyze the user's image, predict the expression of the user and suggest songs suitable to the detected mood.")

def dev():
    st.write('### Km Varsha (dishanipandey311019@gmail.com)')
    st.write('### Prabin Kumar Mohanta (prabinkumarmohanta8@gmail.com)')

class EmotionDetectionProcessor(VideoProcessorBase):
    def __init__(self, model, emotions):
        self.model = model
        self.emotions = emotions
        self.haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        faces = self.haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) / 255.0
            gray = gray.reshape(1, 48, 48, 1)

            predicts = self.model.predict(gray)[0]
            label = self.emotions[predicts.argmax()]
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Recommend songs by detecting facial emotion")

    activities = ["Home", "About", "Developer"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    model = keras.models.load_model('model.h5')

    if choice == "Home":
        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # Mode selection
        mode = st.radio("Choose input mode", ("Image", "Live Video"))

        if mode == "Image":
            # Image upload
            image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
            if image_file is not None:
                img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                st.image(img)
                
                if st.button("Process"):
                    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    faces = haar_cascade.detectMultiScale(img)
                    if len(faces) == 0:
                        st.write("### No Face Detected! Please reupload the image.")
                    elif len(faces) > 1:
                        st.write("### Multiple Faces Detected! Please reupload the image.")
                    else:
                        face = faces[0]
                        x, y, w, h = face
                        frame = cv2.resize(img[y:y+h, x:x+w], (48, 48), interpolation=cv2.INTER_AREA)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
                        gray = gray.reshape(1, 48, 48, 1)

                        predicts = model.predict(gray)[0]
                        label = EMOTIONS[predicts.argmax()]
                        st.write('Detected emotion is', label)
                        
                        recommend_music(label)

        elif mode == "Live Video":
            st.write("Starting live video...")
            webrtc_ctx = webrtc_streamer(
                key="emotion-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: EmotionDetectionProcessor(model, EMOTIONS),
                async_processing=True
            )

    elif choice == "About":
        about()
        
    elif choice == "Developer":
        dev()

def recommend_music(emotion):
    if emotion == 'angry':
        st.video("https://www.youtube.com/watch?v=vIs4pNy8hzI")
    elif emotion == 'disgust':
        st.video("https://www.youtube.com/watch?v=KwiDJclWo44")
    elif emotion == 'fear':
        st.video("https://www.youtube.com/watch?v=Kjyr9JYd3-I")
    elif emotion == 'happy':
        st.video("https://www.youtube.com/watch?v=IwSZzuvevyc")
    elif emotion == 'neutral':
        st.video("https://www.youtube.com/watch?v=EzPg8-285YI")
    elif emotion == 'sad':
        st.video("https://www.youtube.com/watch?v=284Ov7ysmfA")
    elif emotion == 'surprise':
        st.video("https://www.youtube.com/watch?v=zlt38OOqwDc")

if __name__ == "__main__":
    main()
