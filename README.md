# Embodied-Conversational-Agent
Affective Computing (CMPT 419 / 983) at Simon Fraser University.

This project contains the following files:

1. real-time-app.py   (contains the actual program to be run)
2. AudioEmotionConvNet.ipynb  (Jupyter notebook consisting of building and training of machine learning model for Audio Emotion Recognition)
3. Video_Recognition.ipynb (Jupyter notebook consisting of building and training of machine learning model for Video Emotion Recognition)
4. control-module.py  (contains the script for the visuals to be run) (but we did not succeed in fusing it with the real-time app)
5. Speech_Emotion_Recognition125.h5 (file for the Audio Emotion machine learning model)
6. fer_model_59.h5  (file for Video Emotion machine learning model)
7. haarcascade_frontalface_default.xml  (file for facial recognition)

This program would need some python libraries installed. Namely, numpy, open-cv, tensorflow, pyaudio, speech_recognition, librosa, time, threading, keyboard.
Also, SmartBody needs to be installed for the visuals.
(https://sourceforge.net/projects/smartbody/)


Overview:

This project aims to build a Socially Aware Agent which predicts the emotion using audio and video input from the user and generating affect bursts similar to the
user's emotion. We built machine learning models for both Audio Emotion recognition (using PyAudio, Speech_Recognition, keras and sci-kit learn) and Video 
Emotion Recognition (using OpenCV). Combining Audio and Video ML models, we made a Fusion Model, which predicts the emotion using both the ML models.

We initially planned to use Unity 3D, but later shifted to SmartBody.
However, we succeeded in creating the Fusion Model which predicts emotions using Audio and Video input, but 
could not implement the Agent's visual part for the same. This was due to some late observed difficulties
in the way how SmartBody functions.
