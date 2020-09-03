# Importing necessary libraries
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import pyaudio
import speech_recognition as sr
import librosa
import time
import threading
import keyboard
import sys

# Initializing global variables needed to run in real-time
global pred_fer         # Array of probabilities for emotion predictions from video
global pred_aud         # Array of probabilities for emotion predictions from audio
global start_time_vid   # Time of start of video recording
global start_time_aud   # Time of start of audio recording


# Function to extract audio features (MFCCs) from the wav audio files
def extract_features(filename):

  X, samp_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc_val = np.mean(librosa.feature.mfcc(y=X, sr=samp_rate, n_mfcc=40), axis=1)
  mfcc_n = np.asarray(mfcc_val)
  mfcc_n = mfcc_n.reshape((1,40))
  mfcc_n = np.expand_dims(mfcc_n, axis=2)

  return mfcc_n


# Function to record audio from the user
def aud_rec():
    rec = sr.Recognizer()

    # Taking audio input from the user
    with sr.Microphone() as source:
        print('Say something')
        audio = rec.listen(source)

    # To create a wav file of the recorded audio
    with open('recaudio.wav', 'wb') as fil:
        fil.write(audio.get_wav_data())


# Model for Audio Emotion Recognition
model = keras.models.load_model('./Speech_Emotion_Recognition125.h5')


# Model for Video Emotion Recognition
# Emotion Classifier
model_fer = tf.keras.models.load_model("./fer_model_59.h5")

# Face Classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Labels
# For Video: FER+ Dataset
types_fer = ("Fear", "Disgust", "Anger", "Happiness", "Sadness", "Surprise", "Neutral")

# For Audio: RAVDESS Dataset
types_aud = ('Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised')



# Function for Real-Time Video Analysis using OpenCV
def realtime_video():

    # Stating the global variables to be used
    global start_time_aud
    global start_time_vid
    global pred_fer
    global pred_aud

    pred_fer = [None]*8         # Initializing the array for video to NONE
    start_time_vid = None       # Initializing the video start time to NONE

    capture = cv2.VideoCapture(0)

    try:
        while True:
            
            
            ret_value, image = capture.read()

            if not capture:
                continue

            faces_detected = classifier.detectMultiScale(image, 1.32, 5)

            for (x,y,w,h) in faces_detected:

                # Cropping region of interest i.e. face area from image
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), thickness=7)
                roi=image[y:y+w,x:x+h]
                roi=cv2.resize(roi,(48, 48))

                img_pixels = tf.keras.preprocessing.image.img_to_array(roi)
                img_pixels = np.expand_dims(img_pixels, axis = -1)
                img_pixels = img_pixels.reshape((-1, 48, 48, 1))
                
                # Predict emotions using trained model for VIDEO using FER+ Dataset
                pred_fer = model_fer.predict(img_pixels)

                # Record time for video prediction
                start_time_vid = time.time()

                # Predictions on FER+ Dataset
                # max_index_fer = np.argmax(pred_fer[0])
                # predicted_emotion_fer = types_fer[max_index_fer]

                # print("FER+:")
                # for idx, each in enumerate(pred_fer[0]):
                #     if each > 0:
                #         print(types_fer[idx] + ": "+ ("%.3f" % each) + " ")
                # print("\n")

                #cv2.putText(image, predicted_emotion_fer, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            cv2.imshow("Image", image)
            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass


# Function for Real-Time Audio Analysis
def realtime_audio():

    # Stating global variables to be used
    global start_time_aud
    global start_time_vid
    global pred_aud
    global pred_fer
    

    try:
        while True:
            aud_rec()           # Record audio from user
            rec_path = r'./recaudio.wav'    # SAVE audio as wav file
            
            # Extract features (MFCCs) form the wav audio file
            features = extract_features(rec_path)

            # Predict emotions using trained model for AUDIO using RAVDESS Dataset
            pred_aud = model.predict(features)

            # Record time for audio prediction
            start_time_aud = time.time()

            ############################################## IMPORTANT INFORMATION ##########################################################
            # The RAVDESS and FER+ Dataset have all the same emotion labels other than the emotion 'CALM'
            # which is present in RAVDESS but not in FER+ Dataset

            # So, we came up with an idea to omit the 'Calm' emotion, since it is being predicted only by AUDIO analysis and not by VIDEO
            # and normalize the other probabilities in the predictions array for Audio Emotion Analysis

            # We tried several methods, but the best one was to multiply all the probabilities (except 'Calm') in pred_aud
            # with  1/(sum of all probabilities except 'Calm')
            # By doing this, the sum of all probabilities except 'Calm' sums up to 1, which goes along with the rules of probability

            ################################################################################################################################

            # prob_sum is the sum of all probabilities in pred_aud except 'Calm'
            prob_sum = pred_aud[0][0] + pred_aud[0][2] + pred_aud[0][3] + pred_aud[0][4] + pred_aud[0][5] + pred_aud[0][6] + pred_aud[0][7]

            # 1/(sum of all probabilities except 'Calm') as mentioned above
            x = (1)/(prob_sum)
            
            # Multiplying all of the probabilities
            for j in range(8):
                if j==1:        # Make the probability of Calm zero, which is at index 1
                    pred_aud[0][j] = 0
                else:  
                    pred_aud[0][j] = (x)*(pred_aud[0][j])

            # To make sure that VIDEO analysis has also been done
            if pred_fer is not None:    
                fused_pred(pred_aud,pred_fer)

    except KeyboardInterrupt:
        pass




# FUSION MODEL

# Function to fuse together the predictions made from AUDIO and VIDEO analysis
# Takes in the array of probabilities from AUDIO and VIDEO
def fused_pred(pred_aud,pred_fer):

    # Stating global variables to be used
    global start_time_aud
    global start_time_vid
    
    # New list of emotion labels for the Fusion Model
    emotions_f = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprise']

    # Intiliaze array of fused probabilities to NONE
    pred_f = [None]*7
    

    #######################################
    # We had to allocate proportion of contribution for each of the AUDIO and VIDEO models
    # After testing and analyzing our models, we figured out 60% contribution from AUDIO
    # and 40% contribution from VIDEO as the ideal allocation

    # So we take all the calculated probabilities and multiply them by 
    # 0.6 and 0.4 for audio and video respectively
    #######################################

    
    if start_time_vid is not None:
        # To make sure that the audio and video recorded is in the same timeframe
        # Noticing some lag in the prediction, we chose 3 seconds as the optimal time for the timeframe
        if abs(start_time_vid-start_time_aud) <= 3:         
            pred_f[0] = pred_fer[0][6]*(0.4) + pred_aud[0][0]*(0.6)             # Neutral
            pred_f[1] = pred_fer[0][3]*(0.4) + pred_aud[0][2]*(0.6)             # Happy
            pred_f[2] = pred_fer[0][4]*(0.4) + pred_aud[0][3]*(0.6)             # Sad
            pred_f[3] = pred_fer[0][2]*(0.4) + pred_aud[0][4]*(0.6)             # Angry
            pred_f[4] = pred_fer[0][0]*(0.4) + pred_aud[0][5]*(0.6)             # Fearful
            pred_f[5] = pred_fer[0][1]*(0.4) + pred_aud[0][6]*(0.6)             # Disgust
            pred_f[6] = pred_fer[0][5]*(0.4) + pred_aud[0][7]*(0.6)             # Surprise

            # Rounding the probabilities to 5 decimal places
            for i in range(7):
                pred_f[i] = round(pred_f[i],5)

            print('Probabilities for the emotions are :')
            print(pred_f)
            print(emotions_f)

            pred_f_emo = emotions_f[np.argmax(pred_f)]
            print('The predicted emotion is : ' + str(pred_f_emo))

def terminate_func():
    while True:
        if keyboard.is_pressed('q'):
            sys.exit()

# We used the THREADING module in Python to run AUDIO and VIDEO concurrently

# Declaring threading functions
t1 = threading.Thread(target=realtime_video)
t2 = threading.Thread(target=realtime_audio)
t3 = threading.Thread(target=terminate_func)

# Starting the threading functions
t1.start()
t2.start()
t3.start()

# Joining the threading functions
t1.join()
t2.join()
t3.join()