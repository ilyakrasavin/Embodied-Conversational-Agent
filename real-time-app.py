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


global pred_aud
global pred_fer
global pred_ck
global start_time_vid
global start_time_aud

def extract_features(filename):

  X, samp_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc_val = np.mean(librosa.feature.mfcc(y=X, sr=samp_rate, n_mfcc=40), axis=1)
  mfcc_n = np.asarray(mfcc_val)
  mfcc_n = mfcc_n.reshape((1,40))
  mfcc_n = np.expand_dims(mfcc_n, axis=2)

  return mfcc_n


def aud_rec():
    rec = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something')
        audio = rec.listen(source)

    # To create a wav file of the recorded audio
    with open('recaudio.wav', 'wb') as fil:
        fil.write(audio.get_wav_data())


# Model for Audio Emotion Recognition
model = keras.models.load_model('./Speech_Emotion_Recognition125.h5')


# Models for Video Emotion Recognition
# Emotion Classifier
model_fer = tf.keras.models.load_model("./fer_model_59.h5")
model_ck = tf.keras.models.load_model("./ck_model_90.h5")

# Face Classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Labels
types_ck = ("Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", "Contempt")
types_fer = ("Fear", "Disgust", "Anger", "Happiness", "Sadness", "Surprise", "Neutral")


def realtime_video():

    global start_time_aud
    global start_time_vid
    global pred_aud
    global pred_fer
    global pred_ck

    pred_fer = [None]*8
    start_time_vid = None
    capture = cv2.VideoCapture(0)

    try:
        while True:
            
            
            ret_value, image = capture.read()

            if not capture:
                continue

            faces_detected = classifier.detectMultiScale(image, 1.32, 5)

            for (x,y,w,h) in faces_detected:

                # cropping region of interest i.e. face area from image
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), thickness=7)
                roi=image[y:y+w,x:x+h]
                roi=cv2.resize(roi,(48, 48))

                img_pixels = tf.keras.preprocessing.image.img_to_array(roi)
                img_pixels = np.expand_dims(img_pixels, axis = -1)
                img_pixels = img_pixels.reshape((-1, 48, 48, 1))

                #pred_ck = model_ck.predict(img_pixels)
                pred_fer = model_fer.predict(img_pixels)
                start_time_vid = time.time()
                # # Predictions on CK dataset
                # max_index_ck = np.argmax(pred_ck[0])
                # predicted_emotion_ck = types_ck[max_index_ck]

                # print("CK+:")
                # for idx, each in enumerate(pred_ck[0]):
                #     if each > 0:
                #         print(types_ck[idx] + ": "+ ("%.3f" % each) + " ")
                # print("\n")


                # Predictions on FER+ Dataset
                max_index_fer = np.argmax(pred_fer[0])
                predicted_emotion_fer = types_fer[max_index_fer]

                # print("FER+:")
                # for idx, each in enumerate(pred_fer[0]):
                #     if each > 0:
                #         print(types_fer[idx] + ": "+ ("%.3f" % each) + " ")
                # print("\n")


                #cv2.putText(image, predicted_emotion_ck, (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(image, predicted_emotion_fer, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            cv2.imshow("Image", image)
            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass



def realtime_audio():

    global start_time_aud
    global start_time_vid
    global pred_aud
    global pred_fer
    

    try:
        while True:
            aud_rec()
            rec_path = r'./recaudio.wav'
            features = extract_features(rec_path)

            start_time_aud = time.time()

            pred_aud = model.predict(features)
            
            if pred_fer is not None:    
                fused_pred(pred_aud,pred_fer)
            #print(pred_aud)
            #print(pred_aud[0])
            #emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
            #maxprob = np.argmax(pred_aud[0])
            #print(emotions[maxprob])

    except KeyboardInterrupt:
        pass

def fused_pred(pred_aud,pred_fer):

    emotions_f = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprise', 'Calm']
    pred_f = [None]*8
    global start_time_aud
    global start_time_vid
    #print('pred_fer')
    #print(start_time_vid)
    #print(pred_fer[0][6])
    #print('pred_aud') 
    #print(start_time_aud)
    #print(pred_aud[0][0])
    #print(start_time_aud)
    #print(start_time_vid)
    if start_time_vid is not None:
        if abs(start_time_vid-start_time_aud) <= 3:
            pred_f[0] = pred_fer[0][6]*(0.4) + pred_aud[0][0]*(0.6)
            pred_f[1] = pred_fer[0][3]*(0.4) + pred_aud[0][2]*(0.6)
            pred_f[2] = pred_fer[0][4]*(0.4) + pred_aud[0][3]*(0.6)
            pred_f[3] = pred_fer[0][2]*(0.4) + pred_aud[0][4]*(0.6)
            pred_f[4] = pred_fer[0][0]*(0.4) + pred_aud[0][5]*(0.6)
            pred_f[5] = pred_fer[0][1]*(0.4) + pred_aud[0][6]*(0.6)
            pred_f[6] = pred_fer[0][5]*(0.4) + pred_aud[0][7]*(0.6)
            pred_f[7] = pred_aud[0][0]

            print(pred_f)
            pred_f_emo = emotions_f[np.argmax(pred_f)]
            print(pred_f_emo)


t1 = threading.Thread(target=realtime_video)
t2 = threading.Thread(target=realtime_audio)

t1.start()
t2.start()

t1.join()
t2.join()
