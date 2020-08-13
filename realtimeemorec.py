import librosa
#from librosa import display
import os
import pandas as pd
import glob
import numpy as np
import tensorflow
from tensorflow import keras
import pyaudio

import speech_recognition as sr

def rec_audio():
    rec = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something')
        audio = rec.listen(source)
    return audio

def extract_features(filename):
  X, samp_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc_val = np.mean(librosa.feature.mfcc(y=X, sr=samp_rate, n_mfcc=40), axis=1)
  return mfcc_val


model = keras.models.load_model('Speech_Emotion_Recognition800.h5')

def main():
    while True:

        aud_file = rec_audio()
        features = extract_features(aud_file) 
        feat = np.asarray(features)
        pred = model.predict(feat)
        maxprob = np.argmax(pred[0])
        print(maxprob)

        

main()
