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


#def rec_audio():
    #rec = sr.Recognizer()

    #with sr.Microphone() as source:
    #    print('Say something')
    #    audio = rec.listen(source)
    #return audio


def extract_features(filename):
  X, samp_rate = librosa.load(filename, res_type='kaiser_fast')
  mfcc_val = np.mean(librosa.feature.mfcc(y=X, sr=samp_rate, n_mfcc=40), axis=1)
  return mfcc_val


def aud_rec():
    rec = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something')
        audio = rec.listen(source)

        # To create a wav file of the recorded audio
    with open('recaudio.wav', 'wb') as fil:
        fil.write(audio.get_wav_data())


model = keras.models.load_model('Speech_Emotion_Recognition800.h5')


def main():

    aud_rec()
    rec_path = r'C:\Users\Naqsh Thind\Desktop\CMPT419 Project\recaudio.wav'
    features = extract_features(rec_path)
    feat = np.asarray(features)
    feat_n = feat.reshape((1,40))
    print(feat_n.shape)
    #np.flip(feat_n,1)
    #print(feat_n.shape)
    #print(feat.shape)
    feat_nn = np.expand_dims(feat_n, axis=2)
    #print(feat_n.ndim)
    pred = model.predict(feat_nn)
    print(pred)
    print(pred[0])

    emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
    maxprob = np.argmax(pred[0])
    print(emotions[maxprob])

#print(model.get_weights())
#print(model)   
#aud_rec()
main()
