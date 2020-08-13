# To convert speech to text in real time

import speech_recognition as sr

def speechtotext():
    rec = sr.Recognizer()

    with sr.Microphone() as source:
        print('Say something')
        audio = rec.listen(source)

        try:
            text = rec.recognize_google(audio)
            print('Speech to text is: {}'.format(text))

        except:
            print('Error while recognizing!')

        # To create a wav file of the recorded audio
        with open('recaudio.wav', 'wb') as fil:
            fil.write(audio.get_wav_data())


speechtotext()
    

