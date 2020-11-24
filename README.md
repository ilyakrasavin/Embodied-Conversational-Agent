# Facial Emotion Recognition Framework
Affective Computing (CMPT 419 / 983) at Simon Fraser University.

## About
This base Module provides Affective Context Awareness for an early-stage Embodied Conversational Agent.<br>
Microsoft FER2013 and Cohn-Kanade+ Datasets are used to train ConvNets in Keras, which are employed for Facial Emotion Recognition using OpenCV software.

## Contents

__/Source__ folder contains the following:<br>

### Code
Video_Recognition.ipynb<br>
- Main module: Datasets, ConvNets Training & Evaluation<br>

video_testing.ipynb<br>
- Testing Emotion Recognition on Videos. Includes a frame extraction script<br>

real_time_processing.py<br>
- Real-Time Face Detection & Emotion Recognition using OpenCV<br>

### Trained Models
ck_model_90.h5: CK+ Dataset<br>

fer_model_59.h5: FER2013 Dataset<br>

haarcascade_frontalface_default.xml: HaarCascade Face Classifier<br>

## Running

Modules mentioned above rely on the following packages:<br>
> Pandas, Numpy, h5py, os, tensorflow, pydot, graphviz, matplotlib, PIL, seaborn, scikit-learn, io, cv2

Notebooks can be accessed by running jupyter-notebook locally from your terminal<br>
Real-time script needs to be run the following way:
> python3 real_time_processing.py


## Credits
Ilia Krasavin (Facial Emotion Recognition - OpenCV, Keras, TF)<br>
Naqsh Thind (Speech Emotion Recognition - PyAudio, Keras)<br>
Sage Clouston (Embodied Agent - SmartBody)<br>
