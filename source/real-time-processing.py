import numpy as np
import os
import cv2
import tensorflow as tf

# Emotion Classifier
model_fer = tf.keras.models.load_model("./fer_model_59.h5")
model_ck = tf.keras.models.load_model("./ck_model_90.h5")

# Face Classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Labels
types_ck = ("Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", "Contempt")
types_fer = ("Fear", "Disgust", "Anger", "Happiness", "Sadness", "Surprise", "Neutral")

capture = cv2.VideoCapture(0)

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

        predictions_ck = model_ck.predict(img_pixels)
        predictions_fer = model_fer.predict(img_pixels)

        # Predictions on CK dataset
        max_index_ck = np.argmax(predictions_ck[0])
        predicted_emotion_ck = types_ck[max_index_ck]

        print("CK+:")
        for idx, each in enumerate(predictions_ck[0]):
            if each > 0:
                print(types_ck[idx] + ": "+ ("%.3f" % each) + " ")
        print("\n")


        # Predictions on FER+ Dataset
        max_index_fer = np.argmax(predictions_fer[0])
        predicted_emotion_fer = types_fer[max_index_fer]

        print("FER+:")
        for idx, each in enumerate(predictions_fer[0]):
            if each > 0:
                print(types_fer[idx] + ": "+ ("%.3f" % each) + " ")
        print("\n")


        cv2.putText(image, predicted_emotion_ck, (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, predicted_emotion_fer, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    cv2.imshow("Image", image)
    cv2.waitKey(1)
