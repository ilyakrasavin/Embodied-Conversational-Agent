import numpy as np
import os
import cv2
import tensorflow as tf

# Emotion Classifier
# model = tf.keras.models.load_model("./fer_model_42.h5")
model = tf.keras.models.load_model("./ck_model_81.h5")

# Face Classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:

    ret_value, image = capture.read()

    if not capture:
        continue

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = classifier.detectMultiScale(image, 1.32, 5)

    for (x,y,w,h) in faces_detected:

        #cropping region of interest i.e. face area from  image
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi=image[y:y+w,x:x+h]
        roi=cv2.resize(roi,(48, 48))

        img_pixels = tf.keras.preprocessing.image.img_to_array(roi)
        img_pixels = np.expand_dims(img_pixels, axis = -1)
        img_pixels = img_pixels.reshape((-1, 48, 48, 1))

        predictions = model.predict(img_pixels)
        probs = model.predict_proba(img_pixels)

        # CK_Correct Labels
        types = ("Anger", "Disgust", "Fear", "Happy", "Sadness", "Surprise", "Contempt")

        # FER_Correct Labels
        # types = ("Fear", "Disgust", "Anger", "Happiness", "Sadness", "Surprise", "Neutral")


        # find max indexed array
        max_index = np.argmax(predictions[0])
        predicted_emotion = types[max_index]


        # Partial predictions on CK+ dataset.
        max_non_neutral = 0
        max_non_neutral_idx = 0
        for idx, each in enumerate(predictions[0]):
            if each > 0 and idx != 2:
                if each > max_non_neutral:
                    max_non_neutral = each
                    max_non_neutral_idx = idx
                    print(types[max_non_neutral_idx] + ": "+ ("%.3f" % max_non_neutral) + " ")
        print("\n")


        # Partial predictions on FER+ dataset.
        # max_non_neutral = 0
        # max_non_neutral_idx = 0
        # for idx, each in enumerate(predictions[0]):
        #     if each > 0 and idx != 3:
        #         if each > max_non_neutral:
        #             max_non_neutral = each
        #             max_non_neutral_idx = idx
        #         print(types[max_non_neutral_idx] + ": "+ ("%.3f" % max_non_neutral) + " ")
        # print("\n")


        cv2.putText(image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    cv2.imshow("Image", image)
    cv2.waitKey(1)
