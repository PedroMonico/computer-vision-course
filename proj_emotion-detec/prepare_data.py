import os
import cv2
import numpy as np

from utils import get_face_landmarks

data_dir = './dataset'
output = []

for emotion_idx, emotion in enumerate(sorted(os.listdir(data_dir))):
    i = 0
    for image_path in os.listdir(os.path.join(data_dir, emotion)):
        i = i+1
        if i<500:
            image_path = os.path.join(data_dir, emotion, image_path)

            image = cv2.imread(image_path)

            face_landmarks = get_face_landmarks(image)  # to extract all the landmarks

            if len(face_landmarks) == 1404:
                face_landmarks.append(int(emotion_idx))
                output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))  # not to consume so much memory -> to train the model