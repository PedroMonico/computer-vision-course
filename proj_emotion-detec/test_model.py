import cv2
import pickle 
from utils import get_face_landmarks

emotions = ['HAPPY', 'SAD', 'SURPRISED']

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, draw =True,static_image_mode=False) 

    if face_landmarks and len(face_landmarks) > 0:
            output = model.predict([face_landmarks])        # shape (1, 4)
            emotion = emotions[int(output[0])]      # returns: array([1])
            cv2.putText(frame,
                emotions[int(output[0])], 
                (10, frame.shape[0] - 1), 
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0,255,0),
                5)
    else:
        emotion = "NO FACE"




    cv2.imshow('webcam frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows