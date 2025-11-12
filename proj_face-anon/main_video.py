import cv2
import os
import mediapipe as mp

# read video
video_path = os.path.join('.', 'OpenCV - lib', 'data', 'face.mp4')

video = cv2.VideoCapture(video_path)

# detect faces
mp_face_detection = mp.solutions.face_detection

ret = True

while ret:
    ret, frame = video.read()

    if ret is True:
        H, W, _ = frame.shape

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = face_detection.process(frame_rgb)

            if out.detections is not None:
                for detection in out.detections:
                    location_data = detection.location_data
                    bbox = location_data.relative_bounding_box

                    x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                    x1 = int(x1 * W)
                    y1 = int(y1 * H)
                    w = int(w * W)
                    h = int(h * H)

                    # blur
                    frame[y1:y1 + h, x1:x1 + w] = cv2.blur(frame[y1:y1 + h, x1:x1 + w], (50, 50))

                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

    cv2.imshow('test', frame)
    cv2.waitKey(40)


video.release()
cv2.destroyAllWindows()