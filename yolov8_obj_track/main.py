import cv2
from ultralytics import YOLO

# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './test.mp4'
video = cv2.VideoCapture(video_path)

# read frames
ret = True
while ret:
    ret, frame = video.read()
    if ret:
        # detect objects | track objects
        results = model.track(frame, persist=True)    # to remember the detection os the last frame

        # plot results
        # cv2.rectangle and cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break