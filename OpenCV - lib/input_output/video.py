import os
import cv2

# read video
video_path = os.path.join('.', 'data', 'example.mp4')
video = cv2.VideoCapture(video_path)

# visualize video

ret = True

while ret:
    ret, frame = video.read()       # at the end ret will be false (no frame)

    cv2.imshow('frame', frame)
    cv2.waitKey(40)                 # video is 25fps - 1/25 -> 0.04s per photo

video.release()
cv2.destroyAllWindows()