import cv2
from PIL import Image
from color_region import get_limits

yellow = [0, 255, 255]
webcam = cv2.VideoCapture(0)

while True:        
    ret, frame = webcam.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=yellow)

    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)       # convert image from array(openCV) representation to pillow)

    bbox = mask_.getbbox()

    if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow('yellow detection', frame)
    cv2.imshow('mask', mask)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()