import cv2
import os
import mediapipe as mp

def process_image(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            print(bbox)

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            print(x1, y1, w, h)

            # blur
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (50, 50))

            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)

    return img

output_dir = './proj_face-anon/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(output_dir)

# read image
image_path = os.path.join('.', 'OpenCV - lib', 'data', 'face.jpg')

img = cv2.resize(cv2.imread(image_path), (960, 640))

H, W, _ = img.shape

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img = process_image(img, face_detection)

cv2.imshow('test', img)
cv2.waitKey(0)

#cv2.imwrite(os.path.join(output_dir, 'output.jpg'), img)