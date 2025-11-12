import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize face mesh once outside the function
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5)

def get_face_landmarks(image, draw=False, static_image_mode=True):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        face_landmark = results.multi_face_landmarks[0]

        if draw:
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        xs_, ys_, zs_ = [], [], []

        for point in face_landmark.landmark:
            xs_.append(point.x)
            ys_.append(point.y)
            zs_.append(point.z)

        min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)

        for j in range(len(xs_)):
            image_landmarks.extend([
                xs_[j] - min_x,
                ys_[j] - min_y,
                zs_[j] - min_z
            ])

    return image_landmarks
