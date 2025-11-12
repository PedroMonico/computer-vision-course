import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'mclaren.jpg'))

print(img.shape)

cropped_img = img[320:640, 420:840]

cv2.imshow('cropped img', cropped_img)
cv2.waitKey(0)