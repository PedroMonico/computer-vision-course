import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'mclaren.jpg'))

print(img.shape)

resized_img = cv2.resize(img, (960, 640))

print(resized_img.shape)

cv2.imshow('resized img', resized_img)
cv2.waitKey()
