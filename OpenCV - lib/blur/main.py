import os
import cv2

img = cv2.resize(cv2.imread(os.path.join('.', 'data', 'mclaren.jpg')), (960, 640))
k_size = 7

img_blur = cv2.blur(img, (k_size, k_size))

img_gaussianblur = cv2.GaussianBlur(img, (k_size, k_size), 3)

img_median_blur = cv2.medianBlur(img, k_size)


cv2.imshow('img blur', img_blur)
cv2.imshow('img gaussian blur', img_gaussianblur)
cv2.imshow('img median blur', img_median_blur)

cv2.waitKey(0)