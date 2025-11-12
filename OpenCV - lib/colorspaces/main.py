import os
import cv2

resized_img = cv2.resize(cv2.imread(os.path.join('.', 'data', 'mclaren.jpg')), (960, 640))

img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)  #  for color detector

cv2.imshow('image', resized_img)
cv2.imshow('image rgb', img_rgb)
cv2.imshow('image gray', img_gray)
cv2.imshow('image hsv', img_hsv)

cv2.waitKey(0)