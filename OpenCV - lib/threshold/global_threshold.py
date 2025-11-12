import os
import cv2

img = cv2.resize(cv2.imread(os.path.join('.', 'data', 'mclaren.jpg')), (960, 640))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('thresh_img', thresh)

cv2.waitKey(0)
