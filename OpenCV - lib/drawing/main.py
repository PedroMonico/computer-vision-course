import os
import cv2

img = cv2.resize(cv2.imread(os.path.join('.', 'data', 'mclaren.jpg')), (960, 640))

print(img.shape)

# line
cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)

# rectangle
cv2.rectangle(img, (200,350), (450, 600), (0, 0, 255), 5) 

# circle
cv2.circle(img, (500,550), 15, (255, 0, 0), 3)

# text
cv2.putText(img,'Hey!', (800, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)


cv2.imshow('img', img)
cv2.waitKey(0)