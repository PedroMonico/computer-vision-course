import os
import cv2

# read image
image_path = os.path.join('.', 'data', 'mclaren.jpg')    # '.' -> ./ 
image = cv2.imread(image_path)

#write imge
cv2.imwrite(os.path.join('.', 'data', 'output.jpg'), image)

# visualize image

cv2.imshow('image', image)
cv2.waitKey(0)  # or number for m sec