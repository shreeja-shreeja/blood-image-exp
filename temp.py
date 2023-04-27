import numpy as np
import cv2

#img = np.zeros([640, 640, 3])
img = cv2.imread('/home/sarayu941/blood_imaging/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00017.jpg')


tmp = np.ones([176, 2, 3]) * 0
img[215:391, 126:128, 0:3] = tmp
tmp = np.ones([2,208,3]) * 0
img[213:215, 128:336, 0:3] = tmp
tmp = np.ones([176,2,3]) * 0
img[215:391, 334:336, 0:3] = tmp
tmp = np.ones([2,208,3]) * 0
img[389:391, 128:336, 0:3] = tmp
cv2.imshow('Filtered_original',img)
cv2.waitKey()

print()
