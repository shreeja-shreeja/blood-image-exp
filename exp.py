import numpy as np
import cv2
import pandas as pd

#Function to create an image with single color background and specified size
def create(rows, cols, color=False):
    if not color:
        myimg = np.ones([rows, cols]) * 255
    else:
        myimg = np.ones([rows, cols, 3]) * 255
        myimg[:, :, 0] = np.ones([rows, cols]) * color[0]
        myimg[:, :, 1] = np.ones([rows, cols]) * color[1]
        myimg[:, :, 2] = np.ones([rows, cols]) * color[2]
        

    return myimg

#Function to create an image with black rectangle in the center
def black_rect(img):
    #width of the rectangle is 2 pixels
    rows, cols, clr = img.shape

    rows_rect = int(rows/4)
    cols_rect = int(cols/4)

    img[rows_rect:rows_rect+2, cols_rect:3*cols_rect, :] = np.zeros([2, 2*cols_rect, clr])
    img[3*rows_rect:3*rows_rect+2, cols_rect:3*cols_rect, :] = np.zeros([2, 2*cols_rect, clr])
    
    img[rows_rect:3*rows_rect, cols_rect:cols_rect+2, :] = np.zeros([2*rows_rect, 2, clr])
    img[rows_rect:3*rows_rect, 3*cols_rect:3*cols_rect+2, :] = np.zeros([2*rows_rect, 2, clr])
    
    return img


# Call function to create the image
img = create(480, 640, color=[0, 128, 128])
img = black_rect(img)

# display image
cv2.imshow('My Image',img)
cv2.waitKey()

print()


