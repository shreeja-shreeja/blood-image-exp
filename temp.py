import numpy as np
import cv2
import pandas as pd

def smear(xmin,ymin,xmax,ymax,img,type,color):
    if (xmin<2):
        xmin = 2
    if (ymin<2):
        ymin = 2

    tmp = np.ones([(ymax-ymin), 2, 3]) * 0
    img[ymin:ymax, (xmin-2):xmin, 0:3] = tmp
    tmp = np.ones([2,(xmax-xmin),3]) * 0
    img[(ymin-2):ymin, xmin:xmax, 0:3] = tmp
    tmp = np.ones([(ymax-ymin),2,3]) * 0
    img[ymin:ymax, (xmax-2):xmax, 0:3] = tmp
    tmp = np.ones([2,(xmax-xmin),3]) * 0
    img[(ymax-2):ymax, xmin:xmax, 0:3] = tmp

    cv2.putText(img, type, (xmin + 10, ymin + 15),
							cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], color, 2)
    return 

def blood_seg_new(xmin, ymin, xmax, ymax, img):
    y, x, c = img.shape
    img[0:ymin, :, :] = np.zeros([ymin, x, 3])
    img[ymax:, :, :] = np.zeros([y-ymax, x, 3])
    img[:, 0:xmin, :] = np.zeros([y, xmin, 3])
    img[:, xmax:, :] = np.zeros([y, x-xmax, 3])
    
    #cv2.putText(img, type, (xmin + 10, ymin + 15),
							#cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], color, 2)
    return 
    
blood = "BloodImage_00123.jpg"
#img = np.zeros([640, 640, 3])
img = cv2.imread('/home/sarayu941/blood_imaging/BCCD_Dataset/BCCD/JPEGImages/'+blood)

df = pd.read_csv('/home/sarayu941/blood_imaging/BCCD_Dataset/test.csv')
rslt_df = df.loc[df['filename'] == blood]
#celltypes = ["RBC", "WBC", "Platelets"]

celltypes = ["WBC"]

colors = [(0,0,255),(255,0,0),(0,0,0)]

for i,name in enumerate(celltypes):
    rslt_df1 = rslt_df.loc[rslt_df['cell_type'] == name]

    for ind in rslt_df1.index:
        xmin = int(rslt_df1["xmin"][ind])
        xmax = int(rslt_df1["xmax"][ind])
        ymin = int(rslt_df1["ymin"][ind])
        ymax = int(rslt_df1["ymax"][ind])
        smear(xmin,ymin,xmax,ymax,img,type=name,color=colors[i])

        blood_seg_new(xmin, ymin, xmax, ymax, img)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bob_img = cv2.imread('/home/sarayu941/blood_imaging/BCCD_Dataset/BCCD/JPEGImages/'+blood, cv2.IMREAD_GRAYSCALE)

cv2.imshow('Filtered_original',bob_img)
cv2.waitKey()

print()






