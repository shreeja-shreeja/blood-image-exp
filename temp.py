import numpy as np
import cv2
blood = "BloodImage_00022.jpg"
#img = np.zeros([640, 640, 3])
img = cv2.imread('/home/sarayu941/blood_imaging/BCCD_Dataset/BCCD/JPEGImages/'+blood)

import pandas as pd

df = pd.read_csv('/home/sarayu941/blood_imaging/BCCD_Dataset/test.csv')
rslt_df = df.loc[df['filename'] == blood]
rslt_df1 = rslt_df.loc[rslt_df['cell_type'] == "Platelets"]

for ind in rslt_df1.index:
    xmin = int(rslt_df1["xmin"][ind])
    xmax = int(rslt_df1["xmax"][ind])
    ymin = int(rslt_df1["ymin"][ind])
    ymax = int(rslt_df1["ymax"][ind])

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
cv2.imshow('Filtered_original',img)
cv2.waitKey()

print()

