import numpy as np
import cv2
<<<<<<< HEAD

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
=======
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

blood = "BloodImage_00123.jpg"
#img = np.zeros([640, 640, 3])
img = cv2.imread('/home/sarayu941/blood_imaging/BCCD_Dataset/BCCD/JPEGImages/'+blood)

df = pd.read_csv('/home/sarayu941/blood_imaging/BCCD_Dataset/test.csv')
rslt_df = df.loc[df['filename'] == blood]
celltypes = ["RBC", "WBC", "Platelets"]
colors = [(0,0,255),(255,0,0),(0,0,0)]

for i,name in enumerate(celltypes):
    rslt_df1 = rslt_df.loc[rslt_df['cell_type'] == name]

    for ind in rslt_df1.index:
        xmin = int(rslt_df1["xmin"][ind])
        xmax = int(rslt_df1["xmax"][ind])
        ymin = int(rslt_df1["ymin"][ind])
        ymax = int(rslt_df1["ymax"][ind])
        smear(xmin,ymin,xmax,ymax,img,type=name,color=colors[i])

>>>>>>> 4a932583ddb69fe2b7fa103c88f6503fdbc4ef45
cv2.imshow('Filtered_original',img)
cv2.waitKey()

print()
<<<<<<< HEAD
=======






>>>>>>> 4a932583ddb69fe2b7fa103c88f6503fdbc4ef45
