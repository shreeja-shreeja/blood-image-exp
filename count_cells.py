# Import libraries
import numpy as np
import cv2
import pandas as pd
import requests

from matplotlib import pyplot as plt

# Define functions

#Function to download file from internet url
def get_image(url):
    local_filename = 'blood.jpg'
    
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    
    img = cv2.imread(local_filename)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

# Function to draw a rectangle around the set points and annotate
def smear(xmin,ymin,xmax,ymax,img,ctype,color):
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

    cv2.putText(img, ctype, (xmin + 10, ymin + 15),
				cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], color, 2)
    return 

# Apply HSV thresholding
def hsv_filter(img, hsv, threshold):
    # Threshold into a range.
    mask = cv2.inRange(hsv, threshold[0], threshold[1])
    
    if len(threshold) > 2:
        # Used with Red color 
        mask2 = cv2.inRange(hsv, threshold[2], threshold[3])
        mask = mask + mask2
        
    output = cv2.bitwise_and(img,img, mask= mask)
    
    return output

# Apply Otsu's binary thresholding
# Adapted from https://learnopencv.com/otsu-thresholding-with-opencv/
def otsu_th(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is cdetermined automatically.
    otsu_th, image_result = cv2.threshold(
        v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    
    return image_result

# Apply the otsu's filter on the color image
def apply_otsu(hsv_img, otsu_mask):
    h, s, v = cv2.split(hsv_img)
    v1 = v*otsu_mask
    hsv_image = cv2.merge([h, s, v1])

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return out

# Blob detection
# https://stackoverflow.com/questions/65026468/cv2-simpleblobdetector-difficulties for explanation
def detect_blobs(img, area):
    params = cv2.SimpleBlobDetector_Params() 

    params.filterByColor = False
    params.blobColor = 0 

    params.minThreshold = 0 
    params.maxThreshold = 255.0 
    params.thresholdStep = 5 
    params.minDistBetweenBlobs = 3.0 
    params.minRepeatability = 2 
    
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.3

    # Filter by Area.
    params.filterByArea = True
    params.minArea = area[0]
    params.maxArea = area[1]
    
    detector = cv2.SimpleBlobDetector_create(params)

    #keypoints = detector.detect(cv2.bitwise_not(img))
    keypoints = detector.detect(img)
    
    return keypoints

# Set the different thresholds
celltypes = ["WBC", "RBC", "Platelets"]
# Colors to mark annotations
colors = [(0,0,255),(255,0,0),(0,0,0)]

# HSV thresholds
wbc_th = [np.array([120, 30, 180]), np.array([135, 160, 225])]
plt_th = [np.array([110, 55, 180]), np.array([135, 110, 210])]
rbc_th = [np.array([0, 20, 170]), np.array([20, 55, 220]), np.array([150, 20, 170]), np.array([179, 55, 220])]
                                           
ths = [wbc_th, rbc_th, plt_th]

# Area thresholds
areas = [[4000, 60000], [2500, 10000], [150, 300]]

#specify the csv path containing identifiers
csv_path = 'https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/test.csv'
#specify root folder for blood images
root = 'https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/'
# Read annotations
df = pd.read_csv(csv_path)

all_files = df['filename'].to_list()
all_files = list(set(all_files))
all_files.sort()

print('Name,\tWBC,\tWBC-Det,\tRBC,\tRBC-det,\tPlatelets,\tPlatelets-det')

for file in all_files:
    # Get the annotations in the current file
    rslt_df = df.loc[df['filename'] == file]
    
    img = get_image(root+file)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    all_keypts = []

    mydict = {}
    mydict['name'] = file

    for i, cell in enumerate(celltypes):
        rslt_df1 = rslt_df.loc[rslt_df['cell_type'] == cell]

        mydict[cell] = len(rslt_df1)

        # Apply color filtering on the image 
        filt_op = hsv_filter(img, hsv_img, ths[i])
        # Apply image segmentation
        seg_op = otsu_th(filt_op)

        # Apply blob detection
        if cell == 'Platelets':
            for keypt in all_keypts[0]:
                x = int(keypt.pt[0])
                y = int(keypt.pt[1])
                size = int(keypt.size)
                seg_op = cv2.circle(seg_op, (x, y), size, (0, 0, 0), cv2.FILLED, 8, 0)
        
        keypts = detect_blobs(seg_op, areas[i])

        all_keypts.append(keypts)

        mydict[cell + ' detected'] = len(keypts)

    print('{},\t{},\t{},\t{},\t{},\t{},\t{}'.format(mydict['name'],
                                            mydict['WBC'], mydict['WBC detected'],
                                            mydict['RBC'], mydict['RBC detected'],
                                            mydict['Platelets'], mydict['Platelets detected']))

    


    

        
    