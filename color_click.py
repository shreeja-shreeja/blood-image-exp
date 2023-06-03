import cv2
import numpy as np
import requests
import pandas as pd

blood = "BloodImage_00304.jpg"

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
    
    return img

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

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(ann_img,(x,y),5,(255,0,0),-1)
        mouseX,mouseY = x,y


#specify the csv path containing identifiers
csv_path = 'https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/test.csv'

root = 'https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/'
img = get_image(root+blood)

# Read annotations
df = pd.read_csv(csv_path)
rslt_df = df.loc[df['filename'] == blood]

celltypes = ["RBC", "WBC", "Platelets"]
colors = [(0,0,255),(255,0,0),(0,0,0)]

ann_img = img.copy()

for i,name in enumerate(celltypes):
    rslt_df1 = rslt_df.loc[rslt_df['cell_type'] == name]

    for ind in rslt_df1.index:
        xmin = int(rslt_df1["xmin"][ind])
        xmax = int(rslt_df1["xmax"][ind])
        ymin = int(rslt_df1["ymin"][ind])
        ymax = int(rslt_df1["ymax"][ind])
        smear(xmin,ymin,xmax,ymax,ann_img,type=name,color=colors[i])

# Create a window
cv2.namedWindow('image')

#img = cv2.imread('blood.jpg')
output = ann_img
waitTime = 1000

cv2.setMouseCallback('image',draw_circle)

curX = curY = 0
mouseX = mouseY = 0

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while(1):
    # Display output image
    cv2.imshow('image',output)

    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

    if( (curX != mouseX) or (curY != mouseY) ):
        #print('MouseX = {} MouseY = {}'.format(mouseX, mouseY))
        #print('H = {} S = {} V = {}'.format(hsv[mouseY, mouseX, 0], hsv[mouseY, mouseX, 1], hsv[mouseY, mouseX, 2]))
        print('{:3d} , {:3d} , {:3d}'.format(hsv[mouseY, mouseX, 0], hsv[mouseY, mouseX, 1], hsv[mouseY, mouseX, 2]))
        curX = mouseX
        curY = mouseY

cv2.destroyAllWindows()
        
    