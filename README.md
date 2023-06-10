# blood-image-exp
BCCD dataset https://github.com/Shenggan/BCCD_Dataset

# IEEE paper on blood cell counting
https://ieeexplore.ieee.org/document/8652384

The primary objective is to present a more accurate counting of blood cells using the python OpenCV programming language. It covers image processing and analysis of platelets, red blood cells and white blood cells. This study presented a accurate counting of the specified types of cell. The algorithm used in this implimentation consist of five (5) steps: Image Uploading, Color Filtering, Image Segmentation, Blob Detection and the Cell Counting.

![img](algorithm.png)

### Image Uploading

BCCD dataset https://github.com/Shenggan/BCCD_Dataset were uploaded in the Python based program processed and analyzed. In order for a python program to process the image, OpenCV library is imported.

### Color Filtering

Color determination was done to distinctively characterize the WBC, RBC and Platelet cells from each other. Images were converted from BGR to HSV. Then upper and lower bounds as the range of color values were identified and later used for segmentation.

The HSV values of RBC, WBC and Platelets were marked on several images using color_click.py and ploted using seaborn. Thresholds were determined from these values using "max" and "min".

### HSV for RBC, WBC and Platelets respectively

<figure float="left">
  <img src="outputrbc.png" width=240 title="RBC-HSV"/>
  <img src="outputwbc.png" width=240 title="WBC-HSV"/>
  <img src="outputplatelets.png" width=240 title="Platelets-HSV"/>
</figure>

### Image Segmentation
 
Images were segmentated using Otsu’s binarization technique. Only grayscale version of the color filtered images were used.

### Blob Detection

OpenCV SimpleBlobDetector was used to detect individual cells as blobs. Currently only area of cells is used as a criterion to identify the blobs. Since the WBC and platelets have similar color thresholds, the identified WBC are cut out before applying blob detection for platelets.

### Cell Counting

Finally the number of keypoints detected with each of the thresholds corresponding to the different cell types are returned as the number of cells.



The original paper splits blood images into 10 segments before applying the algorithm. This has not been done in this code base.






# Run the notebook in colab
https://colab.research.google.com/github/shreeja-shreeja/blood-image-exp/blob/main/blood_segment.ipynb

https://colab.research.google.com/github/shreeja-shreeja/blood-image-exp/blob/main/blood_cell_count.ipynb
