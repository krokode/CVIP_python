import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Image path
imagePath = DATA_PATH + "images/CoinsA.png"
# Read image
# Store it in the variable image
###
image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
###
imageCopy = image.copy()
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()

# Convert image to grayscale
# Store it in the variable imageGray
###
imageGray = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)
###

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(122)
plt.imshow(imageGray);
plt.title("Grayscale Image");
plt.show()

# Split cell into channels
# Store them in variables imageB, imageG, imageR
###
imageB = imageCopy[:,:,0]
imageG = imageCopy[:,:,1]
imageR = imageCopy[:,:,2]
###

plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(142)
plt.imshow(imageB);
plt.title("Blue Channel")
plt.subplot(143)
plt.imshow(imageG);
plt.title("Green Channel")
plt.subplot(144)
plt.imshow(imageR);
plt.title("Red Channel");
plt.show()

###
# to iterate though all one channel images
images = [imageGray, imageB, imageG, imageR]
###
# Set threshold and maximum value
thresh = [100, 50, 40]
maxValue = [255, 100, 70]
thresh_results = []

for i in range(len(images)):
    for j in range(len(thresh)):
        retval, dst = cv2.threshold(images[i], thresh[j], maxValue[j], cv2.THRESH_BINARY_INV)
        thresh_results.append(dst)

plt.figure(figsize=[30, 10])
for i in range(len(thresh_results)):
    plt.subplot(1,len(thresh_results),i+1);plt.imshow(thresh_results[i]);plt.title(i+1)

plt.show()

# Display the thresholded image
###
# my choice is plot 9 Green chanel thresh 40 maxValue 70
retval, dst = cv2.threshold(images[2], thresh[2], maxValue[2], cv2.THRESH_BINARY_INV)
plt.imshow(dst);plt.title("Green chanel thresh 40 maxValue 70")
plt.show()
###

###
kSize = (3,3)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

dilated = cv2.dilate(dst, kernel)
eroded = cv2.erode(dilated, kernel)
closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=3)
###
###
kSize1 = (5,5)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize1)

dilated1 = cv2.dilate(dst, kernel1)
eroded1 = cv2.erode(dilated1, kernel1)
closed1 = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel1, iterations=1)
###
# Display all the images
# you have obtained in the intermediate steps
###
plt.subplot(261);plt.imshow(dilated)
plt.subplot(262);plt.imshow(eroded)
plt.subplot(263);plt.imshow(closed)
plt.subplot(264);plt.imshow(dilated1)
plt.subplot(265);plt.imshow(eroded1)
plt.subplot(266);plt.imshow(closed1)
plt.show()
###
# Get structuring element/kernel which will be used for dilation
###
kSize1 = (3,3)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize1)
###
dilated = cv2.dilate(dst, kernel1, iterations=2)
###
fine = cv2.erode(dilated, kernel1, iterations=4)

_, fine_binary = cv2.threshold(fine, 1, 255, cv2.THRESH_BINARY_INV)
fine_binary = cv2.erode(fine_binary, kernel1, iterations=2)
plt.imshow(fine_binary);plt.title("Image after extra erosion")
plt.show()
###
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 255

params.minDistBetweenBlobs = 1

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs
###
keypoints = detector.detect(fine_binary)
###
# Print number of coins detected
###
print("Coins detected:", len(keypoints))
###
# Mark coins using image annotation concepts we have studied so far
###
im = image.copy()
for k in keypoints:
    x,y = k.pt
    x=int(round(x))
    y=int(round(y))
    # Mark center in BLACK
    cv2.circle(im,(x,y),5,(0,0,0),-1)
    # Get radius of blob
    diameter = k.size
    radius = int(round(diameter/2))
    # Mark blob in RED
    cv2.circle(im,(x,y),radius,(0,0,255),2)
    
###
# Display the final image
###
plt.imshow(im[:,:,::-1])
plt.show()
###
def displayConnectedComponents(im):
    imLabels = im
    # The following line finds the min and max pixel values
    # and their locations in an image.
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imLabels)
    # Normalize the image so the min value is 0 and max value is 255.
    imLabels = 255 * (imLabels - minVal)/(maxVal-minVal)
    # Convert image to 8-bits unsigned type
    imLabels = np.uint8(imLabels)
    # Apply a color map
    imColorMap = cv2.applyColorMap(imLabels, cv2.COLORMAP_JET)
    # Display colormapped labels
    plt.imshow(imColorMap[:,:,::-1])
    plt.show()

# Find connected components
###
print("dtype:", fine_binary.dtype)
print("unique values:", np.unique(fine_binary))

plt.imshow(fine_binary, cmap='gray')
plt.title("Binary mask")
plt.imsave("9blobs.png", fine_binary)
plt.imshow(fine_binary)
plt.show()

# Get labels AND statistics (stats)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fine_binary, 8, cv2.CV_32S)
# set min blob size
MIN_PIXEL_AREA = 150

# Create a new labels array, initialized to 0 (background)
filtered_labels = np.zeros_like(labels, dtype=labels.dtype)

# Initialize a counter for the new, re-assigned labels
new_label_ID = 1

# Iterate through all components (skip label 0 - background)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    
    if area >= MIN_PIXEL_AREA:
        filtered_labels[labels == i] = new_label_ID
        new_label_ID += 1

###
# Print number of connected components detected
###
print(f"Original connected components count (including background): {num_labels}")
print(f"Total filtered labels (excluding background): {new_label_ID - 1}")
###
# Display connected components using displayConnectedComponents
# function
###
displayConnectedComponents(filtered_labels)
###
# Find all contours in the image
###
contours, hierarchy = cv2.findContours(fine_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
###
# Print the number of contours found
###
actual_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 190:
        actual_contours.append(cnt)
print("Number of contours found = {}".format(len(actual_contours)))
###
# Draw all contours
###
color_fine_binary = cv2.cvtColor(fine_binary, cv2.COLOR_GRAY2BGR)
        
cv2.drawContours(color_fine_binary, actual_contours, -1, (0,255,0), 3);

plt.imshow(color_fine_binary[:,:,::-1])
plt.show()
###
for index,cnt in enumerate(actual_contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))

for cnt in actual_contours:
    # Fit a circle
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(imageCopy, (int(x),int(y)), int(round(radius)), (0,255,0), 3)

plt.imshow(imageCopy[:,:,::-1])
plt.show()

# Image path
imagePath = DATA_PATH + "images/CoinsB.png"
# Read image
# Store it in variable image
###
image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
###
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()
# Convert to grayscale
# Store in variable imageGray
###
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
###
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(122)
plt.imshow(imageGray);
plt.title("Grayscale Image");
plt.show()
# Split cell into channels
# Variables are: imageB, imageG, imageR
###
imageCopy = image.copy()
imageB = imageCopy[:,:,0]
imageG = imageCopy[:,:,1]
imageR = imageCopy[:,:,2]
###
plt.figure(figsize=(20,12))
plt.subplot(141)
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.subplot(142)
plt.imshow(imageB);
plt.title("Blue Channel")
plt.subplot(143)
plt.imshow(imageG);
plt.title("Green Channel")
plt.subplot(144)
plt.imshow(imageR);
plt.title("Red Channel");
plt.show()
###
retval, thresholded = cv2.threshold(imageB, 125, 255, cv2.THRESH_BINARY)
###
# Display image using matplotlib
###
plt.imshow(thresholded)
plt.show()
###
###
kSize2 = (3,3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize2)
###
###
eroded2 = cv2.erode(thresholded, kernel2, iterations=1)
###
###
plt.imshow(eroded2)
plt.show()
###
###
kSize3 = (15,15)
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize3)
###
###
closing3 = cv2.morphologyEx(eroded2, cv2.MORPH_CLOSE, kernel3, iterations=1)
opening3 = cv2.morphologyEx(closing3, cv2.MORPH_CLOSE, kernel3, iterations=1)
eroded3 = cv2.erode(opening3, kernel3, iterations=1)

plt.imshow(eroded3)
plt.show()
###
###
retval, thresholded3 = cv2.threshold(eroded3, 1, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresholded3)
plt.show()
###
print("dtype:", thresholded3.dtype)
print("unique values:", np.unique(thresholded3))
###
#dtype: uint8
#unique values: [  0 255]
###
plt.imshow(thresholded3, cmap='gray')
plt.title("What the detector sees")
plt.imsave("10_Coin_blobs.png", thresholded3)
plt.show()
###
###
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8
# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs
###
keypoints2 = detector.detect(thresholded3)
print(len(keypoints2))

