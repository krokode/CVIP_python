from pathlib import Path
import cv2

# load an image
dirPath = Path('sub')
imgPath = Path(dirPath, 'truth.png')

img = cv2.imread(str(imgPath), cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"There is no image in {imgPath}")
    exit() # Exit if the image didn't load
img_height, img_width = img.shape[:2]
max_height = 1920
max_width = 1080
# By default, the scale is 1.0 (no change)
scale = 1.0
# Check if the image is too tall
if img_height > max_height:
    # Calculate the scale needed to fit the height
    scale = max_height / img_height

# Check if the image is too wide, Use min() to find the *smallest* scale. 
# This is the one that will make *both* dimensions fit.
if img_width > max_width:
    scale = min(scale, max_width / img_width)

# We also convert to int, as pixel dimensions must be whole numbers.
new_height = int(img_height * scale)
new_width = int(img_width * scale)

if scale < 1.0:
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

im = img
maxScaleUp = 100
scaleFactor = 1
scaleType = 0
maxType = 1

windowName = "Resize Image"
trackbarValue = "Scale"
trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

# Callback functions
def scaleImage(*args):
    global scaleFactor
    global scaleType
    
    if scaleType == 0:
        # Get the scale factor from the trackbar 
        scaleFactor = 1 + args[0]/100.0
    else:
        scaleFactor = 1 - args[0]/100.0
    
    # Perform check if scaleFactor is zero
    if scaleFactor == 0:
        scaleFactor = 1
    
    # Resize the image
    scaledImage = cv2.resize(im, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)

# Callback functions
def scaleTypeImage(*args):
    global scaleType
    global scaleFactor
    scaleType = args[0]
    
    if scaleType == 0:
        # Get the scale factor from the trackbar 
        scaleFactor = 1 + scaleFactor/100.0
    else:
        scaleFactor = 1 - scaleFactor/100.0

    if scaleFactor ==0:
        scaleFactor = 1
    scaledImage = cv2.resize(im, None, fx=scaleFactor,
                             fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)


cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)
cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleTypeImage)

cv2.imshow(windowName, im)
c = cv2.waitKey(0)

cv2.destroyAllWindows()