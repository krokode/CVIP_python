import numpy as np
import cv2 as cv
from pathlib import Path

folder = Path("data/images")
file_name = Path("circles.jpg")
file_name_save = Path("houghcircles.jpg")

img = cv.imread(folder / file_name, 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, 
                          param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# cv.imshow('detected circles', cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cv.imwrite(folder / file_name_save, cimg)

file_name_save1 = Path("houghcircles1.jpg")

img1 = cv.imread(folder / file_name, cv.IMREAD_COLOR)
# Convert to gray-scale
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# Blur the image to reduce noise
img_blur = cv.medianBlur(gray1, 5)

circles1 = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, 50, param1=450, param2=10, minRadius=30, maxRadius=40)
# Draw detected circles
if circles1 is not None:
    circles1 = np.uint16(np.around(circles1))
    for i in circles1[0, :]:
        # Draw outer circle
        cv.circle(img1, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv.circle(img1, (i[0], i[1]), 2, (0, 0, 255), 3)
cv.imshow('detected circles1', img1)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite(folder / file_name_save1, img1)