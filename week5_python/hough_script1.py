import cv2 as cv
import numpy as np
from pathlib import Path

folder = Path("data/images")
file_name = Path("scanned-form.jpg")
file_name_save = Path("houghlines2.jpg")

img = cv.imread(folder / file_name)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imwrite(folder / file_name_save, img)