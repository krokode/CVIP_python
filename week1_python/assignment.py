from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

imgFolder = Path('data', 'images')
imgName = 'IDCard-Satya.png'

imgPath = Path(imgFolder, imgName)

img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
cv2.imshow('IDCard', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

qrDecoder = cv2.QRCodeDetector()
opencvData, bbox, rectifiedImage = qrDecoder.detectAndDecode(img)
n = len(bbox[0])
for i in range(n):
    nextPointIndex = (i+1) % n
    pts1 = tuple(map(int, (bbox[0][i])))
    pts2 = tuple(map(int, bbox[0][nextPointIndex]))
    cv2.line(img, pts1, pts2, (255,0,0, 255), 5)
print(opencvData)
cv2.imwrite('QROut.png', img)
cv2.imshow('IDCard', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img)


