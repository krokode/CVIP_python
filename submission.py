import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

filename = "haarcascade_frontalface_default.xml"
# if not os.path.exists(filename):
#     import urllib.request
#     url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
#     urllib.request.urlretrieve(url, filename)
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)

def apply_sunglasses_filter(face_cascade, frame, glasses):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = frame.copy()

    for (x, y, w, h) in faces:
        eye_y_level = int(y + h * 0.42)
            
        glasses_width = int(w * 0.9)
        glasses_height = int(glasses_width * 0.4)
            
        start_x = x + (w - glasses_width) // 2
        start_y = eye_y_level - (glasses_height // 2)
            
        end_x = min(result.shape[1], start_x + glasses_width)
        end_y = min(result.shape[0], start_y + glasses_height)
        start_x = max(0, start_x)
        start_y = max(0, start_y)

        if start_x >= end_x or start_y >= end_y:
            continue

        roi_glasses = result[start_y:end_y, start_x:end_x]
        h_g, w_g = roi_glasses.shape[:2]
        sunglasses = cv2.resize(glasses, (w_g, h_g))

        alpha = sunglasses[:, :, 3] / 255.0
        alpha = alpha * 0.5  # Make sunglasses 50% transparent to see eyes
        
        for c in range(3):
            result[start_y:end_y, start_x:end_x, c] = (
                alpha * sunglasses[:, :, c] + (1.0 - alpha) * result[start_y:end_y, start_x:end_x, c]
            ).astype(np.uint8)
                
    return result

glasses = cv2.imread("sunglass.png", cv2.IMREAD_UNCHANGED)
frame = cv2.imread("musk.jpg")

sunglasses_on = apply_sunglasses_filter(face_cascade, frame, glasses)

plt.imshow(cv2.cvtColor(sunglasses_on, cv2.COLOR_BGR2RGB))
plt.title('Sunglasses Filter')
plt.axis('off')
plt.show()





