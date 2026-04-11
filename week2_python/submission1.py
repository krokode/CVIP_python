from pathlib import Path
import cv2

dirPath = Path('sub')
imgPath = Path(dirPath, 'trick-9136139_1920.png')

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

winName = 'Assignment #2 by krokodeele'
source = img.copy()
leftup_corners=[]
rightdown_corners=[]
msg_line = 0


def drawrectangle(action, x, y, flags, userdata):
    global leftup_corners, rightdown_corners, source, img, winName, msg_line

    # Mouse pressed — start rectangle
    if action == cv2.EVENT_LBUTTONDOWN:
        leftup_corners = [(x, y)]
        rightdown_corners = []
        cv2.circle(source, leftup_corners[0], 1, (255,255,0), 2, cv2.LINE_AA )
    
    # Mouse moving — draw preview
    elif action == cv2.EVENT_MOUSEMOVE and leftup_corners:
        temp = source.copy()
        rightdown_corners = [(x, y)]
        cv2.rectangle(temp, leftup_corners[0], rightdown_corners[0], (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow(winName, temp)

    # Mouse released — finalize rectangle
    elif action == cv2.EVENT_LBUTTONUP and leftup_corners:
        rightdown_corners = [(x, y)]
        x1, y1 = leftup_corners[0]
        x2, y2 = rightdown_corners[0]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        rect = img[y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)

        if area > 1000:
            out = Path(dirPath, f'copied_rectangle_{x1}_{y1}_{x2}_{y2}.png')
            cv2.imwrite(str(out), rect)
            y_text = 60 + msg_line * 25
            cv2.putText(source, f"Saved {out.name}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        else:
            y_text = 60 + msg_line * 25
            cv2.putText(source, f"Rectangle is very small {x2-x1} for {y2-y1} only nothing to save...", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                
        msg_line += 1

        cv2.rectangle(source, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow(winName, source)
        leftup_corners = []
        rightdown_corners = []


cv2.namedWindow(winName)
# highgui function called when mouse events occur
cv2.setMouseCallback(winName, lambda action, x, y, flags, userdata: drawrectangle(action, x, y, flags, userdata))
k = 0
# loop until escape character is pressed
while k!=27 :
    display = source.copy()
    cv2.putText(source,'''Choose leftUP corner, and drag on rightDOWN, 
                      Press ESC to exit and c to clear''' ,
              (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
              0.7,(255,255,255,255), 2 )
    cv2.imshow(winName, display)
    k = cv2.waitKey(20) & 0xFF
    # if 'c' is pressed, clear the image
    if k == ord('c'):
        source = img.copy()
        leftup_corners = []
        rightdown_corners = []
        msg_line = 0

cv2.destroyAllWindows()

