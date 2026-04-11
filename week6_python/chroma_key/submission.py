# Enter your code here
import cv2
import numpy as np

# Global variables for color picking
picked_color = None
picked_coords = None
color_picker_enabled = False

# Standard callback for trackbars
def nothing(x):
    pass

def apply_chroma_key(frame, frame1, background=None, win_size=None, window_name='Chroma Key'):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions from trackbars
    lh = cv2.getTrackbarPos('L | h', window_name)
    uh = cv2.getTrackbarPos('U | h', window_name)
    ls = cv2.getTrackbarPos('L | s', window_name)
    us = cv2.getTrackbarPos('U | s', window_name)
    lv = cv2.getTrackbarPos('L | v', window_name)
    uv = cv2.getTrackbarPos('U | v', window_name)
    lower_green = np.array([lh, ls, lv])
    upper_green = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.resize(frame1, (frame.shape[1], frame.shape[0]))

    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg_part = cv2.bitwise_and(bg, bg, mask=mask)

    combined = cv2.add(fg, bg_part)
    
    if background is not None:
        combined_hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        combined_mask = cv2.inRange(combined_hsv, lower_green, upper_green)
        combined_mask_inv = cv2.bitwise_not(combined_mask)
        background_resized = cv2.resize(background, (combined.shape[1], combined.shape[0]))
        fg_combined = cv2.bitwise_and(combined, combined, mask=combined_mask_inv)
        bg_combined = cv2.bitwise_and(background_resized, background_resized, mask=combined_mask)
        combined = cv2.add(fg_combined, bg_combined)
   
    if win_size:
            combined = cv2.resize(combined, win_size)
    return combined

def mouse_callback(event, x, y, flags, param):
    global picked_color, picked_coords, color_picker_enabled
    if color_picker_enabled and event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        picked_color = hsv_frame[y, x]
        picked_coords = (x, y)
        print(f"Color picked at ({x}, {y}): HSV = {picked_color}")

def pick_color(window_name):
    global color_picker_enabled
    
    # Create a checkbox using trackbar (1 = checked, 0 = unchecked)
    cv2.createTrackbar('Enable Color Picker', window_name, 0, 1, nothing)
    
    return picked_color, picked_coords

if __name__ == "__main__":
    # Chroma Keying with Trackbars
    win_name = 'Chroma Key'
    # Setup Window and Trackbars ONCE
    cv2.namedWindow(win_name)

    cv2.createTrackbar('L | h', win_name, 40, 179, nothing)
    cv2.createTrackbar('U | h', win_name, 80, 179, nothing)
    cv2.createTrackbar('L | s', win_name, 70, 255, nothing)
    cv2.createTrackbar('U | s', win_name, 255, 255, nothing)
    cv2.createTrackbar('L | v', win_name, 70, 255, nothing)
    cv2.createTrackbar('U | v', win_name, 255, 255, nothing)
    
    # Add color picker control
    pick_color(win_name)
    
    cap = cv2.VideoCapture("greenscreen-demo.mp4")
    cap1 = cv2.VideoCapture("greenscreen-asteroid.mp4")
    
    # Create red background for testing with size matching video frames
    background = np.zeros((480, 640, 3), dtype=np.uint8)
    background[:, :] = [0, 0, 255]  # Red background

    # Set up mouse callback
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        
        if not ret or not ret1:
            # Restart videos if they end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Check if color picker is enabled
        color_picker_enabled = cv2.getTrackbarPos('Enable Color Picker', win_name)
        
        # Pass frame to mouse callback
        cv2.setMouseCallback(win_name, mouse_callback, frame)
        
        # Apply the logic using live trackbar values
        combined = apply_chroma_key(frame, frame1, 
                                    background=background,
                                    win_size=(640, 480), 
                                    window_name=win_name)
        
        cv2.imshow(win_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap1.release()
    cv2.destroyAllWindows()