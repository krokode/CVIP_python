import cv2
import numpy as np

# Global variables for state management
picked_color = [0, 0, 255] # Default to Red (BGR)
color_picker_enabled = False

def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    global picked_color, color_picker_enabled
    if color_picker_enabled and event == cv2.EVENT_LBUTTONDOWN:
        picked_color = param[y, x].tolist()

def run_chroma_key(win_name, win_size=(640, 480)):
    global color_picker_enabled, picked_color

    cv2.namedWindow(win_name)

    cv2.createTrackbar('Picker', win_name, 0, 1, nothing)

    # Video Sources
    cap = cv2.VideoCapture("greenscreen-demo.mp4")
    cap_asteroid = cv2.VideoCapture("greenscreen-asteroid.mp4")

    # Pre-allocate background image
    bg_canvas = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        ret_ast, frame_ast = cap_asteroid.read()

        # Loop videos if they reach the end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if not ret_ast:
            cap_asteroid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ast, frame_ast = cap_asteroid.read()

        # Resize inputs to standard processing size once
        frame = cv2.resize(frame, (win_size[0], win_size[1]))
        frame_ast = cv2.resize(frame_ast, (win_size[0], win_size[1]))

        # Update Color Picker State
        color_picker_enabled = cv2.getTrackbarPos('Picker', win_name)
        cv2.setMouseCallback(win_name, mouse_callback, frame)

        # Update the background canvas with the picked color
        bg_canvas[:] = picked_color

        # Process the Asteroid Video
        ast_hsv = cv2.cvtColor(frame_ast, cv2.COLOR_BGR2HSV)
        ast_mask = cv2.inRange(ast_hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        ast_mask_inv = cv2.bitwise_not(ast_mask)
        
        # Layer: Asteroid on top of Solid Color
        bg_combined = cv2.bitwise_and(bg_canvas, bg_canvas, mask=ast_mask)
        fg_asteroid = cv2.bitwise_and(frame_ast, frame_ast, mask=ast_mask_inv)
        dynamic_background = cv2.add(bg_combined, fg_asteroid)

        # Process the Main Subject
        main_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_bound = np.array([35, 50, 50])
        upper_bound = np.array([85, 255, 255])
        
        main_mask = cv2.inRange(main_hsv, lower_bound, upper_bound)
        main_mask_inv = cv2.bitwise_not(main_mask)

        # Extract subject from main frame
        subject_fg = cv2.bitwise_and(frame, frame, mask=main_mask_inv)
        # Extract background from our combined (Color + Asteroid) frame
        final_bg = cv2.bitwise_and(dynamic_background, dynamic_background, mask=main_mask)
        
        final_output = cv2.add(subject_fg, final_bg)

        # Display result
        cv2.imshow(win_name, final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap_asteroid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_chroma_key(win_name='Chroma Key Pro')