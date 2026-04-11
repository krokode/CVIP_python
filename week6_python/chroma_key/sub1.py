import cv2
import numpy as np

# Global variables
picked_color = [0, 0, 255] # Default Red (BGR)
color_picker_enabled = True

def nothing(x):
    pass

def create_color_palette(width=300, height=300):
    """Generates a Photoshop-like HSV color gradient."""
    hsv_palette = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # Hue 0-179, Saturation 0-255, Value 255
            hue = int((x / width) * 179)
            sat = int((y / height) * 255)
            hsv_palette[y, x] = [hue, sat, 255]
    
    return cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)

def palette_callback(event, x, y, flags, param):
    """Captures color from the palette window."""
    global picked_color
    if event == cv2.EVENT_LBUTTONDOWN:
        palette_img = param
        picked_color = palette_img[y, x].tolist()

def run_chroma_key(win_name='Main Output', palette_name='Color Selector'):
    global picked_color

    # Initialize Windows
    cv2.namedWindow(win_name)
    cv2.namedWindow(palette_name)

    # Create and setup the Palette
    palette_img = create_color_palette()
    cv2.imshow(palette_name, palette_img)
    cv2.setMouseCallback(palette_name, palette_callback, palette_img)

    # Video Sources
    cap = cv2.VideoCapture("greenscreen-demo.mp4")
    cap_asteroid = cv2.VideoCapture("greenscreen-asteroid.mp4")

    win_size = (640, 480)
    bg_canvas = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        ret_ast, frame_ast = cap_asteroid.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if not ret_ast:
            cap_asteroid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ast, frame_ast = cap_asteroid.read()

        # Processing
        frame = cv2.resize(frame, win_size)
        frame_ast = cv2.resize(frame_ast, win_size)
        bg_canvas[:] = picked_color

        # 1. Overlay Asteroid on Picked Color
        ast_hsv = cv2.cvtColor(frame_ast, cv2.COLOR_BGR2HSV)
        ast_mask = cv2.inRange(ast_hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        ast_mask_inv = cv2.bitwise_not(ast_mask)
        
        bg_combined = cv2.bitwise_and(bg_canvas, bg_canvas, mask=ast_mask)
        fg_asteroid = cv2.bitwise_and(frame_ast, frame_ast, mask=ast_mask_inv)
        dynamic_background = cv2.add(bg_combined, fg_asteroid)

        # 2. Final Chroma Key
        main_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Standard Green Range
        main_mask = cv2.inRange(main_hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        main_mask_inv = cv2.bitwise_not(main_mask)

        subject_fg = cv2.bitwise_and(frame, frame, mask=main_mask_inv)
        final_bg = cv2.bitwise_and(dynamic_background, dynamic_background, mask=main_mask)
        
        final_output = cv2.add(subject_fg, final_bg)

        # UI Feedback: Show a small square of the picked color on the main window
        cv2.rectangle(final_output, (10, 10), (60, 60), picked_color, -1)
        cv2.putText(final_output, "Active BG", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win_name, final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cap_asteroid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_chroma_key()