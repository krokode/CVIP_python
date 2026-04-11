# Enter your code here
import cv2
import numpy as np

# Utility Class for Mouse Handling
class MouseHandler():
    def __init__(self, window_name):
        self.window_name = window_name
        self.points = []
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            #print(f"Point selected: ({x}, {y})")
            return (x, y)

# Blemish Removal 
class Blemish():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    def remove_blemish_seamlessly(self, xy, radius=15):
        x, y = xy
        h, w = self.image.shape[:2]

        # Define the source patch area (offset from the blemish)
        # We'll grab a patch from slightly to the left/top
        src_x = max(x - radius * 2, radius)
        src_y = max(y - radius * 2, radius)
        
        # Extract source patch
        source = self.image[src_y-radius:src_y+radius, src_x-radius:src_x+radius]
        
        # Create the mask (same size as the source patch)
        mask = np.zeros(source.shape, dtype=self.image.dtype)
        cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)

        # Apply Seamless Clone
        # The 'center' is where the source patch's center will be placed on self.image
        center = (x, y)

        self.image = cv2.seamlessClone(
            source,
            self.image,
            mask,
            center,
            cv2.NORMAL_CLONE
            )
        
        return self.image

    def remove_blemish_inpaint(self, xy, radius=15):
        x, y = xy
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Smaller, tighter mask
        radius = max(radius, 10) // 3
        cv2.circle(mask, (x, y), radius, 255, -1)

        # Small radius = texture preservation
        self.image = cv2.inpaint(
            self.image,
            mask,
            inpaintRadius=1,
            flags=cv2.INPAINT_NS
        )
        return self.image
    
if __name__ == "__main__":
    # Example usage of Blemish removal
    mouse_handler = MouseHandler("Blemish Removal")
    blemish_removal = Blemish("blemish.png")
    
    while True:
        cv2.imshow("Blemish Removal", blemish_removal.image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if mouse_handler.points:
            point = mouse_handler.points[-1]
            # blemish_removal.remove_blemish_seamlessly(point, radius=10)
            # Alternatively, use inpaint method
            blemish_removal.remove_blemish_inpaint(point, radius=15)
            mouse_handler.points.pop()  # Remove the processed point
            
    cv2.destroyAllWindows()
    cv2.imwrite("blemish_removed.png", blemish_removal.image)
