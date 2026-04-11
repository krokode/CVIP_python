import cv2
import numpy as np

class InstagramFilters():
    def __init__(self, source, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.height = height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def apply_cartoon_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    def apply_cartoon_stylized_filter(self, frame):
        cartoon_stylized = cv2.stylization(frame, sigma_s=200, sigma_r=0.1)
        return cartoon_stylized

    def apply_pencil_sketch_filter(self, frame):
        # gray, color = cv2.pencilSketch(frame, sigma_s=100, sigma_r=0.05, shade_factor=0.08)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        pencilSketchImage = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return pencilSketchImage

    def start_filters(self, filter=None, sigma_s=None, sigma_r=None, shade_factor=None):
        filter_type = filter.lower() if filter else None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if filter_type == "cartoon":
                filtered_frame = self.apply_cartoon_filter(frame)
                cv2.imshow('Cartoon Filter', filtered_frame)
            elif filter_type == "cartoon_stylized":
                if sigma_s is not None and sigma_r is not None:
                    filtered_frame = cv2.stylization(frame, sigma_s=sigma_s, sigma_r=sigma_r)
                else:
                    filtered_frame = self.apply_cartoon_stylized_filter(frame)
                cv2.imshow('Cartoon Stylized Filter', filtered_frame)
            elif filter_type == "pencil":
                if sigma_s is not None and sigma_r is not None:
                    gray, color = cv2.pencilSketch(frame, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor if shade_factor else 0.08)
                    filtered_frame = gray
                else:
                    filtered_frame = self.apply_pencil_sketch_filter(frame)
                cv2.imshow('Pencil Sketch Filter', filtered_frame)
            else:
                cv2.imshow('Original', frame)
                print("No filter selected, showing original.")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

# Utility Class for Mouse Handling
# highGUI module and thus
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
        self.image = cv2.imread(image_path)

    def remove_seamlessly(self, xy, w, h):
        x, y = xy
        mask = 255 * np.ones(self.image[y:y+h, x:x+w].shape, self.image.dtype)
        center = (x + w // 2, y + h // 2)
        output = cv2.seamlessClone(self.image[y:y+h, x:x+w], self.image, mask, center, cv2.NORMAL_CLONE)
        self.image[y:y+h, x:x+w] = output[y:y+h, x:x+w]
        return self.image

    def remove_inpaint_blemish(self, xy, w, h):
        x, y = xy
        # Create a black mask the same size as the full image
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Draw a white circle or rectangle on the mask at the blemish location
        # Using a circle often yields smoother results for blemishes
        radius = max(w, h) // 2
        cv2.circle(mask, (x + w//2, y + h//2), radius, 255, -1)
        
        # Inpaint the entire image using that mask
        # Note: self.image and mask now have matching dimensions
        self.image = cv2.inpaint(self.image, mask, 3, cv2.INPAINT_NS)
        
        return self.image
    
# Chroma Keying
# class ChromaKey():
#     def __init__(self, video_source, background_image_path):
#         self.cap = cv2.VideoCapture(video_source)
#         self.background = cv2.imread(background_image_path)

#     def apply_chroma_key(self, lower_color, upper_color):
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             mask = cv2.inRange(hsv, lower_color, upper_color)
#             mask_inv = cv2.bitwise_not(mask)
#             fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
#             bg = cv2.bitwise_and(self.background, self.background, mask=mask)
#             combined = cv2.add(fg, bg)
#             cv2.imshow('Chroma Key', combined)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         self.cap.release()
#         cv2.destroyAllWindows()


if __name__ == "__main__":
    source = 0  # Change to video file path for video input
    instagram_filters = InstagramFilters(source)
    instagram_filters.start_filters_onvideo(filter="pencil")  # Change filter as needed

    # Example usage of Blemish removal
    # blemish_removal = Blemish("Lincoln.jpg")
    # mouse_handler = MouseHandler("Blemish Removal")

    # while True:
    #     cv2.imshow("Blemish Removal", blemish_removal.image)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
        
    #     if mouse_handler.points:
    #         # blemish_removal.remove_seamlessly(mouse_handler.points[-1], 15, 15)
    #         blemish_removal.remove_inpaint_blemish(mouse_handler.points[-1], 15, 15)

    
    # cv2.destroyAllWindows()
