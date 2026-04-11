import cv2
import numpy as np

def imread_custom(image_path, options=cv2.IMREAD_UNCHANGED):
    """
    Read an image from a file path.
    Advantages:
        - Handles file paths with non-ASCII characters.
        - More robust than cv2.imread for certain file path issues.
        - Supports various image formats.
        - Returns images in BGR format as used by OpenCV.
        - Longer file paths are supported on Windows systems.
        - Can read images from network drives or special file systems.
        - Almost equivalent performance to cv2.imread.
    Args:
        image_path (str or Path): The path to the image file.
        options (int): The OpenCV image reading options.
    Returns:
        numpy.ndarray: The image as a NumPy array.
    """
    img_array = np.fromfile(str(image_path), dtype=np.uint8)

    # If file read failed, return None
    if img_array.size == 0:
        return None

    # Decode the array directly
    img = cv2.imdecode(img_array, options)
    return img

class InstagramFilters():
    """
    A class that applies various Instagram-like filters to images and videos using OpenCV.
    This class supports real-time video processing from a webcam or video file, as well as static image processing. It includes filters such as cartoon, cartoon stylization, pencil sketch, skin smoothing, and a fun sunglasses overlay. The class is designed to be flexible and efficient, with pre-loaded Haar Cascades for face and eye detection to optimize performance during video processing.
    Attributes:
        face_cascade (cv2.CascadeClassifier): Haar Cascade for face detection.
        eye_cascade (cv2.CascadeClassifier): Haar Cascade for eye detection.
        cap (cv2.VideoCapture): Video capture object for webcam or video file.
        image (numpy.ndarray): Loaded image for static processing.
        width (int): Width of the video frames or loaded image.
        height (int): Height of the video frames or loaded image.
        is_image (bool): Flag to indicate if the source is an image or video.
    Methods:
        apply_cartoon_filter(frame): Applies a cartoon filter to the input frame.
        apply_cartoon_stylized_filter(frame): Applies a cartoon stylization filter to the input frame.
        apply_pencil_sketch_filter(frame): Applies a pencil sketch filter to the input frame.
        apply_skin_smoothing_filter(frame): Applies a skin smoothing filter to the input frame.
        apply_sunglasses_filter(frame): Applies a sunglasses filter to the input frame.
        start_filters_onvideo(filter, sigma_s, sigma_r, shade_factor): Starts the video stream and applies the selected filter in real-time.
        start_filters_onimage(filter, sigma_s, sigma_r, shade_factor): Applies the selected filter to the image and displays the result.
    """
    def __init__(self, glasses_path, source, width=640, height=480):
        # Load Haar Cascades once during initialization to save performance during video loop
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        if glasses_path:
            self.glasses = imread_custom(glasses_path)
        else:
            self.glasses = None

        if isinstance(source, int) or source.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
            self.width = width
            self.height = height
            self.is_image = False
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif source.lower() in ['webcam', 'camera']:
            self.width = width
            self.height = height
            self.is_image = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif source.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'webp']:
            self.image = imread_custom(source)
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
            self.is_image = True
        else:
            raise ValueError("Unsupported source type. Please provide a valid video file, webcam, or image file.")

    def apply_cartoon_filter(self, frame):
        """
        Apply a cartoon filter to the input frame.
        This method converts the input frame to grayscale, applies a median blur, and then uses adaptive thresholding to create a cartoon effect by combining the edges with a bilateral filter for color smoothing.
        Args:            frame (numpy.ndarray): The input image frame (BGR format).
        Returns:            numpy.ndarray: The output image frame with the cartoon effect applied.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    def apply_cartoon_stylized_filter(self, frame):
        """
        Apply a cartoon stylization filter to the input frame.
        This method uses OpenCV's built-in stylization function to create a cartoon-like effect by smoothing the image while preserving edges.
        Args:            frame (numpy.ndarray): The input image frame (BGR format).
        Returns:            numpy.ndarray: The output image frame with the cartoon stylization effect applied.
        """
        cartoon_stylized = cv2.stylization(frame, sigma_s=200, sigma_r=0.1)
        return cartoon_stylized

    def apply_pencil_sketch_filter(self, frame):
        """
        Apply a pencil sketch filter to the input frame.
        This method converts the input frame to grayscale, applies a Gaussian blur, and then uses adaptive thresholding to create a pencil sketch effect.
        Args:            frame (numpy.ndarray): The input image frame (BGR format).
        Returns:            numpy.ndarray: The output image frame with the pencil sketch effect applied.
        """
        # gray, color = cv2.pencilSketch(frame, sigma_s=100, sigma_r=0.05, shade_factor=0.08)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        pencilSketchImage = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return pencilSketchImage

    def apply_skin_smoothing_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        result = frame.copy()

        for (x, y, w, h) in faces:
            # Create a soft elliptical mask to avoid smoothing hair/background
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.4), int(h*0.5)), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0 # Soft edges

            roi = result[y:y+h, x:x+w]
            
            # Strong smoothing that preserves edges (Bilateral)
            smoothed_roi = cv2.bilateralFilter(roi, 15, 80, 80)

            # Blend smoothed version with original based on the mask
            for c in range(3):
                roi[:, :, c] = (smoothed_roi[:, :, c] * mask + roi[:, :, c] * (1 - mask)).astype(np.uint8)

            result[y:y+h, x:x+w] = roi
                
        return result

    def _create_sunglasses_overlay(self, width, height):
        """Programmatically creates a transparent sunglasses overlay to avoid missing file errors"""
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Draw left lens
        cv2.ellipse(overlay, (width//4, height//2), (width//5, height//2), 0, 0, 360, (0, 0, 0, 220), -1)
        # Draw right lens
        cv2.ellipse(overlay, (3*width//4, height//2), (width//5, height//2), 0, 0, 360, (0, 0, 0, 220), -1)
        # Draw bridge connecting lenses
        cv2.line(overlay, (width//4 + width//5, height//2), (3*width//4 - width//5, height//2), (0, 0, 0, 220), max(2, height//10))
        # Draw arms of glasses
        cv2.line(overlay, (0, height//2), (width//4 - width//5, height//2), (0, 0, 0, 220), max(2, height//10))
        cv2.line(overlay, (3*width//4 + width//5, height//2), (width, height//2), (0, 0, 0, 220), max(2, height//10))
        
        return overlay
    
    def apply_sunglasses_filter(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
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
            if self.glasses is not None:
                sunglasses = cv2.resize(self.glasses, (w_g, h_g))
            else:
                sunglasses = self._create_sunglasses_overlay(w_g, h_g)
            
            alpha = sunglasses[:, :, 3] / 255.0
            alpha = alpha * 0.5
            
            for c in range(3):
                result[start_y:end_y, start_x:end_x, c] = (
                    alpha * sunglasses[:, :, c] + (1.0 - alpha) * result[start_y:end_y, start_x:end_x, c]
                ).astype(np.uint8)
                    
        return result    

    def start_filters_onvideo(self, filter=None, sigma_s=None, sigma_r=None, shade_factor=None):
        """
        Start the video stream and apply the selected filter in real-time.
        This method captures video frames from the webcam or video file, applies the specified filter to each frame, and displays the result in a window. The user can exit the video stream by pressing the 'q' key.
        Args:           filter (str): The name of the filter to apply. 
                                      Supported values are "cartoon", "cartoon_stylized", "pencil", "skin", and "sunglasses". If None, no filter is applied and the original video stream is shown.
                        sigma_s (float): Optional parameter for stylization and pencil sketch filters to control the amount of smoothing. Higher values result in a more pronounced effect.
                        sigma_r (float): Optional parameter for stylization and pencil sketch filters to control the range of colors that are smoothed together. Higher values result in more colors being blended.
                        shade_factor (float): Optional parameter for pencil sketch filter to control the intensity of the shading. Higher values result in darker shading.
        Returns:         None   
        """
        if not self.is_image:
            filter_type = filter.lower() if filter else None
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for a mirror effect (standard for webcam)
                frame = cv2.flip(frame, 1) 
                
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
                elif filter_type == "skin":
                    filtered_frame = self.apply_skin_smoothing_filter(frame)
                    cv2.imshow('Skin Smoothing Filter', filtered_frame)
                elif filter_type == "sunglasses":
                    filtered_frame = self.apply_sunglasses_filter(frame)
                    cv2.imshow('Sunglasses Filter', filtered_frame)
                else:
                    cv2.imshow('Original', frame)
                    # print("No filter selected, showing original.")
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            print("The provided source is an image. Please use start_filters_onimage for image sources.")


    def start_filters_onimage(self, filter=None, sigma_s=None, sigma_r=None, shade_factor=None):
        """
        Apply the selected filter to the image and display the result.
        This method applies the specified filter to the loaded image and displays it in a window. The user can close the window to exit.
        Args:           filter (str): The name of the filter to apply.
                                      Supported values are "cartoon", "cartoon_stylized", "pencil", "skin", and "sunglasses". If None, no filter is applied and the original image is shown.
                        sigma_s (float): Optional parameter for stylization and pencil sketch filters to control the amount of smoothing. Higher values result in a more pronounced effect.
                        sigma_r (float): Optional parameter for stylization and pencil sketch filters to control the range of colors that are smoothed together. Higher values result in more colors being blended.
                        shade_factor (float): Optional parameter for pencil sketch filter to control the intensity of the shading. Higher values result in darker shading.
        Returns:         None
        """
        if self.is_image:
            filter_type = filter.lower() if filter else None
            if filter_type == "cartoon":
                filtered_image = self.apply_cartoon_filter(self.image)
                cv2.imshow('Cartoon Filter', filtered_image)
            elif filter_type == "cartoon_stylized":
                if sigma_s is not None and sigma_r is not None:
                    filtered_image = cv2.stylization(self.image, sigma_s=sigma_s, sigma_r=sigma_r)
                else:
                    filtered_image = self.apply_cartoon_stylized_filter(self.image)
                cv2.imshow('Cartoon Stylized Filter', filtered_image)
            elif filter_type == "pencil":
                if sigma_s is not None and sigma_r is not None:
                    gray, color = cv2.pencilSketch(self.image, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor if shade_factor else 0.08)
                    filtered_image = gray
                else:
                    filtered_image = self.apply_pencil_sketch_filter(self.image)
                cv2.imshow('Pencil Sketch Filter', filtered_image)
            elif filter_type == "skin":
                filtered_image = self.apply_skin_smoothing_filter(self.image)
                cv2.imshow('Skin Smoothing Filter', filtered_image)
            elif filter_type == "sunglasses":
                filtered_image = self.apply_sunglasses_filter(self.image)
                cv2.imshow('Sunglasses Filter', filtered_image)
            else:
                cv2.imshow('Original Image', self.image)
                print("No filter selected, showing original image.")
                
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("The provided source is not an image. Please use start_filters_onvideo for video sources.")

# Blemish Removal 
class Blemish():
    def __init__(self, image_path):
        self.image = imread_custom(image_path)

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


if __name__ == '__main__':
    video_source = 0  # Use 0 for webcam, or replace with video file path like 'video.mp4'
    glusses_path = "sunglass.png"

    insta = InstagramFilters(glusses_path, video_source)
    
    # Change the string to test different features: 
    filters_list = ['cartoon', 'cartoon_stylized', 'pencil', 'skin', 'sunglasses', None]
    selected_filter = filters_list[4] 
    
    print(f"Applying {selected_filter} filter. Press 'q' to quit.")
    insta.start_filters_onvideo(filter=selected_filter)