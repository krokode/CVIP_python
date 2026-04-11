import cv2
import numpy as np

def assignmentDilation(im):
    # Creating kernel
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    print(element)

    # Creating size regarding variables
    ksize = element.shape[0]
    height,width = im.shape[:2]
    border = ksize//2
    print(f"ksize = {ksize} | height, width = ({height, width}) | border = {border}")

    # Creating border(likeframe) on image to make it bigger for border size, color of frame is black value = 0
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
    paddedDilatedIm = paddedIm.copy()
    print(f"Copy of paddedIm = {paddedDilatedIm}")

    # Create a VideoWriter object
    # Use frame size as 50x50
    ###
    frame_width = 50
    frame_height = 50
    # Define codec 
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename="dilationScratch.avi", fourcc=fourcc, fps=10.0, frameSize=(frame_width, frame_height))
    ###

    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            # Extract region from padded image
            step_im = paddedIm[h_i-1:h_i+2, w_i-1:w_i+2]
            if np.any(step_im[element == 1]):
                paddedDilatedIm[h_i, w_i] = 1

            ###
            # Resize output to 50x50 before writing it to the video
            ###
            frame = paddedDilatedIm[border:-border, border:-border]
            frame = cv2.resize(frame, (50, 50), interpolation=cv2.INTER_NEAREST)
            ###
            # Convert resizedFrame to BGR before writing
            ###
            frame = (frame * 255).astype(np.uint8)  # scale to 0–255
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
            ###

    # Release the VideoWriter object
    ###
    out.release()
    cv2.destroyAllWindows()
    ###

    # Display final image (cropped)
    ###
    cap = cv2.VideoCapture("dilationScratch.avi")
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")
    
    cv2.namedWindow("Custom Dilation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Custom Dilation", 600, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        
        cv2.imshow("Custom Dilation", frame)

        # 100 ms between frames ≈ 10 FPS
        if cv2.waitKey(100) & 0xFF == 27:  # ESC to quit early
            break

    cap.release()
    cv2.destroyAllWindows() 
    ###


def assignmentErode(im):
    # Creating kernel
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    print(element)

    # Creating size regarding variables
    ksize = element.shape[0]
    height,width = im.shape[:2]
    border = ksize//2
    print(f"ksize = {ksize} | height, width = ({height, width}) | border = {border}")

    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 1)
    paddedErodedIm = paddedIm.copy()
    # Create a VideoWriter object
    # Use frame size as 50x50
    ###
    frame_width = 50
    frame_height = 50
    # Define codec 
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(filename="erosionScratch.avi", fourcc=fourcc, fps=10.0, frameSize=(frame_width, frame_height))
    ###
    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            ###
            # Extract region from padded image
            step_im = paddedIm[h_i-1:h_i+2, w_i-1:w_i+2]
            if np.all(step_im[element == 1] == 1):
                paddedErodedIm[h_i, w_i] = 1
            else:
                paddedErodedIm[h_i, w_i] = 0
            ###
            # Resize output to 50x50 before writing it to the video
            ###
            frame = paddedErodedIm[border:-border, border:-border]
            frame = cv2.resize(frame, (50, 50), interpolation=cv2.INTER_NEAREST)
            ###
            # Convert resizedFrame to BGR before writing
            ###
            frame = (frame * 255).astype(np.uint8)  # scale to 0–255
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
            ###
    # Release the VideoWriter object
    ###
    out.release()
    cv2.destroyAllWindows()
    ###
    # Display final image (cropped)
    ###
    cap = cv2.VideoCapture("erosionScratch.avi")
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")
    
    cv2.namedWindow("Custom Erossion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Custom Erossion", 600, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video
        
        cv2.imshow("Custom Erossion", frame)

        # 100 ms between frames ≈ 10 FPS
        if cv2.waitKey(100) & 0xFF == 27:  # ESC to quit early
            break

    cap.release()
    cv2.destroyAllWindows() 
    ###

def dilate_custom(im, element, iterrations=1):
    # Convert standard grayscale 0-255 image to 0-1 binary image
    # Note: Assumes im.dtype is np.uint8 or similar where 255 is max
    if im.dtype != np.bool_ and np.max(im) > 1:
        im = (im > 0).astype(np.uint8) 
        # Now white is 1, black is 0
    # Creating size regarding variables
    ksize = element.shape[0]
    height,width = im.shape[:2]
    odd = (ksize % 2) > 0
    half_element = ksize//2
    border = half_element
    another_half = (half_element + 1) if odd else half_element
    # Creating border(likeframe) on image to make it bigger for border size, color of frame is black value = 0
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
    paddedDilatedIm = paddedIm.copy()
    for iter in range(iterrations):
        if iter > 0:
            paddedIm = paddedDilatedIm.copy()     
        for h_i in range(border, height+border):
            for w_i in range(border,width+border):
                # Extract region from padded image
                step_im = paddedIm[h_i-half_element:h_i+another_half, w_i-half_element:w_i+another_half]
                if np.any(step_im[element == 1]):
                    paddedDilatedIm[h_i, w_i] = 1
        iter += 1
    dilated = paddedDilatedIm[border:-border, border:-border]
    return dilated


def erode_custom(im, element, iterrations=1):
    # Convert standard grayscale 0-255 image to 0-1 binary image
    # Note: Assumes im.dtype is np.uint8 or similar where 255 is max
    if im.dtype != np.bool_ and np.max(im) > 1:
        im = (im > 0).astype(np.uint8) 
        # Now white is 1, black is 0
    # Creating size regarding variables
    ksize = element.shape[0]
    height,width = im.shape[:2]
    odd = (ksize % 2) > 0
    half_element = ksize//2
    border = half_element
    another_half = (half_element + 1) if odd else half_element
    # Creating border(likeframe) on image to make it bigger for border size, color of frame is white value = 1
    
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 1)
    current_im = paddedIm.copy()
    
    for _ in range(iterrations): 
        # Create a fresh, BLACK output buffer (height+2*border, width+2*border) 
        # This prevents accidental retention of pixels
        next_im = np.zeros(current_im.shape, dtype=current_im.dtype) 
        
        # For the unpadded output, copy the current input's WHITE border onto the BLACK buffer
        # This keeps the necessary white frame for boundary checks in the next iteration
        next_im = cv2.copyMakeBorder(next_im[border:-border, border:-border], 
                                     border, border, border, border, 
                                     cv2.BORDER_CONSTANT, value = 1)
        
        for h_i in range(border, height+border):
            for w_i in range(border,width+border):
                # Extract region from current input image
                step_im = current_im[h_i-half_element:h_i+another_half, w_i-half_element:w_i+another_half]
                
                # Erosion Condition: The center pixel (h_i, w_i) is set to 1 
                # ONLY IF all pixels covered by the element (value 1) are also 1 in the input neighborhood.
                if np.all(step_im[element == 1] == 1):
                    # Write 1 (Foreground) to the output buffer
                    next_im[h_i, w_i] = 1
                # ELSE: The pixel remains 0 (Background) due to the initialization of next_im

        # The result (next_im) becomes the input (current_im) for the next iteration
        current_im = next_im

    eroded = current_im[border:-border, border:-border]
    return eroded

def openning_morph(im, kernel, iter):
    eroded = erode_custom(im=im, element=kernel, iterrations=iter)
    openning = dilate_custom(eroded, element=kernel, iterrations=iter)
    return openning

def closing_morph(im, kernel, iter):
    dilated = dilate_custom(im, element=kernel, iterrations=iter)
    closing = erode_custom(im=dilated, element=kernel, iterrations=iter)
    return closing

if __name__=="__main__":
    # additional imports
    from dataPath import DATA_PATH
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    matplotlib.rcParams['image.cmap'] = 'gray'

    # Creating black image
    im = np.zeros((10,10),dtype='uint8')

    # Adding blobs to image
    im[0,1] = 1
    im[-1,0]= 1
    im[-2,-1]=1
    im[2,2] = 1
    im[5:8,5:8] = 1
    
    plt.imshow(im)
    plt.title("Blobed image")
    plt.show()

    assignmentDilation(im)
    assignmentErode(im)

    imageName = DATA_PATH + "images/opening.png"
    # Image taken as input
    image = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    # Check for invalid input
    if image is None:  
        print("Could not open or find the image")
    

    # Specify Kernel Size
    kernelSize = 10

    # Create the Kernel
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1, 2*kernelSize+1),
                                        (kernelSize, kernelSize))
    
    iter = 3

    plt.figure(figsize=[15, 15])
    plt.subplot(151)
    plt.imshow(image)
    plt.title("Original")

    dilated = dilate_custom(im=image, element=element, iterrations=iter)
    plt.subplot(152)
    plt.imshow(dilated)
    plt.title(f"Dilated {iter}(times)")

    eroded = erode_custom(im=image, element=element, iterrations=iter)
    plt.subplot(153)
    plt.imshow(eroded)
    plt.title(f"Eroded {iter}(times)")

    open_morph = openning_morph(im=image, kernel=element, iter=iter)
    plt.subplot(154)
    plt.imshow(open_morph)
    plt.title(f"Openning morph {iter}(times)")

    clos_morph = closing_morph(im=image, kernel=element, iter=iter)
    plt.subplot(155)
    plt.imshow(clos_morph)
    plt.title(f"Closing morph {iter}(times)")

    plt.show()