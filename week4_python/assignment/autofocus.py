# Import modules
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.interpolation'] = 'bilinear'

def var_abs_laplacian(image):
    """
    Implement Variance of absolute values of Laplacian - Method 1
    Input: image
    Output: Floating point number denoting the measure of sharpness of image
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = (1/6) * np.array([[0,-1,0],
                             [-1,4,1],
                             [0,-1,0]], 
                             np.float32)
    lap = cv2.filter2D(image.astype(np.float32), -1, kernel)
    res = np.abs(lap).var()
    return float(res)


def sum_modified_laplacian(im):
    """
    Implement Sum Modified Laplacian - Method 2
    Input: image
    Output: Floating point number denoting the measure of sharpness of image
    """
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)

    kx = np.array([[0, 0, 0],
                   [-1, 2, -1],
                   [0, 0, 0]], dtype=np.float32)

    ky = np.array([[0, -1, 0],
                   [0,  2, 0],
                   [0, -1, 0]], dtype=np.float32)

    lx = cv2.filter2D(im, -1, kx)
    ly = cv2.filter2D(im, -1, ky)

    ml = np.abs(lx) + np.abs(ly)

    return float(ml.sum())

def main(DATA_PATH, ROI):
    # Read input video filename
    filename = DATA_PATH + 'videos/focus-test.mp4'

    # Create a VideoCapture object
    cap = cv2.VideoCapture(filename)

    # Read first frame from the video
    ret, frame = cap.read()

    # Display total number of frames in the video
    print("Total number of frames : {}".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    
    maxV1 = 0
    maxV2 = 0

    # Frame with maximum measure of focus
    # Obtained using methods 1 and 2
    bestFrame1 = 0 
    bestFrame2 = 0 

    # Frame ID of frame with maximum measure
    # of focus
    # Obtained using methods 1 and 2
    bestFrameId1 = 0 
    bestFrameId2 = 0 

    # Specify the ROI for flower in the frame (top 10, left 420, bottom 650, right 1060)
    # UPDATE THE VALUES BELOW
    top = ROI[0]
    left = ROI[2]
    bottom = ROI[1]
    right = ROI[3]

    # Iterate over all the frames present in the video
    while(ret):
        # Crop the flower region out of the frame
        flower = frame[top:bottom, left:right]
        # Get measures of focus from both methods
        val1 = var_abs_laplacian(flower)
        val2 = sum_modified_laplacian(flower)
        
        #get the current frame
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        print("Frame: %d, VAR_LAP: %f, SML: %d" % (frame_id,val1,val2))
        
        # If the current measure of focus is greater 
        # than the current maximum
        if val1 > maxV1 :
            # Revise the current maximum
            maxV1 = val1
            # Get frame ID of the new best frame
            bestFrameId1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Revise the new best frame
            bestFrame1 = frame.copy()
            print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))

        # If the current measure of focus is greater 
        # than the current maximum
        if val2 > maxV2 : 
            # Revise the current maximum
            maxV2 = val2
            # Get frame ID of the new best frame
            bestFrameId2 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # Revise the new best frame
            bestFrame2 = frame.copy()
            print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))
            
        # Read a new frame
        ret, frame = cap.read()


    print("================================================")
    # Print the Frame ID of the best frame
    print("Frame ID of the best frame [Method 1]: {}".format(bestFrameId1))
    print("Frame ID of the best frame [Method 2]: {}".format(bestFrameId2))

    # Release the VideoCapture object
    cap.release()

    # Stack the best frames obtained using both methods
    out = np.hstack((bestFrame1, bestFrame2))

    # Display the stacked frames
    plt.figure()
    plt.imshow(out[:,:,::-1]);
    plt.axis('off');
    plt.show()

if __name__=="__main__":
    datapath = '../data/'
    # [top, bottom, left, right]
    ROI = [10, 650, 420, 1060]

    main(datapath, ROI)