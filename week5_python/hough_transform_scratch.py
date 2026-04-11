import numpy as np
import matplotlib.pyplot as plt


def plot_hough_lines(image, accumulator, rhos, thetas, threshold=100):
    plt.imshow(image, cmap='gray')
    peaks = np.where(accumulator > threshold)
    
    for i in range(len(peaks[0])):
        rho = rhos[peaks[0][i]]
        theta = thetas[peaks[1][i]]
        
        # Convert polar to cartesian for plotting
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend the lines far enough to cover the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        plt.plot([x1, x2], [y1, y2], 'r', alpha=0.5)
    
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.show()

def generate_test_edges(size=(200, 200), num_lines=3):
    # Create an empty black image (0 = background)
    edge_image = np.zeros(size, dtype=np.uint8)
    
    # Draw a horizontal line
    edge_image[50, 20:180] = 255
    
    # Draw a vertical line
    edge_image[20:180, 150] = 255
    
    # Draw a diagonal line (y = x)
    for i in range(40, 120):
        edge_image[i, i] = 255
        
    return edge_image


def generate_rand_lines_edges(size=(200, 200), num_lines=3):
    edge_image = np.zeros(size, dtype=np.uint8)
    
    for _ in range(num_lines):
        # Pick two random points (y, x)
        p1 = np.array([np.random.randint(0, size[0]), np.random.randint(0, size[1])])
        p2 = np.array([np.random.randint(0, size[0]), np.random.randint(0, size[1])])
        
        # Determine how many pixels we need to draw 
        # (the distance between points ensures no gaps)
        dist = int(np.linalg.norm(p1 - p2))
        
        # Linearly interpolate between the two points
        # this creates 'dist' number of points between p1 and p2
        y_coords = np.linspace(p1[0], p2[0], dist).astype(int)
        x_coords = np.linspace(p1[1], p2[1], dist).astype(int)
        
        # Draw the pixels (clamping to image bounds just in case)
        y_coords = np.clip(y_coords, 0, size[0] - 1)
        x_coords = np.clip(x_coords, 0, size[1] - 1)
        
        edge_image[y_coords, x_coords] = 255
        
    return edge_image


def generate_random_noise(size=(200, 200), density=0.01):
    """
    density: percentage of pixels that will be 'edges' (0.01 = 1%)
    """
    # Create random values between 0 and 1
    random_matrix = np.random.random(size)
    
    # Create a binary mask where True becomes 255 (edge)
    edge_image = (random_matrix < density).astype(np.uint8) * 255
    
    return edge_image


def hough_transform(edge_image, rho_res=1, theta_res=np.pi/180):
    """
    Simplified Hough Transform from scratch.
    edge_image: Binary image (2D array)
                Run a Canny or Sobel filter first (Hough Transform operates on binary edge images)
    rho_res: Distance resolution in pixels
    theta_res: Angle resolution in radians
    """
    height, width = edge_image.shape
    
    # Define the parameter space
    # Max possible distance is the diagonal of the image
    diagonal = np.sqrt(height**2 + width**2)
    rhos = np.arange(-diagonal, diagonal, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    
    # Initialize the Accumulator (rho x theta)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)
    
    # Find indices of edge pixels
    y_idxs, x_idxs = np.nonzero(edge_image)
    
    # Precompute cos and sin values for efficiency
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    # Vote!
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        
        for t_idx in range(len(thetas)):
            # Calculate rho for each theta
            rho = x * cos_t[t_idx] + y * sin_t[t_idx]
            
            # Find the closest rho index in our accumulator
            # We add 'diagonal' to map negative rhos to positive indices
            rho_idx = int((rho + diagonal) / rho_res)
            accumulator[rho_idx, t_idx] += 1
            
    return accumulator, rhos, thetas

if __name__=="__main__":
    
    # Generate the image
    # lines_img = generate_test_edges(size=(200, 200))
    lines_img = generate_rand_lines_edges(size=(200, 200), num_lines=4)
    noze_img = generate_random_noise(size=(200, 200), density=0.01)
    edge_img = np.maximum(lines_img, noze_img)

    # Run your Hough function
    accumulator, rhos, thetas = hough_transform(edge_img)
    dynamic_threshold = np.max(accumulator) * 0.5
    
    plot_hough_lines(image=edge_img, accumulator=accumulator, rhos=rhos, thetas=thetas, threshold=dynamic_threshold)
