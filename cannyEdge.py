import numpy as np
import cv2
import sys
import os

"""
    The Canny edge detection algorithm is composed of 5 steps:

1. Noise reduction;
2. Gradient calculation;
3. Non-maximum suppression;
4. Double threshold;
5. Edge Tracking by Hysteresis.
    
    """

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Noise reduction using gaussian
def gaussian_filter(image, kernel_size=7, sigma=1.5):
    kernel = np.zeros((kernel_size, kernel_size))
    mean = kernel_size // 2
    sum_val = 0.0
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = np.exp(-((x - mean) ** 2 + (y - mean) ** 2) / (2 * sigma ** 2))
            sum_val += kernel[x, y]
    kernel /= sum_val
    #print(kernel)
    padded_image = np.pad(image, pad_width=((mean, mean), (mean, mean)), mode='constant')
    #print(padded_image)
    smoothed_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            smoothed_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])
    return smoothed_image

# 2. gradients calculation
def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)
    
    G = np.hypot(Ix, Iy)  # Gradient magnitude
    G = G / G.max() * 255  # Normalize to 255
    theta = np.arctan2(Iy, Ix)
    return G, theta

# 3. Non-maximum suppression
def non_maximum_suppression(gradient_magnitude, theta):
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.float64)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]
                
                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass
    return Z

# 4. Double threshold
def double_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 75
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    zero_i, zero_j = np.where(image < low_threshold)
    
    image[strong_i, strong_j] = strong
    image[weak_i, weak_j] = weak
    image[zero_i, zero_j] = 0
    
    return image, weak, strong

# 5. Edge tracking by hysteresis
def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if image[i, j] == weak:
                if ((image[i+1, j-1:j+2] == strong).any() or
                    (image[i-1, j-1:j+2] == strong).any() or
                    (image[i, [j-1, j+1]] == strong).any()):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

# Canny edge detection function
def canny_edge_detection(image, low_threshold=30, high_threshold=100):
    gray_image = to_grayscale(image)
    # cv2.imshow("grey_image" , gray_image)
        
    smoothed_image = gaussian_filter(gray_image, kernel_size=7, sigma=1.5)
    #print(smoothed_image)
    
    gradient_magnitude, theta = sobel_filters(smoothed_image)
    

    non_max_suppressed = non_maximum_suppression(gradient_magnitude, theta)

    thresholded_image, weak, strong = double_threshold(non_max_suppressed, low_threshold, high_threshold)
    

    final_image = hysteresis(thresholded_image, weak, strong)
    
    return final_image


def main():
    if len(sys.argv) != 2:
        print("Usage: python canny_edge.py <image_file>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        sys.exit(1)
    
    edges = canny_edge_detection(image)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_edge.png"

    cv2.imwrite(output_filename, edges)
    print(f"Edge detected image saved as {output_filename}")


if __name__ == "__main__":
    main()

# To run this : python .\cannyEdge.py "C:\Users\Madhawa\Desktop\Machine Vision\cannyEdge\vk.jpg"
