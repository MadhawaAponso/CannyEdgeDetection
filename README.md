# Canny Edge Detection in Python

This project implements the **Canny Edge Detection** algorithm using Python and OpenCV. The algorithm is composed of five steps:

1. Noise reduction using a Gaussian filter.
2. Gradient calculation using Sobel filters.
3. Non-maximum suppression to thin out edges.
4. Double threshold to identify strong, weak, and non-relevant edges.
5. Edge tracking by hysteresis to finalize edges.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)

## Installation

To run this project, make sure you have Python installed along with the required libraries:

```bash
pip install numpy opencv-python

```

To run this code ,
```bash
python cannyEdge.py <your_image_path>
```
# The original Image
![alt text](https://github.com/MadhawaAponso/CannyEdgeDetection/blob/main/vk.jpg?raw=true)

# The cannyEdge detected image
![alt text](https://github.com/MadhawaAponso/CannyEdgeDetection/blob/main/vk_edge.png?raw=true)

# Canny Edge Detection Algorithm
The Canny edge detection algorithm is a popular technique used in image processing for detecting edges within images. Developed by John F. Canny in 1986, it aims to identify significant transitions in intensity, which often correspond to the boundaries of objects. The algorithm is widely employed due to its effectiveness in detecting edges while minimizing noise, making it an essential tool in various computer vision applications.

Canny edge detection is utilized in a range of use cases, including object detection, image segmentation, and feature extraction. It is particularly beneficial in scenarios where precise edge localization is crucial, such as in autonomous driving systems for detecting lanes, obstacles, and road signs. Additionally, it plays a vital role in medical imaging, industrial inspection, and robotics, helping in the analysis of structures and shapes.

The Canny algorithm operates in five distinct steps. First, it applies a Gaussian filter to the image for noise reduction. Next, it computes the gradient magnitude and direction using Sobel filters, which helps identify areas with high intensity changes. Afterward, non-maximum suppression is applied to thin out the edges by retaining only the local maxima in the gradient direction. A double threshold is then used to classify the edges into strong, weak, and non-relevant categories. Finally, edge tracking by hysteresis is performed to connect weak edges to strong ones, ensuring a complete and accurate edge map. This systematic approach makes the Canny edge detector one of the most robust methods available for edge detection in images.

source : https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123



