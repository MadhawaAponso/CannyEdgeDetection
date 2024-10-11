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

