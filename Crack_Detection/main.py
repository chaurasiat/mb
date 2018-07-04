import numpy as np
import matplotlib.pyplot as plt
import cv2
from Crack_Detection import gradient_detection as gd
from Crack_Detection import noise_reduction as nd

import matplotlib.image as mpimg

# kernel = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# img=mpimg.imread("testimg.jpg")
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# plt.imshow(opening)
# plt.show()

sobel_output = gd.apply_sobel_operations(show_plots=True)
morph_output = nd.apply_morphological_ops(sobel_output,kernel)


