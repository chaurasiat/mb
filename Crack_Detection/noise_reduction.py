import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

def apply_morphological_ops(sobel_output,kernel):

    #kernel = np.ones((5, 5), np.uint8)
    #erosion = cv2.erode(img, kernel, iterations=1)
    kernel2= np.ones((2,2), np.uint8)

    images = glob.glob('test_images/*.jpg')
    orignal_images=[]
    for fname in images:
        image = mpimg.imread(fname)
        orignal_images.append(image)

    x=0
    for image in sobel_output:
        dilate = cv2.dilate(image,kernel,iterations = 1)
        erosion = cv2.erode(dilate, kernel2, iterations=4)
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        f, (ax1, ax2,ax3,ax4,ax5,ax6) = plt.subplots(1, 6, figsize=(24, 7))
        f.tight_layout()

        ax1.imshow(orignal_images[x])
        ax1.set_title('original', fontsize=25)

        ax2.imshow(image, cmap='gray')
        ax2.set_title('Sobel_output', fontsize=25)

        ax3.imshow(erosion, cmap='gray')
        ax3.set_title('erosion', fontsize=25)

        ax4.imshow(dilate, cmap='gray')
        ax4.set_title('dilate', fontsize=25)

        ax5.imshow(closing, cmap='gray')
        ax5.set_title('closing', fontsize=25)

        ax6.imshow(dilate, cmap='gray')
        ax6.set_title('final', fontsize=25)

        x+=1

        plt.show()