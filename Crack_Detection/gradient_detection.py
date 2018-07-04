import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh_min=0, thresh_max=255):

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #because iam using cv2 to read

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    else:
        gradient = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(gradient)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    binary_output = np.copy(sxbinary)

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # because iam using cv2 to read
    abs_sobel_x=np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel))
    abs_sobel_y=np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))

    grad_direction = np.arctan2(abs_sobel_y, abs_sobel_x)

    sxbinary = np.zeros_like(grad_direction)
    sxbinary[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 3) Calculate the magnitude
    grad_magnitude = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary


def test_gradients(orient='x'):

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images = glob.glob('test_images/*.jpg')
    for fname in images:
        image = cv2.imread(fname)
        grad_binary = abs_sobel_thresh(image, orient=orient, thresh_min=50, thresh_max=150)

        #plot
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(grad_binary, cmap='gray')
        ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

def test_directions():

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images = glob.glob('test_images/*.jpg')
    for fname in images:
        image = mpimg.imread(fname)
        dir_binary = dir_threshold(image, sobel_kernel=5, thresh=(0.7,1.4))

        #plot
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('original Image', fontsize=50)
        ax2.imshow(dir_binary, cmap='gray')
        ax2.set_title('dir_thresholded', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

def apply_sobel_operations(show_plots=False):

    # Choose a Sobel kernel size
    ksize = 17  # Choose a larger odd number to smooth gradient measurements
    x_gradient_threshold = (50, 150)
    y_gradient_threshold = (50, 150)
    direction_threshold = (0.6, 1.4)
    magnitude_threshold = (70, 150)

    output_images=[]

    images = glob.glob('test_images/*.jpg')
    for fname in images:

        image = mpimg.imread(fname)
        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=x_gradient_threshold[0],thresh_max=x_gradient_threshold[1])
        grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=y_gradient_threshold[0],thresh_max=y_gradient_threshold[1])

        mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=magnitude_threshold)
        dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=direction_threshold)

        combined = np.zeros_like(dir_binary)
        #combined[(gradx == 1) & (dir_binary == 1)] = 1
        #combined[((gradx == 1) ) | ((dir_binary == 1) & (mag_binary==1))] = 1
        #combined[((gradx == 1) | (grady==1)) | ((dir_binary == 1) & (mag_binary == 1))] = 1
        combined[((gradx == 1) | (grady == 1))  & (mag_binary == 1)] = 1

        #plot
        if show_plots == True:
            f, (ax1, ax2, ax3, ax4, ax5 ) = plt.subplots(1, 5, figsize=(24, 9))
            f.tight_layout()

            ax1.imshow(image)
            ax1.set_title('original', fontsize=25)

            ax2.imshow(gradx, cmap='gray')
            ax2.set_title('gradx', fontsize=25)

            ax3.imshow(grady, cmap='gray')
            ax3.set_title('grady', fontsize=25)

            ax4.imshow(mag_binary, cmap='gray')
            ax4.set_title('mag', fontsize=25)

            #ax4.imshow(dir_binary, cmap='gray')
            #ax4.set_title('dir', fontsize=25)

            ax5.imshow(combined, cmap='gray')
            ax5.set_title('comb', fontsize=25)

            output_images.append(combined)

            plt.savefig('output_images/gradient_analysis/'+fname.split('\\')[1])
        #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return output_images


#test_gradients()
#test_directions()
#combinations()