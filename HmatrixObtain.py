import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define paths and read images
images = os.path.expanduser('~/Desktop/Project/Scripts/IMGs/')
top_view_image_file = 'topView01.jpg'
top_view_image_path = os.path.join(images, top_view_image_file)
top_view_img = cv2.imread(top_view_image_path)

perspective_view_image_file = 'perspectiveView01.jpg'
perspective_view_image_path = os.path.join(images, perspective_view_image_file)
perspective_view_img = cv2.imread(perspective_view_image_path)

test_image_file = 'img_right12.jpg'
test_image_path = os.path.join(images, test_image_file)
test_img = cv2.imread(test_image_path)

# convert to grayscale
gray_top_view = cv2.cvtColor(top_view_img, cv2.COLOR_BGR2GRAY)
gray_perspective_view = cv2.cvtColor(perspective_view_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_img1 = cv2.GaussianBlur(gray_top_view, (11, 11), 1.6)
blurred_img2 = cv2.GaussianBlur(gray_perspective_view, (11, 11), 1.6)

# Checkerboard size 
checkerboard_size = (6, 5)  
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Detect chessboard corners
ret_top, corners_top = cv2.findChessboardCorners(gray_top_view, checkerboard_size, None)
ret_perspective, corners_perspective = cv2.findChessboardCorners(gray_perspective_view, checkerboard_size, None)

# Check if corners were found in both images
if ret_top and ret_perspective:
    #  corner positions
    corners_top = cv2.cornerSubPix(gray_top_view, corners_top, checkerboard_size, (-1, -1), criteria)
    corners_perspective = cv2.cornerSubPix(gray_perspective_view, corners_perspective, checkerboard_size, (-1, -1), criteria)

    # Draw  corners 
    cv2.drawChessboardCorners(top_view_img, checkerboard_size, corners_top, ret_top)
    cv2.drawChessboardCorners(perspective_view_img, checkerboard_size, corners_perspective, ret_perspective)

    # estinmate the homography matrix
    corners_top = corners_top.reshape(-1, 2)
    corners_perspective = corners_perspective.reshape(-1, 2)
    H, status = cv2.findHomography(corners_perspective, corners_top, cv2.RANSAC)
    print('Homography is:')
    print(H)

    # Print the homography matrix in LaTeX 
    print("Homography matrix in LaTeX format:")
    for row in H:
        latex_row = " & ".join(f"{value:.6f}" for value in row)
        print(f"{latex_row} \\\\")

    # Warp the test image using the homography
    width, height = gray_perspective_view.shape
    warped_perspective = cv2.warpPerspective(test_img, H, (height, width))

    # Display the images in a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.rc('axes', titlesize=9) 

    axs[0, 0].imshow(cv2.cvtColor(perspective_view_img, cv2.COLOR_BGR2GRAY),cmap='gray')
    axs[0, 0].set_title("Perspective View CheckerboardPattern")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(cv2.cvtColor(top_view_img, cv2.COLOR_BGR2GRAY),cmap='gray')
    axs[0, 1].set_title("Top View ChekerboardPattern")
    axs[0, 1].axis("off")
    
    axs[0, 0].imshow(perspective_view_img)
    axs[0, 0].set_title("Perspective View CheckerboardPattern")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(top_view_img)
    axs[0, 1].set_title("Top View ChekerboardPattern")
    axs[0, 1].axis("off")

    
    axs[1, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Test Image (Original)")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(cv2.cvtColor(warped_perspective, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Warped Test Image")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("Error: Could not find checkerboard corners in one or both images.")
