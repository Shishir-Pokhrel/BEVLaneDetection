import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, morphology, filters
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes
import os 

calibration_flags = cv.CALIB_FIX_INTRINSIC

H = np.array([[2.88681721e+00, 4.69945408e+00, -1.20809081e+03],
              [-3.17056795e-01, 1.29358378e+01, -4.66821762e+03],
              [-2.62183378e-04, 7.07383918e-03, 1.00000000e+00]])
mtx1 = np.array([[1.49218452e+03, 0.00000000e+00, 6.03490449e+02],
                 [0.00000000e+00, 1.49610277e+03, 4.03113144e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist1 = np.array([[0.14043345, 0.02129057, -0.02553297, -0.01685553, -1.2258668]])
mtx2 = np.array([[1.51120008e+03, 0.00000000e+00, 6.48536573e+02],
                 [0.00000000e+00, 1.51645403e+03, 5.15486362e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist2 = np.array([[0.08486609, 0.10679398, 0.01633373, 0.00649924, 0.11807069]]) 
         
R = np.array([[0.99835944, -0.03553422, 0.04489709],
              [0.0430387, 0.9828851, -0.17912162],
              [-0.03776373, 0.18076007, 0.98280196]])
              
T = np.array([[-3.9336552], [-0.17502972], [0.46588329]])


def rectify_images(imgL, imgR, R, T, mtx1, dist1, mtx2, dist2):
    img1 = cv.undistort(imgL, mtx1, dist1)
    img2 = cv.undistort(imgR, mtx2, dist2)
   
    # Compute rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, (imgL.shape[1], imgL.shape[0]), R, T, alpha = 0, flags=calibration_flags)
    
    map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, (imgL.shape[1], imgL.shape[0]), cv.CV_16SC2)
    map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, (imgR.shape[1], imgR.shape[0]), cv.CV_16SC2)

    # Rectify the images
    rectified_img1 = cv.remap(imgL, map1x, map1y, cv.INTER_CUBIC)
    rectified_img2 = cv.remap(imgR, map2x, map2y, cv.INTER_CUBIC)
    
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
        
    return rectified_img1, rectified_img2, img1, img2, Q
    
def calculate_disparity_map(rectified_img1, rectified_img2):
    print('Disparity map calculation') # Remove negative disparity values     
    gaussianSize = (11,11)
    blurred_img1 = cv.GaussianBlur(rectified_img1, gaussianSize, 0)
    blurred_img2 = cv.GaussianBlur(rectified_img2, gaussianSize, 0)
    
    #leftG = cv.cvtColor(blurred_img1, cv.COLOR_BGR2GRAY)
    #rightG = cv.cvtColor(blurred_img2, cv.COLOR_BGR2GRAY)
    
    leftG = cv.cvtColor(rectified_img1, cv.COLOR_BGR2GRAY)
    rightG = cv.cvtColor(rectified_img2, cv.COLOR_BGR2GRAY)
    
    stereo01 = cv.StereoBM_create(numDisparities=16, blockSize=5)    
    stereo = cv.StereoSGBM_create(minDisparity=40, 
                                  numDisparities=64, 
                                  blockSize=5, 
                                  P1=1*1*13*15, P2=1*8*25*20, 
                                  disp12MaxDiff=1, 
                                  uniquenessRatio=10, 
                                  speckleWindowSize=16, speckleRange=132)

    disparity_map = stereo01.compute(leftG, rightG)
    #disparity_map = stereo.compute(blurred_img1, blurred_img2)
   
    disparity_map[disparity_map < 0] = 0 
    print('Disparity map calculation') # Remove negative disparity values  
          
    return disparity_map 
    
def process_binary_line_image(line_image):
    print('binary line image') # Remove negative disparity values     

    # Threshold to create binary image for lines
    _, binary_line_image = cv.threshold(line_image, 155, 255, cv.THRESH_BINARY)
    inverted_line_image = cv.bitwise_not(binary_line_image)  # Invert to make lines black (0) and background white (255)
    return inverted_line_image

def binarize_disparity_map(disparity_map):
    print('binarise disp map')
    # Create a binary version of the disparity map
    binary_disparity_map = np.where(disparity_map > 0, 1, 0).astype(np.uint8)  # Set positives to 1, others to 0
    return binary_disparity_map

def subtract_line_from_disparity(binary_disparity_map, inverted_line_image):
    #  both images have the same size?
    if binary_disparity_map.shape != inverted_line_image.shape:
        inverted_line_image = cv.resize(inverted_line_image, (binary_disparity_map.shape[1], binary_disparity_map.shape[0]))
    
    # both images are single-channel ???
    if len(binary_disparity_map.shape) > 2:
        binary_disparity_map = cv.cvtColor(binary_disparity_map, cv.COLOR_BGR2GRAY)
    if len(inverted_line_image.shape) > 2:
        inverted_line_image = cv.cvtColor(inverted_line_image, cv.COLOR_BGR2GRAY)
    
    # Convert both images to binary values (0 or 1) 
    binary_disparity_map = (binary_disparity_map > 0).astype(np.uint8)
    inverted_line_image = (inverted_line_image > 0).astype(np.uint8)
    
    # Perform the subtraction operation
    line_only_map = cv.subtract(binary_disparity_map, inverted_line_image)
    
    return line_only_map

def create_masked_disparity_map(disparity_map, line_only_map):
    # Multiply line-only map by constant, subtract from disparity map, and mask negatives
    mask = line_only_map * 300
    masked_disparity = disparity_map - mask
    masked_disparity[masked_disparity < 0] = 0  # Set any negatives to zero
    
    return masked_disparity
    
def overlay_disparity_on_image(img, masked_disparity_map, colormap='jet'):
    # Normalize the disparity map to a range of 0 to 1
    normalized_disparity = cv.normalize(masked_disparity_map, None, 55, 255, cv.NORM_MINMAX)

    # Apply the colormap 
    color_disparity_map = cv.applyColorMap(np.uint8(255 * normalized_disparity), cv.COLORMAP_JET)

    # Combine and color
    alpha = 0.75 # Transparency 
    overlay = cv.addWeighted(img, 1 - alpha, color_disparity_map, alpha, 0)

    return overlay

def plot_stereo_results(line_only_map, binary_disparity_map, rectified_img1, rectified_img2, inverted_line_image, masked_disparity_map, disparity_map):
  fig, axs = plt.subplots(5, 2, figsize=(8,12))
  plt.rcParams['font.size'] = 8  # Adjust the value as needed
  plt.rc('axes', titlesize=9) 
  plt.xticks(fontsize=5)
  plt.yticks(fontsize=5)
    
  axs[0, 0].imshow(imgL)
  axs[0, 0].set_title('Left Image')
  
  axs[0, 1].imshow(imgR)
  axs[0, 1].set_title('Right Image')
  
  axs[1, 0].imshow(cv.cvtColor(rectified_img1, cv.COLOR_BGR2RGB))
  axs[1, 0].set_title('Rectified Left Image')
  
  axs[1, 1].imshow(cv.cvtColor(rectified_img2, cv.COLOR_BGR2RGB))
  axs[1, 1].set_title('Rectified Right Image')
      
  axs[2, 0].imshow(inverted_line_image)
  axs[2, 0].set_title('Inverted Line Image')
    
  axs[2, 1].imshow(line_only_map)
  axs[2, 1].set_title('Line only map')
      
  axs[3, 0].imshow(disparity_map, cmap='jet')
  axs[3, 0].set_title('Disparity Map')
  
  axs[3, 1].imshow(masked_disparity_map, cmap='jet')
  axs[3, 1].set_title('Masked Disparity Map')
      
  axs[4, 0].imshow(overlayed_image)
  axs[4, 0].set_title('Masked Disparity Overlay')
   
  axs[4, 1].imshow(depth_map)
  axs[4, 1].set_title('Depth from MaskedDisparity')
    
  for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=5)

  img_height = rectified_img1.shape[0]
  num_lines = 4
  line_y_positions = [(i + 1) * img_height // (num_lines + 1) for i in range(num_lines)]
  for y in line_y_positions:
    for ax in axs.flat:
      ax.axhline(y=y, color='green', linestyle='--', linewidth=0.5)
      
  for i, ax in enumerate(axs.flat):
        # Show x-axis ticks only for subplots in the bottom row
        if i >= 8:  
            ax.set_xticks(np.arange(0, ax.get_xlim()[1], 300))  # intervals of 300
        else:
            ax.set_xticks([])  # Remove x ticks for other rows
        
        ax.tick_params(axis='both', which='major', labelsize=5)
      
  plt.tight_layout()
  plt.show()
 
 
def calculate_depth_map(disparity_map, baseline, focal_length):

    disparity_map = np.where(disparity_map == 0.0, np.nan, disparity_map)
    
    # Calculate the depth map (Z) from disparity (d) using formula Z = (f * B) / d
    depth_map = (baseline * focal_length) / disparity_map
    depth_map = np.nan_to_num(depth_map, nan=0.0)
    
    normalized_depth_map = cv.normalize(depth_map, None, 0, 1, cv.NORM_MINMAX)

    return depth_map, normalized_depth_map
 
 
#Location of the files
frames_folder = "/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImgg/synched/"
before_frames_folder = "/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImgg/"

#image from the two cameras
imgL = cv.imread(os.path.join(before_frames_folder, "img_left1.jpg"))
imgR= cv.imread(os.path.join(before_frames_folder, "img_right1.jpg"))

rectified_img1, rectified_img2, img1, img2, Q = rectify_images(imgL, imgR, R, T, mtx1, dist1, mtx2, dist2)
disparity_map = calculate_disparity_map(rectified_img1, rectified_img2)


imgRR = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)
gray_imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

inverted_line_image = process_binary_line_image(imgR)
binary_disparity_map = binarize_disparity_map(disparity_map)
line_only_map = subtract_line_from_disparity(binary_disparity_map, inverted_line_image)
masked_disparity_map = create_masked_disparity_map(disparity_map, line_only_map)
overlayed_image = overlay_disparity_on_image(imgR, masked_disparity_map)
depth_map, normalized_depth_map = calculate_depth_map(disparity_map, 0.39, 1500)
#plot_stereo_results(line_only_map, binary_disparity_map, rectified_img1, rectified_img2, 
                        # inverted_line_image, masked_disparity_map, disparity_map)

height, width = imgR.shape[:2]
output_size = (width, height)

# Apply the homography transformation
birdseye_frame_overlaid = cv.warpPerspective(overlayed_image, H, output_size)
birdseye_frame_original = cv.warpPerspective(imgR, H, output_size)

fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# Plot the images 
axs[0, 0].imshow(imgL)  # Use 'gray' for grayscale images
axs[0, 0].set_title('Left Image')
axs[0, 1].imshow(imgR)  # Use 'gray' for grayscale images
axs[0, 1].set_title('Right Image')
axs[1, 0].imshow(disparity_map, cmap='jet')
axs[1, 0].set_title('Disparity Map')
axs[1, 1].imshow(masked_disparity_map, cmap='jet')
axs[1, 1].set_title('Masked Disparity Map')

# Adjust layout and display
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot the images on the subplots
axs[0, 0].imshow(imgR)
axs[0, 0].set_title('Input Image')
axs[0, 1].imshow(birdseye_frame_original)
axs[0, 1].set_title('BEV of Image')
axs[1, 0].imshow(overlayed_image)
axs[1, 0].set_title('Overlaid Image')
axs[1, 1].imshow(birdseye_frame_overlaid)
axs[1, 1].set_title('BEV of Overlaid Image')

# Adjust layout and display
plt.tight_layout()
plt.show()

gOverlaidImage = cv.cvtColor(overlayed_image, cv.COLOR_BGR2GRAY)
_, bOverlaidImage = cv.threshold(gOverlaidImage, 137, 245, cv.THRESH_BINARY)

cv.waitKey(0)
cv.destroyAllWindows()

