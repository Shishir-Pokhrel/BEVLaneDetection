import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

rows = 6 #number of checkerboard rows.
columns = 5 #number of checkerboard columns.
world_scaling = 1 #change this to the real world square size. Or not.
convSize = (11,11) #size of the convolution matrix, will be used to in chessboard corners.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
calibration_flags = cv.CALIB_FIX_INTRINSIC

def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
 
    #coordinates of squares in  checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    imgpoints = [] # 2d points in image plane.
    objpoints = [] # 3d point in real world space
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
  
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, convSize, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            #cv.imshow('img', frame)
            #k = cv.waitKey(800)
 
            objpoints.append(objp)
            imgpoints.append(corners)
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    
    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    image_files = os.listdir(frames_folder)
    image_files.sort()

    num_images = len(image_files)
    mid_index = num_images // 2

    c1_images_names = image_files[:mid_index]
    c2_images_names = image_files[mid_index:]

    c1_images = []
    c2_images = []

    for im1, im2 in zip(c1_images_names, c2_images_names):
        # where are the images??
        full_path_im1 = os.path.join(frames_folder, im1)
        full_path_im2 = os.path.join(frames_folder, im2)

        # Here are the images
        img1 = cv.imread(full_path_im1)
        img2 = cv.imread(full_path_im2)

        # Determine the smaller image size
        min_height = max(img1.shape[0], img2.shape[0])
        min_width = max(img1.shape[1], img2.shape[1])

        # resize  big image to match smalls size
        if img1.shape[0] > min_height or img1.shape[1] > min_width:
            img1 = cv.resize(img1, (min_width, min_height))

        # Append the resized images 
        c1_images.append(img1)
        c2_images.append(img2)
     
    #coordinates of squares in the checkerboard  
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame 
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    imgpoints_left = [] # 2d points in image 
    imgpoints_right = []
    objpoints = [] # 3d points of checkerboard in world
 
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (6, 5), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (6, 5), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, convSize, (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, convSize, (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1,(6,5), corners1, c_ret1) 
            cv.drawChessboardCorners(frame2, (6,5), corners2, c_ret2)
             
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
  
    ret, CM1, distt1, CM2, distt2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, 
                                                                   mtx1, dist1,mtx2, dist2,(width, height), 
                                                                   criteria = criteria, flags = calibration_flags)
    
    print('Reprojection Error: Stereo ' , ret)
    print('RotationStereo', R)
    print('TranslationStreo', T)
      
    return R, T
 
def rectify_images(imgL, imgR, R, T, mtx1, dist1, mtx2, dist2):
    img1 = cv.undistort(imgL, mtx1, dist1)
    img2 = cv.undistort(imgR, mtx2, dist2)
   
    # Compute rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify( mtx1, dist1, mtx2, dist2, 
                                                     (imgL.shape[1], imgL.shape[0]), R, T, 
                                                     alpha = 0, flags=calibration_flags)
    
    map1x, map1y = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, (imgL.shape[1], imgL.shape[0]), cv.CV_16SC2)
    map2x, map2y = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, (imgR.shape[1], imgR.shape[0]), cv.CV_16SC2)

    # Rectify the images
    rectified_img1 = cv.remap(imgL, map1x, map1y, cv.INTER_CUBIC)
    rectified_img2 = cv.remap(imgR, map2x, map2y, cv.INTER_CUBIC)
         
    return rectified_img1, rectified_img2, img1, img2, Q

#Calibrate individual cameras to obtain Distortion, Rotation matrices
mtx1, dist1 = calibrate_camera(
    images_folder = '/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImgg/left/*')
mtx2, dist2 = calibrate_camera(
    images_folder = '/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImgg/right/*')

frames_folder = "/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImgg/synched/"
before_frames_folder = "/home/shishir/Desktop/Project/Scripts/IMGs/StereoTest/CalImg/"

imgL = cv.imread(os.path.join(before_frames_folder, "img_left00.jpg"))
imgR= cv.imread(os.path.join(before_frames_folder, "img_right00.jpg"))

#Stereo calibration
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder)
# Rectify images
rectified_img1, rectified_img2, img1, img2, Q = rectify_images(imgL, imgR, R, T, mtx1, dist1, mtx2, dist2)
