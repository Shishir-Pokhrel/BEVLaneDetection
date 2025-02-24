# BEVLaneDetection
Bird Eye view transformation and lane detection in automotive

This repository walks through an implementation of Bird eye view of captured images/video and detect lanes. 

The requirements of this project are: 
- Python-3.x
- OpenCV2

Bird Eye View:
To obtain a bird;s eye view on an image, a homography transformation is done. Two images of the same scene are related by a homography. Two scenes (perspective, and a top-down) views are related by a homography. This homography transforms a perspective view to a bird eye view. 

How to obtain the homography?
The script obtainHomography.py implements the following: 
- A checkerboard pattern in perspective view and in a top down view is fed to the script
- RANSAC algorithm is used to detect edges of the checkerboard.
- Comparision of two views gives a relationship called the homography

How to use the homography?
- This homography matrix of 3x3 size is used to warp any other input images.
- LaneDetect.py uses this homography to warp perspective image into a bird eye view.

How is lane detected?
- The BEV image is processed with Gaussian blur to reduce noise, Canny Edge detection to detect edges, and the Hough transformation is used to obtain lines from the detected images.
- From the detected lines, a polynomial fit is done on the image, which effectively is lane boundary for an automobile to keep inside of.
  

