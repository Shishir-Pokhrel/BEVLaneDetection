import cv2
import numpy as np
import os

# Define the directory where your calibration images are located
img_dir = os.path.expanduser('~/Desktop/Project/Scripts/IMGs/VideoFeeds/')
video_file = 'videoLeftnew04.mp4'  
video_path = os.path.join(img_dir, video_file)

def rotation_matrix(theta_x, theta_y, theta_z):
       
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    R = Rz @ Rx @ Ry
    return R

H_matrix = np.array([[2.88681721e+00, 4.69945408e+00, -1.20809081e+03],
              [-3.17056795e-01, 1.29358378e+01, -4.66821762e+03],
              [-2.62183378e-04, 7.07383918e-03, 1.00000000e+00]])

roll, pitch, yaw = 1e-3* np.deg2rad([20,0,0])
R = rotation_matrix(roll, pitch, yaw)
H = H_matrix @ R
             
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open the video from {video_path}")
    exit(1)
  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_video.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

def preprocess_image(frame):
    birdseye_framePreprocess= cv2.warpPerspective(frame, H, (1280,960))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graybev  = cv2.cvtColor(birdseye_framePreprocess, cv2.COLOR_BGR2GRAY)
   
    #equ = cv2.equalizeHist(gray)
    bgray= cv2.GaussianBlur(frame, (15, 15), 1.5)
    bevgray= cv2.GaussianBlur(birdseye_framePreprocess, (3, 3), 1)
 
    # Adaptive Thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold (gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,17, 3 )

    # Morphological Opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Morphological Closing to bridge small gaps in lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Apply Canny Edge Detection
    ret, threshold_image = cv2.threshold(bgray, 130, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bgray, 230, 250)
    
    #cv2.imshow('edgess', edges)
    threshold_image_gray = cv2.cvtColor(threshold_image, cv2.COLOR_BGR2GRAY)
    edgesThresholded = edges + threshold_image_gray
   
    return edges, closed, thresh
    
def transform_image(frame, homography_matrix, roll, pitch, yaw, output_size):
    
    roll_rad, pitch_rad, yaw_rad = np.deg2rad([roll, pitch, yaw])
    rotation_mat = rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    transformation_matrix = homography_matrix @ rotation_mat
    framee = cv2.warpPerspective(frame, transformation_matrix, (640,480))
    #return cv2.warpPerspective(frame, transformation_matrix, (640,480))
    return framee, transformation_matrix
            
def process_frame(frame, H):
    height, width = frame.shape[:2]
    output_size = (width, height)
    
    birdseye_frame = cv2.warpPerspective(frame, H, output_size)
    blurred_frame = cv2.GaussianBlur(birdseye_frame, (17, 17), 1)  # Adjust kernel size and sigmaX
     
    birdseye_frameThresh= cv2.warpPerspective(thresh, H, output_size)
    birdseye_frameClosed = cv2.warpPerspective(closed, H, output_size)
    birdseye_frameEdge = birdseye_frameThresh
    height, width = birdseye_frameEdge.shape

    # Calculate crop width (20% of width)
    left_crop_width = int(width * 0)
    right_crop_width = int(width * 0)
    
    # Crop the image along the x-axis (removing 20% from both sides)
    cropped_imgMask = birdseye_frameEdge[:, left_crop_width:width-right_crop_width]
       
    mask = cropped_imgMask
    cv2.imshow('edges', mask)
    
    #mask = cv2.inRange(birdseye_frameEdge, np.array([200, 200, 227]), np.array([180, 140, 255]))
    mask_yellow = cv2.inRange(birdseye_frame, np.array([15, 120, 100]), np.array([35, 255, 255]))
    msk = mask.copy()
            
    #  histogram to locate left and right lane base points
    histogram = np.sum(mask[(mask.shape[0]) // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # sliding window parameters
    y = mask.shape[1]
    sliding_window_height = 80  
    left_points = []  # Store points for left lane
    right_points = []
    mask_height, mask_width = mask.shape
    dx = 30
    
    while y > 0:       
        left_img = mask[y - sliding_window_height:y, left_base - dx:left_base + dx]
        edges_left = cv2.Canny(left_img, 180, 220)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=10)
        
        if lines_left is not None:
            for line in lines_left:
                x1, y1, x2, y2 = line[0]
                x1 += left_base - dx
                x2 += left_base - dx
                y1 += y - sliding_window_height
                y2 += y - sliding_window_height
                left_points.append((x1, y1))
                left_points.append((x2, y2))
                
                cv2.line(birdseye_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
        # right sliding window
        right_img = mask[y - sliding_window_height:y, right_base - dx:right_base + dx]
        edges_right = cv2.Canny(right_img, 130, 240)
        lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=5, minLineLength=10, maxLineGap=15)
        
        if lines_right is not None:
            for line in lines_right:
                x1, y1, x2, y2 = line[0]
                x1, y1, x2, y2 = line[0]
                x1 += right_base - dx
                x2 += right_base - dx
                y1 += y - sliding_window_height
                y2 += y - sliding_window_height
                right_points.append((x1, y1))
                right_points.append((x2, y2))

                # Draw dashed lines
                cv2.line(birdseye_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                               
        y -= sliding_window_height
        
    def draw_continuous_lane(points, frame, color):
        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            points = points[points[:, 1].argsort()]  # Sort by y-coordinates
            x = points[:, 0]
            y = points[:, 1]
            polynomial = np.polyfit(y, x, 4)  
            y_vals = np.linspace(0, mask_height - 1, mask_height)
            #x_vals = polynomial[0] * y_vals**2 + polynomial[1] * y_vals + polynomial[2]
            #x_vals = polynomial[0] * y_vals**3 + polynomial[1] * y_vals**2+ polynomial[2] * y_vals +  polynomial[3]
            x_vals = polynomial[0] * y_vals**4 + polynomial[1] * y_vals**3+ polynomial[2] * y_vals**2 +  polynomial[3] * y_vals + polynomial[4]

            #x_vals = polynomial[0] * y_vals + polynomial[1] 
            
            pts = np.array([np.transpose(np.vstack([x_vals, y_vals]))], dtype=np.int32)
            cv2.polylines(frame, pts, isClosed=False, color=color, thickness=5)
    cropped_imgBEV= birdseye_frame[:, left_crop_width:width-right_crop_width]
        
    draw_continuous_lane(left_points, birdseye_frame, color=(255, 0, 0))
    draw_continuous_lane(right_points, birdseye_frame, color=(0, 255, 0))
    
       
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # Convert mask to RGB and resize to 640x480
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.resize(mask_rgb, (640, 480))

    msk_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    msk_rgb = cv2.resize(msk_rgb, (640, 480))


    cropped_imgBEV= birdseye_frame[:, left_crop_width:width-right_crop_width]
    
    # Process birdseye frame and resize to 640x480
    birdseye_frame = cv2.resize(birdseye_frame, (640, 480))
 
    # Prepare combined image with a 2x2 grid of 640x480 images
    combined_img = np.zeros((h *2 , w * 2, 3), dtype=np.uint8)

    # Place each processed frame in the 2x2 combined image
    
    combined_img[0:h, 0:w, :] = frame                     
    combined_img[0:h, w:2 * w, :] = birdseye_frame        
    combined_img[h:2 * h, 0:w, :] = mask_rgb              
    #combined_img[h:2 * h, w:2 * w, :] = msk_rgb           

    cv2.imshow('Combined Video', combined_img)
    #cv2.imshow("Lane Detection", frame)
    out_video.write(frame)
    return msk

while cap.isOpened():
    ret, frame = cap.read()
    cv2.waitKey(20)
    (height,width,_) = frame.shape
    #left_crop_widthf = int(width * 0.15)
    #right_crop_widthf = int(width * 0.01)
    #cropped_frame= frame[:, left_crop_widthf:width-right_crop_widthf]
    #frame = cropped_frame
    # Process frame and display
    edges, closed, thresh = preprocess_image(frame)
    output_frame = process_frame(frame, H)   
    output_frame = cv2.resize(output_frame,(640,480))
    
    #birdseye_frame = cv2.warpPerspective(frame, H, (640,480))
    output_size = (640,480)
    birdseye_frameOriginal, HH = transform_image(frame, H, roll=-30, pitch=0, yaw = 0, output_size = (640,480))
    
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out_video.release()
cv2.destroyAllWindows()

