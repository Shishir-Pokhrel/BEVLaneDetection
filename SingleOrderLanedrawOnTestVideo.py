import cv2
import numpy as np
import os

# Where is the file??
img_dir = os.path.expanduser('~/Desktop/Project/Scripts/IMGs/VideoFeeds/')
video_file = 'test_video.mp4'  
video_path = os.path.join(img_dir, video_file)

H = np.array([[2.88681721e+00, 4.69945408e+00, -1.20809081e+03],
              [-3.17056795e-01, 1.29358378e+01, -4.66821762e+03],
              [-2.62183378e-04, 7.07383918e-03, 1.00000000e+00]])
              
# Initialize Video 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open the video from {video_path}")
    exit(1)
    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_video.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
      
        
def process_frame(frame, H):
    height, width = frame.shape[:2]
    output_size = (width, height)
    
    birdseye_frame = cv2.warpPerspective(frame, H, output_size) #Convert to BEV
    blurred_frame = cv2.GaussianBlur(birdseye_frame, (7, 7), 1)  

    mask_white = cv2.inRange(blurred_frame, np.array([0, 15, 125]), np.array([150, 140, 255]))
    mask_yellow = cv2.inRange(birdseye_frame, np.array([15, 120, 100]), np.array([35, 255, 255]))
    mask = cv2.bitwise_or(mask_white, mask_yellow) #OR operation for includoing both white and yellow lines opn the road
    #mask = mask_white
    msk = mask.copy()
    
    #Histogram for finding left and right bases
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Slidingwindow variables
    y = mask.shape[1]
    sliding_window_height = 60  
    window_width = 100 
    left_points = []  # Store points for left lane
    right_points = []
    mask_height, mask_width = mask.shape
    dx = 30
    
    while y > 0:
        left_lane_points = []
        right_lane_points = [] 
        
        #SWLeft
        left_img = mask[y - sliding_window_height:y, left_base - dx:left_base + dx]
        edges_left = cv2.Canny(left_img, 230, 255)
        lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, threshold=20, minLineLength=1, maxLineGap=25)
        
        if lines_left is not None:
            for line in lines_left:
                x1, y1, x2, y2 = line[0]
                x1 += left_base - dx
                x2 += left_base - dx
                y1 += y - sliding_window_height
                y2 += y - sliding_window_height
                left_points.append((x1, y1))
                left_points.append((x2, y2))
                
                cv2.line(birdseye_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
        # SW right
        right_img = mask[y - sliding_window_height:y, right_base - dx:right_base + dx]
        edges_right = cv2.Canny(right_img, 230, 240)
        lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, threshold=5, minLineLength=10, maxLineGap=45)
        
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

                # Draw green lines
                cv2.line(birdseye_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                               
        y -= sliding_window_height
        
    def draw_continuous_lane(points, frame, color=(0, 255, 0)):
        if len(points) > 2:
            points = np.array(points, dtype=np.int32)
            points = points[points[:, 1].argsort()]  # Sort by y-coordinates
            x = points[:, 0]
            y = points[:, 1]
            polynomial = np.polyfit(y, x, 1)  #  n order curve fitting
            y_vals = np.linspace(0, mask_height - 1, mask_height)
            #x_vals = polynomial[0] * y_vals**2 + polynomial[1] * y_vals + polynomial[2]
            x_vals = polynomial[0] * y_vals + polynomial[1] 
            
            pts = np.array([np.transpose(np.vstack([x_vals, y_vals]))], dtype=np.int32)
            cv2.polylines(frame, pts, isClosed=False, color=color, thickness=5)
        
    draw_continuous_lane(left_points, birdseye_frame, color=(0, 255, 0))
    draw_continuous_lane(right_points, birdseye_frame, color=(0, 255, 0))
       
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    #IMPORTANT :: convert to bgr because it will give problems with 2 channel vs 3 channel image size !!!! 
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.resize(mask_rgb, (640, 480))

    msk_rgb = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    msk_rgb = cv2.resize(msk_rgb, (640, 480))
    
    birdseye_frame = cv2.resize(birdseye_frame,(640, 480))

    #Make place to plot in the plotting console
    combined_img = np.zeros((h*2 , w * 2, 3), dtype=np.uint8)

    # Place image accordinglt
    combined_img[0:h, 0:w, :] = frame                     
    combined_img[0:h, w:2 * w, :] = birdseye_frame        
    combined_img[h:2 * h, 0:w, :] = mask_rgb              
    combined_img[h:2 * h, w:2 * w, :] = msk_rgb           

    cv2.imshow('Combined Video', combined_img)
    out_video.write(frame)
    return msk

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame and display
    output_frame = process_frame(frame, H)
    output_frame = cv2.resize(output_frame,(640,480))
    birdseye_frame = cv2.warpPerspective(frame, H, (640,480))

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release all
cap.release()
out_video.release()
cv2.destroyAllWindows()

