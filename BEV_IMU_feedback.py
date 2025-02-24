import cv2
import numpy as np
import serial
import time
import re

# Serial Communication from usb port. This is for linux, in Windows ,
# it is normally called COMx instead of /dev/ttyACM0, 
# please refer to documentations from windows for this. 
def initialize_serial(port='/dev/ttyACM0', baudrate=9600, timeout=1):
   
    ser = serial.Serial(port, baudrate, timeout=timeout)
    time.sleep(0.1)  #delay a little to avoid clash in intialisation and error
    return ser
                
def get_angles_from_serial(line):
    
    ser = initialize_serial()
    line = ser.readline().decode('utf-8').strip()

    #roll, pitch, yaw = None, None, None 
    roll, pitch, yaw = 0, 0, 0
    
    try:
        # use regex to extract data from the sensor output console
        match_roll = re.search(r'orientation roll :([-\d.]+)', line)
        match_pitch = re.search(r'orientation pitch :([-\d.]+)', line)
        match_yaw = re.search(r'orientation Heading :([-\d.]+)', line)
        
        # Extract values if the patterns are matched
        if match_roll:
            roll = float(match_roll.group(1))
        if match_pitch:
            pitch = float(match_pitch.group(1))
        if match_yaw:
            yaw = float(match_yaw.group(1))
        if roll is not None and pitch is not None and yaw is not None:
            return roll, pitch, yaw
        else:
            print(f"data is not there: {line}")
            return None

    except ValueError:
        print(f"data is corrutped: {line}")
        return None

#Rotation matrices
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
    return Rz @ Ry @ Rx

def transform_image(frame, homography_matrix, roll, pitch, yaw, output_size):
    
    roll_rad, pitch_rad, yaw_rad = 1e-3 * np.deg2rad([roll, pitch, yaw])
    rotation_mat = rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    transformation_matrix = homography_matrix @ rotation_mat
    transformed_imageWithRotation = cv2.warpPerspective(frame, transformation_matrix, output_size)
    return transformed_imageWithRotation

def draw_lines(image):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    color = (0, 255, 0)  
    thickness = 1
    cv2.line(image, (0, center_y), (width, center_y), color, thickness)  
    cv2.line(image, (center_x, 0), (center_x, height), color, thickness)
    return image

def main():
    # serial communication
    ser = initialize_serial()

    # Initialize webcam
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Homography matrix 
    H_matrix = np.array([[2.88681721e+00, 4.69945408e+00, -1.20809081e+03],
                         [-3.17056795e-01, 1.29358378e+01, -4.66821762e+03],
                         [-2.62183378e-04, 7.07383918e-03, 1.00000000e+00]])

    # Output size of images
    output_size = (1280, 960)
    
    try:
        while True:
            # Read roll, pitch, and yaw from arduiono
            roll, pitch, yaw = get_angles_from_serial(ser)

            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
                
            birdseye_frame = cv2.warpPerspective(frame, H_matrix, output_size)
            birdseye_frame = cv2.resize(birdseye_frame, (640, 480))
            
            # Apply  transformation
            transformed_frame = transform_image(frame, H_matrix, roll, pitch, yaw, output_size)

            # Add visual guides
            transformed_frame = draw_lines(transformed_frame)
            transformed_frame = cv2.resize(transformed_frame, (640, 480))
            
            #draw 2x2 plot
            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape
            combined_img = np.zeros((h*2, w * 2, 3), dtype=np.uint8)           
            combined_img[0:h, 0:w, :] = frame                     
            combined_img[0:h, w:2 * w, :] = birdseye_frame
            combined_img[h:2 * h, 0:w, :] = transformed_frame             
            cv2.imshow('Combined Video', combined_img)

            # q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        # Release cameras and devices
        cap.release()
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
