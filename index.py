# Import libraries
import cv2
import numpy as np

# Using KNN and MOG2 background subtractors
# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # detectShadows=True: exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # exclude shadow areas from the objects you detected

# Choose your subtractor
bg_subtractor = MOG2_subtractor

# Open video file
camera = cv2.VideoCapture("C:\\Users\\hegde\\OneDrive\\Desktop\\objectTracker using bg subtraction\\walking1.mp4")

while True:
    ret, frame = camera.read()
    
    # Check if the frame was successfully read
    if not ret:
        break

    # Every frame is used both for calculating the foreground mask and for updating the background. 
    foreground_mask = bg_subtractor.apply(frame)

    # Threshold to create a binary image containing only white and black pixels
    ret, threshold = cv2.threshold(foreground_mask.copy(), 120, 255, cv2.THRESH_BINARY)
    
    # Dilation expands or thickens regions of interest in an image.
    dilated = cv2.dilate(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    
    # Find contours 
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check every contour, if it exceeds a certain value, draw bounding boxes
    for contour in contours:
        # If the area exceeds a certain value, draw bounding boxes
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Display the frames
    cv2.imshow("Subtractor", foreground_mask)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Detection", frame)
    
    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) & 0xff == 27:
        break
        
# Release resources
camera.release()
cv2.destroyAllWindows()
