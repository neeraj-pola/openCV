import cv2

# Capture video from the video stream
cap = cv2.VideoCapture(0)

# Set the ROI (region of interest)
x, y, w, h = 100, 100, 200, 200
roi = (x, y, x, y)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture a frame from the video stream
    ret, frame = cap.read()
    
    cv2.rectangle(frame,(0,0),(w,h ),(0,255,0),3)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the background subtractor to identify the moving objects in the ROI
    mask = bg_subtractor.apply(gray[y:y+h, x:x+w])
    
    # Threshold the mask
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    
    # Use morphological transformations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Use contour detection to find the outlines of the moving objects
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    # Draw the outlines of the moving objects on the original frame\\

    if contours:
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Movement Detection", frame)
    
    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
