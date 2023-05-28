import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)



# Choose a tracking algorithm (BOOSTING, MIL, KCF, TLD, MEDIANFLOW, or GOTURN)
tracker = cv2.TrackerKCF_create()

# Read the first frame
ret, frame = cap.read()

# Select the object to track by drawing a bounding box around it
bbox = cv2.selectROI(frame, False)

# Initialize the tracker with the bounding box
tracker.init(frame, bbox)

while True:
    # Read a new frame
    ret, frame = cap.read()

    # Update the tracker
    success, bbox = tracker.update(frame)

    # If the tracking was successful
    if success:
        # Draw the bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
