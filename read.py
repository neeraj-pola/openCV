import cv2 as cv

img = cv.imread('images/download.jpg')
cv.imshow("cat", img)

cv.waitKey(0)

# video reading
cap = cv.VideoCapture("Video/minion.gif")
if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv.imshow('Frame',frame)
        # Press esc to exit
        if cv.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv.destroyAllWindows()