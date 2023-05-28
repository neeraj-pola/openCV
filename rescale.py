import cv2 as cv

# img = cv.imread("images/download.jpg")
# cv.imshow("Cat", img)


def rescaleFrame(frame,scale=0.75):
    # for images,videos,live Videos
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)
    
    """
    INTER_AREA FOR RESCALING TO LOWER SIZE
    INTER_LINEAR OR INTER_CUBIC TO INCREASE SIZE
    """

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)


def changeRes(width,height):
    # only for live videos
    capture.set(3,width)
    capture.set(4,height)

capture = cv.VideoCapture("Video/minion.gif")



while capture.isOpened():
    isTrue, frame = capture.read()

    if isTrue:

        frame_resized = rescaleFrame(frame)

        cv.imshow("Video",frame)
        cv.imshow("Rescaled Video",frame_resized)

        if cv.waitKey(20) & 0xFF == 27:
            break
    else:
        break


cv.waitKey(0)