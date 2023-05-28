import cv2 as cv
import numpy as np

img = cv.imread("images/neeraj.jpg")
resized = cv.resize(img,(500,500))
# cv.imshow("image",resized)

#--------------------translation--------------------

def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])

    return cv.warpAffine(img, transMat,dimensions)

# -x --> Left
# x --> Right
# y --> Down
# -y --> Up

translated = translate(resized, -100,100)
# cv.imshow("translated",translated)



#---------------------- rotation--------------------------
def rotate(img,angle,rotPoint = None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat,dimensions)


rotated = rotate(resized, 270)
cv.imshow("rotated",rotated)

#------------------------------ flip-------------------------------------
# 0 --> verticalyy
#1 --> Horizontally
#-1 --> Both Horizontally and vertically

flipped = cv.flip(resized, -1)
# cv.imshow("flipped",flipped)

# ------------------------------cropping ------------------------------------
cropped = resized[200:300, 300:400]
# cv.imshow("cropped",cropped)















cv.waitKey(0)
