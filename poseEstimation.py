import cv2 as cv
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv.VideoCapture("Video/posedVideos/posingVideos3.mp4")
cTime = 0
pTime = 0

if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        resized = cv.resize(img,(500,500))
        imgRGB = cv.cvtColor(resized,cv.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(resized, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = resized.shape
                print(id,lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv.circle(resized, (cx,cy), 5, (250,15,45),cv.FILLED)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(resized, str(int(fps)), (70,50),cv.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv.imshow("ViDeO",resized)
    
        if cv.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
