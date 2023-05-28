import cv2
import mediapipe as mp
import face_recognition
cap = cv2.VideoCapture(0)


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
faceDetection = mpFaceDetection.FaceDetection(0.75)

first_frame = None

if cap.isOpened() == False:
    print("Error in opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(500,500))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    # cv2.imshow("gray",gray)
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    threshold_frame = cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    # cv2.imshow("threshold frame",threshold_frame)
    threshold_frame = cv2.dilate(threshold_frame,None,iterations=2)
    # cv2.imshow("dilated frame",threshold_frame)
    

    cntr, hierarchy = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    i=0
    for contour in cntr:
        if cv2.contourArea(contour)<5000:
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        # print(contour)
        # print(contour[0][0])
        new = frame[x:x+w,y:y+h]
        newRGB = cv2.cvtColor(new,cv2.COLOR_BGR2RGB)
        results = faceDetection.process(newRGB)
        if new.any():
            # cv2.imwrite("short_title_contour_{0}.jpg".format(i),new)
            # i+=1
            if results.detections:
                for id,detection in enumerate(results.detections):
                    mpDraw.draw_detection(new, detection)
                    # # print(detection.location_data.relative_bounding_box)
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = new.shape
                    bbox = int(bboxC.xmin* iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
                    bx,by,bw,bh = int(bboxC.xmin* iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)
                    cropped_image = new[by:by+bh,bx:bx+bh]

                    if bx>0 and by >0 and bw>0 and bh>0:
                        cv2.imwrite("cropped_face_image{0}.jpg".format(i),cropped_image)
                        i+=1
                    cv2.rectangle(new, bbox, (250,15,27), 2)
                    cv2.putText(new, f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(0,250,25),3)
                
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h ),(0,255,0),3)
                cv2.putText(frame, "not face",(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,250,25),3)

    cv2.imshow("moving",frame)


    if cv2.waitKey(20) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()


