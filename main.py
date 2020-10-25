from cv2 import cv2
import imutils
import dlib
from playsound import playsound
import numpy as np
import helper as hp
from threading import Thread
from os import path

detector = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# takes arguments - path for close eyed photos
THRES = hp.get_initial_threshold(
    './pictures/close.jpg', detector, predict) if path.exists('./pictures/close.jpg') else 0.2

MOTION_THRES = 15
no_movement = 0

count = 0
calibration_count = 0  # for initial calibration of frame


def alarm(path):
    playsound(path)


cap = cv2.VideoCapture(0)
# use first frame for motion calculation
_, prev_img = cap.read()
prev_img = imutils.resize(prev_img, width=500)
prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

while True:
    _, img = cap.read()

    img = hp.apply_clahe(img)
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = detector(gray, 1)
    face3D = hp.ref3DModel()

    # motion detection
    if not boxes:
        print("Landmarks not detected")
        flow = cv2.calcOpticalFlowFarneback(
            prev_img, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        max_elem = np.max(flow)

        if max_elem < MOTION_THRES:
            print("Head is not moving")  # Drowsy?
            no_movement += 1
            if no_movement > 100:
                playsound('./sounds/beep.mp3')
                cv2.putText(img, "MOTION ALERT!", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            no_movement = 0

    else:
        no_movement = 0

    prev_img = gray

    for box in boxes:

        ratio_avg, leftEyeHull, rightEyeHull = hp.handle(gray, box, predict)

        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

        cv2.putText(img, "ratio: "+str(int(ratio_avg*100)/100), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "ratio: "+str(int(ratio_avg*100)/100), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "thres: "+str(int(THRES*100)/100), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "thres: "+str(int(THRES*100)/100), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if ratio_avg < THRES:
            count += 1

            if count == 40:
                t = Thread(target=alarm,
                           args=('./sounds/beep.mp3',))
                t.deamon = True
                t.start()
                cv2.putText(img, "DROWSY ALERT!", (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(img, "DROWSY ALERT!", (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imwrite('sleepy_driver.jpg', img)
                count = 0
        shape = predict(gray, box)
        hp.draw(img, shape)
        refImgPts = hp.ref2dImagePoints(shape)
        height, width, channels = img.shape
        focalLength = height
        cameraMatrix = np.float32([[focalLength, 0, (height/2)],
                                   [0, focalLength, (width/2)],
                                   [0, 0, 1]])

        success, rotationVector, translationVector = cv2.solvePnP(
            face3D, refImgPts, cameraMatrix, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
        mdists = np.zeros((4, 1), dtype=np.float64)
        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        cv2.line(img, p1, p2, (110, 220, 0),
                 thickness=2, lineType=cv2.LINE_AA)

        # calculating euler angles
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        if angles[0] > 150:
            if angles[1] < -15:
                GAZE = "Looking: up-Left"
            elif angles[1] > 15:
                GAZE = "Looking up-Right"
            else:
                GAZE = "Looking Up"
        elif angles[0] < - 20 and angles[0] > -170:
            GAZE = "Looking Down"
        elif angles[1] > 15:
            GAZE = "Looking: Right"
        elif angles[1] < - 15:
            GAZE = "Looking: Left"
        else:
            GAZE = "Forward"

        direction = [int(i*100)/100 for i in angles]
        cv2.putText(img, GAZE + " " + str(direction), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
        cv2.putText(img, GAZE + " " + str(direction), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # calibration for first 120 frames
        if calibration_count < 120:
            new_rat, _, _ = hp.handle(gray, box, predict)
            THRES = max(THRES, 0.6*new_rat)

            cv2.putText(img, "Please keep your eyes open, calibrating....", (60, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2)
            cv2.putText(img, "Please keep your eyes open, calibrating....", (60, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            calibration_count += 1

    cv2.imshow("cctv", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
