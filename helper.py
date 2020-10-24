import numpy as np
from cv2 import cv2
from scipy.spatial import distance as dist
from imutils import face_utils


def aspect_ratio(eye):
    """
    takes as input the eye landmarks, and
    gives the average eye aspect ration as
    the output
    """
    h1 = dist.euclidean(eye[1], eye[5])
    h2 = dist.euclidean(eye[2], eye[4])
    w = dist.euclidean(eye[0], eye[3])
    ar = (h1+h2)/(2.0*w)
    return ar


def handle(gray, box, predict):
    """
    takes as input a grayscale image and the box containing the face
    returns the average eye aspect ratio, and the convex hulls for
    the left and the right eyes
    """
    predicted = predict(gray, box)
    shape = face_utils.shape_to_np(predicted)

    # find the indexes of the two eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)
    ratio = max(aspect_ratio(left_eye), aspect_ratio(right_eye))

    return ratio, leftEyeHull, rightEyeHull


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b = clahe.apply(img[:, :, 0])
    g = clahe.apply(img[:, :, 1])
    r = clahe.apply(img[:, :, 2])
    return np.dstack((b, g, r))


def get_initial_threshold(close_path, detector, predict):
    """
    inputs: paths for images where subject's eyes are
     closed 
    output: the THREShold aspect ratio
    """

    eclose = cv2.cvtColor(cv2.imread(close_path), cv2.COLOR_BGR2GRAY)
    ratio_close, _, _ = handle(eclose, detector(eclose, 1)[0], predict)

    return (ratio_close)  # subject to change


def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def ref2dImagePoints(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)


def drawPolyline(img, shapes, start, end, isClosed=False):
    points = []
    for i in range(start, end + 1):
        point = [shapes.part(i).x, shapes.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], isClosed, (255, 80, 0),
                  thickness=1, lineType=cv2.LINE_8)


def draw(img, shapes):
    drawPolyline(img, shapes, 0, 16)
    drawPolyline(img, shapes, 17, 21)
    drawPolyline(img, shapes, 22, 26)
    drawPolyline(img, shapes, 27, 30)
    drawPolyline(img, shapes, 30, 35, True)
    drawPolyline(img, shapes, 36, 41, True)
    drawPolyline(img, shapes, 42, 47, True)
    drawPolyline(img, shapes, 48, 59, True)
    drawPolyline(img, shapes, 60, 67, True)
