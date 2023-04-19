# https://towardsdatascience.com/face-landmark-detection-using-python-1964cb620837

import cv2
import dlib
import numpy as np
from glob import glob

import matplotlib

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt


def detect_landmarks(img_path, detector, predictor):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)[::4, ::4, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #plt.imshow(image)

    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects[0:1]:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        if False:
            for pt in shape:
                plt.plot(pt[0], pt[1], 'o')

        with open(img_path + ".npy", 'wb') as f:
            np.save(f, shape_np)
    #plt.show()
    #return shape_np


print()
if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    folder_path = "/home/chrisbe/Desktop/20221025_christian/images/*.jpg"

    for img_path in glob(folder_path):
        print(img_path)
        detect_landmarks(img_path, detector, predictor)
