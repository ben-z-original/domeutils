import cv2
import dlib
import argparse
import Metashape
import numpy as np
import pandas as pd
from glob import glob
import mediapipe as mp
from pathlib import Path

from pyntcloud import PyntCloud
from matplotlib import pyplot as plt
from pkg_resources import resource_string, resource_listdir

mp_hands = mp.solutions.hands


def compute_hand_landmarks(image):
    """Computes landmarks of a hand."""
    h, w, _ = image.shape

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        results = hands.process(image)

        if not results.multi_hand_landmarks:
            return [np.nan, np.nan]

        x = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
        y = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h

        if False:
            plt.imshow(image)
            plt.plot(landmark[0], landmark[1], "o")
            plt.show()

        if 0 <= x and 0 <= y and x < w and y < h:
            return [x, y]
        else:
            return [np.nan, np.nan]


def compute_facial_landmarks(root_dir):
    """Computes the 2D and 3D facial landmarks."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')

    doc = Metashape.Document()
    chunk = doc.addChunk()

    images = glob(str(root_dir / "images" / "*"))
    chunk.addPhotos(images)
    chunk.importCameras(str(root_dir / "model" / "cameras.xml"))
    chunk.importModel(str(root_dir / "model" / "model_export.obj"))

    pts3D = np.empty((len(chunk.cameras), 68, 3))
    pts3D[:] = np.nan

    for i, camera in enumerate(chunk.cameras):
        # if i <= 24 or i <= 35:
        #    continue
        gray = cv2.imread(list(filter(lambda x: camera.label in x, images))[0], cv2.IMREAD_GRAYSCALE)

        # detect faces
        rects = detector(gray, 1)

        if len(rects) == 0:
            print(f"{i} No face found")
            continue

        # detect landmarks
        shape = predictor(gray, rects[0])
        shape = np.array([(shape.part(j).x, shape.part(j).y) for j in range(68)])

        np.savetxt(str((root_dir / "landmarks" / "2d" / camera.label).with_suffix(".txt")), shape)

        if False:
            shape = np.loadtxt(str((root_dir / "landmarks" / "2d" / camera.label).with_suffix(".txt")))
            plt.imshow(gray, 'gray')

            plt.plot([rects[0].left(), rects[0].right(), rects[0].right(), rects[0].left(), rects[0].left()],
                     [rects[0].top(), rects[0].top(), rects[0].bottom(), rects[0].bottom(), rects[0].top()])
            plt.plot(shape[:, 0], shape[:, 1], 'o')
            plt.show()

        for j, pt2D in enumerate(shape):
            intersect = cast_metashape_ray(chunk, camera, pt2D)
            print(j, intersect)
            if intersect is not None:
                pts3D[i, j, :] = np.array(intersect[:3])

    landmarks3D = np.nanmedian(pts3D, axis=0)
    random_colors = np.random.random((68, 3))
    data = {}
    data['x'] = landmarks3D[:, 0]
    data['y'] = landmarks3D[:, 1]
    data['z'] = landmarks3D[:, 2]
    data['red'] = random_colors[:, 0]
    data['green'] = random_colors[:, 1]
    data['blue'] = random_colors[:, 2]

    points = pd.DataFrame(data)
    ply = PyntCloud(points)
    ply.to_file(str(root_dir / "landmarks" / "landmarks3d.ply"))


def cast_metashape_ray(chunk, camera, pt2D):
    """Performs ray casting using Agisoft Metashape"""
    # camera origin in chunk world
    center = camera.center
    # pixel in 3D world coordinates
    dot = camera.calibration.unproject(pt2D)
    # transform point into chunk world
    dot = camera.transform * Metashape.Vector((dot[0], dot[1], dot[2], 1.0))
    # ray casting
    intersect = chunk.model.pickPoint(center[:3], dot[:3])
    if intersect is not None:
        intersect = chunk.transform.matrix * Metashape.Vector([intersect[0], intersect[1], intersect[2], 1])

    return intersect


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute the 2D and 3D facial landmarks.")
    parser.add_argument("root_dir", type=str,
                        help="Path to directory hosting the images and the model. Example path: /run/user/1000/gvfs/smb-share:server=klee.medien.uni-weimar.de,share=server_extension/theses/style_transfer/data/20221025_christian")
    parser.add_argument("--logging_on", action="store_true", help="Turn on logging.")
    parser.set_defaults(logging_on=False)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    compute_facial_landmarks(root_dir)
