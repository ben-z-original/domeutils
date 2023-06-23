import cv2
import argparse
import numpy as np
from landmarks import compute_hand_landmarks
from reconstruction import LinearTriangulator


def line2obj(filepath, verts):
    """Exports lines vertices to a line obj."""
    # create trajectory obj
    out = ""
    # for v in points3d:
    for v in verts:
        out += f"v {v[0]} {v[1]} {v[2]}\n"
    out += "l"
    for i in range(len(vertices3d)):
        out += f" {i + 1}"
    with open(filepath, "w") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tracks the index finger in frames of the video cameras.")
    parser.add_argument("input_streams", nargs='+', type=str,
                        help="Provide a list of input streams, e.g. paths to .mp4 files.")
    parser.add_argument("--out_path", type=str,
                        help="Path to the obj that will contain the resulting trajectory.")
    args = parser.parse_args()

    # NOTE: streams, projection matrices, and landmarks are matched by index
    captures = [cv2.VideoCapture(stream) for stream in args.input_streams]

    projection_matrices = np.load("/home/chrisbe/Desktop/HandDrawing/projection_matrices.npy")
    lin_tri = LinearTriangulator(projection_matrices)

    landmarks = np.empty((len(captures), 2))
    vertices3d = []

    while all(map(cv2.VideoCapture.isOpened, captures)):
        frames = list(map(cv2.VideoCapture.read, captures))
        successes = [frame[0] for frame in frames]

        if not all(successes):
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        images = [frame[1] for frame in frames]

        for i, image in enumerate(images):
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            landmarks[i, ...] = compute_hand_landmarks(image)

        vertices3d.append(lin_tri(landmarks))
        print(vertices3d[-1])

        if len(vertices3d) == 100:
            break

    line2obj(args.out_path, vertices3d)


