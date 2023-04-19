import Metashape
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

doc = Metashape.Document()
doc.open("/home/chrisbe/Desktop/20221025_christian/model/metashape_tmp.psx")
chunk = doc.chunk

landmarks = np.load("/home/chrisbe/Desktop/20221025_christian/images/43-1-3-1-132004-553_DxO.jpg.npy")

points = np.empty((0, 3), np.float32)

for i, landmark in enumerate(landmarks):
    pt2D = 4 * landmark

    camera = chunk.cameras[2]

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
        points = np.append(points, np.array(intersect)[:3].reshape(1, 3), axis=0)

    print(i, intersect is not None)
    # intersection = chunk.model.pickPoint(camera.center, pt3)
    print()

# map image to cloud
data = {}
data['x'] = points[:, 0]
data['y'] = points[:, 1]
data['z'] = points[:, 2]

points = pd.DataFrame(data)
ply = PyntCloud(points)
print()