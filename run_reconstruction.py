import pickle
import Metashape
import numpy as np
import open3d as o3d
from glob import glob


doc = Metashape.Document()

if True:
    chunk = doc.addChunk()

    #images = [os.path.join(folder_path, "0_images", elem)
    #          for elem in os.listdir(os.path.join(folder_path, "0_images"))]
    images = glob("/home/chrisbe/Desktop/20221025_christian/images/*")
    chunk.addPhotos(images)
    chunk.matchPhotos()
    chunk.alignCameras()

    doc.save("/home/chrisbe/Desktop/20221025_christian/model/metashape.psx")

doc.open("/home/chrisbe/Desktop/20221025_christian/model/metashape.psx")
chunk = doc.chunk



# prepare true and reconstructed camera centers
with open("/home/chrisbe/Desktop/20221025_christian/model/true_camera_centers.pkl", "br") as f:
    true_camera_centers = pickle.load(f)
reco_camera_centers = dict((cam.label, np.array(cam.center))
                           for cam in chunk.cameras if cam.center is not None)
true_camera_centers = {"-".join(key.split("-")[1:4]): true_camera_centers[key] for key in true_camera_centers.keys()}
reco_camera_centers = {"-".join(key.split("-")[1:4]): reco_camera_centers[key] for key in reco_camera_centers.keys()}

# get corresponding cameras
keys = true_camera_centers.keys() & reco_camera_centers.keys()
true_camera_centers = np.array([true_camera_centers[k] for k in keys])
reco_camera_centers = np.array([reco_camera_centers[k] for k in keys])

# estimate transformation
source_cloud = o3d.geometry.PointCloud()
target_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(reco_camera_centers[:, :3])
target_cloud.points = o3d.utility.Vector3dVector(true_camera_centers[:, :3])
corres_op3d = o3d.utility.Vector2iVector(np.vstack(
    [np.arange(0, len(source_cloud.points)), np.arange(0, len(source_cloud.points))]).T)
transform = o3d.pipelines.registration.TransformationEstimationPointToPoint(True).compute_transformation(
    source_cloud, target_cloud, corres_op3d)

# apply transformation
transform = Metashape.Matrix(transform)
chunk.transform.matrix = transform * chunk.transform.matrix

positions = np.empty((0, 3))
for cam in chunk.cameras:
    if cam.center is not None:
        cam_center = Metashape.Vector([cam.center.x, cam.center.y, cam.center.z, 1])
        positions = np.append(positions, np.array(transform * cam_center)[:3].reshape(1, 3), axis=0)
center = np.mean(positions, axis=0)
radius = np.mean(np.ptp(positions, axis=0)[:2]) / 2  # get range and average x and y range

# crop cloud
for pt in chunk.point_cloud.points:
    if radius * 0.8 < np.linalg.norm(np.array(chunk.transform.matrix * pt.coord)[:3] - center):
        pt.valid = False

# crop to head
max_z = np.array([chunk.transform.matrix * pt.coord for pt in chunk.point_cloud.points if pt.valid]).max(axis=0)[2]
min_z = 0.3# max_z - 0.5

for pt in chunk.point_cloud.points:
    coord_z = np.array(chunk.transform.matrix * pt.coord)[2]
    if max_z < coord_z or coord_z < min_z:
        pt.valid = False

# build mesh and texture
chunk.buildModel(surface_type=Metashape.Arbitrary,  # interpolation=Metashape.EnabledInterpolation,
                 source_data=Metashape.PointCloudData)
chunk.buildUV(mapping_mode=Metashape.GenericMapping)
chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)

chunk.exportModel("/home/chrisbe/Desktop/20221025_christian/model/model.obj")

doc.save("/home/chrisbe/Desktop/20221025_christian/model/metashape_tmp.psx")
print()