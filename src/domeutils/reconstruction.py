import cv2
import pickle
import argparse
import Metashape
import numpy as np
import open3d as o3d
from glob import glob
from pathlib import Path


def sfm_metashape(img_paths, doc_path=None):
    """Structure-from-motion pipeline performed with Agisoft Metashape."""
    doc = Metashape.Document()
    chunk = doc.addChunk()

    images = glob(str(img_paths / "*"))
    chunk.addPhotos(images)
    chunk.matchPhotos()
    chunk.alignCameras()

    if doc_path is not None:
        doc.save(str(doc_path))

    return doc


def register_cameras(chunk, true_camera_centers_path="assets/true_camera_centers.pkl"):
    """Register reconstructed cameras to true camera centers."""
    # prepare true and reconstructed camera centers
    with open(true_camera_centers_path, "br") as f:
        true_camera_centers = pickle.load(f)
    reco_camera_centers = dict(("-".join(cam.label.replace("_", "-").split("-")[1:4]), np.array(cam.center))
                               for cam in chunk.cameras if cam.center is not None)
    true_camera_centers = {"-".join(key.split("-")[1:4]): true_camera_centers[key] for key in
                           true_camera_centers.keys()}

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

    return transform


def crop_model(doc, transform):
    """Removes tie points in order to crop the volume of interest."""
    chunk = doc.chunk

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
    max_z = np.infty  # np.array([chunk.transform.matrix * pt.coord for pt in chunk.point_cloud.points if pt.valid]).max(axis=0)[2]
    min_z = 0.2  # max_z - 0.5

    for pt in chunk.point_cloud.points:
        coord_z = np.array(chunk.transform.matrix * pt.coord)[2]
        if coord_z < min_z or max_z < coord_z:
            pt.valid = False

    chunk.resetRegion()
    return doc


def build_textured_mesh(doc, export_path=None, doc_path=None):
    """Builds textured mesh for Agisoft Metashape project."""
    chunk = doc.chunk
    # build mesh and texture
    chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.AggressiveFiltering)
    chunk.buildDenseCloud()
    chunk.buildModel(surface_type=Metashape.Arbitrary, interpolation=Metashape.EnabledInterpolation,
                     source_data=Metashape.DenseCloud)
    # source_data=Metashape.DepthMapsData)
    # source_data=Metashape.PointCloudData)
    chunk.buildUV(mapping_mode=Metashape.GenericMapping)
    chunk.buildTexture(blending_mode=Metashape.MosaicBlending, texture_size=4096)

    if export_path is not None:
        chunk.exportModel(export_path)

    if doc_path is not None:
        doc.save(doc_path)

    return doc


def export_agisoft_model(chunk, export_path, name):
    """Export textured mesh from agisoft chunk."""
    obj = "mtllib " + name + ".mtl\n"
    obj += "usemtl " + name + "\n"

    model = chunk.model
    transform = np.array(chunk.transform.matrix).reshape(4, 4)

    for vertex in model.vertices:
        coord = np.array([vertex.coord[0], vertex.coord[1], vertex.coord[2], 1])
        coord = transform @ coord
        obj += f"v {coord[0]} {coord[1]} {coord[2]} {vertex.color[0] / 255} {vertex.color[1] / 255} {vertex.color[2] / 255}\n"

    for tex_vertex in model.tex_vertices:
        obj += f"vt {tex_vertex.coord[0]} {tex_vertex.coord[1]}\n"

    for face in model.faces:
        obj += f"f {face.vertices[0] + 1}/{face.tex_vertices[0] + 1} {face.vertices[1] + 1}/{face.tex_vertices[1] + 1} {face.vertices[2] + 1}/{face.tex_vertices[2] + 1}\n"

    with open(str((export_path / name).with_suffix(".obj")), 'w') as f:
        f.write(obj)

    mtl = "newmtl Solid\n"
    mtl += "Ka 1.0 1.0 1.0\n"
    mtl += "Kd 1.0 1.0 1.0\n"
    mtl += "Ks 0.0 0.0 0.0\n"
    mtl += "d 1.0\n"
    mtl += "Ns 0.0\n"
    mtl += "illum 0\n"
    mtl += "\n"
    mtl += "newmtl " + name + "\n"
    mtl += "Ka 1.0 1.0 1.0\n"
    mtl += "Kd 1.0 1.0 1.0\n"
    mtl += "Ks 0.0 0.0 0.0\n"
    mtl += "d 1.0\n"
    mtl += "Ns 0.0\n"
    mtl += "illum 0\n"
    mtl += "map_Kd " + name + ".jpg\n"

    with open(str((export_path / name).with_suffix(".mtl")), 'w') as f:
        f.write(mtl)

    for texture in model.textures:
        h, w, c = texture.image().height, texture.image().width, texture.image().cn
        img = np.frombuffer(texture.image().tostring(), dtype=np.uint8)
        img = img.reshape(h, w, c)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str((export_path / name).with_suffix(".jpg")), img)


def run_pipeline(root):
    img_paths = root / "images"
    export_path = root / "model"

    # compute structure-from-motion
    doc = sfm_metashape(img_paths=root / "images")
    if args.logging_on:
        doc.save(str(export_path / "metashape1.psx"))

    # register and crop
    transform = register_cameras(chunk=doc.chunk)
    doc = crop_model(doc, transform)
    if args.logging_on:
        doc.save(str(export_path / "metashape2.psx"))

    # create textured mesh
    doc = build_textured_mesh(doc=doc)
    if args.logging_on:
        doc.save(str(export_path / "metashape3.psx"))
        doc.chunk.exportCameras(str(export_path / "cameras.xml"))

    # export textured mesh
    export_agisoft_model(chunk=doc.chunk, export_path=export_path, name="model_export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run reconstruction pipeline with Agisoft Metashape.")
    parser.add_argument("root_dir", type=str,
                        help="Path to directory hosting the images and the model. Example path: /run/user/1000/gvfs/smb-share:server=klee.medien.uni-weimar.de,share=server_extension/theses/style_transfer/data/20221025_christian")
    parser.add_argument("--logging_on", action="store_true", help="Turn on logging.")
    parser.set_defaults(logging_on=False)
    args = parser.parse_args()

    root = Path(args.root_dir)

    run_pipeline(root)
