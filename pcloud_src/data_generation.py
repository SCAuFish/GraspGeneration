import xml.etree.ElementTree as ET
import os
from collections import namedtuple
import numpy as np
from transforms3d.quaternions import mat2quat
from PIL import Image
from tqdm import tqdm
import open3d
import sapien.core as sc
from scipy.stats import mode
import socket

import numpy as np

CLOUD_DIR = '/cephfs/chs091/clouds_col/'
if socket.gethostname() == 'AuFish2020':
    # CLOUD_DIR = '~/Documents/grasp_data/clouds_col/'
    CLOUD_DIR = '~/Documents/grasp_data/grasp_gen_prod'

OUTPUT_DIR = CLOUD_DIR

def npz2pcd(npz_file, pcd_out_file):
    npz_pc = np.load(npz_file)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(npz_pc['points'])
    pcd.normals = open3d.utility.Vector3dVector(npz_pc['normals'])

    open3d.io.write_point_cloud(pcd_out_file, pcd)


def direction2quat(direction):
    x = direction / np.linalg.norm(direction)
    r = np.cross(direction, [0, 0, -1])
    if r @ r < 0.01:
        r = np.cross(direction, [0, 1, 0])
    y = r / np.linalg.norm(r)
    z = np.cross(x, y)
    return mat2quat(np.array([x, y, z]).T)


def global_point_cloud(depth, points, normal, model):
    points = points[depth < 1]
    normal = normal[depth < 1]
    points[:, 3] = 1
    world_points = model @ points.T
    world_points = world_points.T
    world_normals = model[:3, :3] @ normal.T[:3]
    world_normals = world_normals.T

    return world_points[:, :3], world_normals[:, :3]


def initial_configuration(articulation: sc.Articulation):
    limits = articulation.get_qlimits()
    q = []
    for l in limits:
        if l[0] < -100:
            q.append(0)
        q.append(l[0])
    return np.array(q)


def random_configuration(seed: int, articulation: sc.Articulation):
    np.random.seed(seed)
    limits = articulation.get_qlimits()
    q = [0 for i in range(len(limits))]
    for i in range(len(limits)):
        l = limits[i]
        if l[0] < -100:
            q[i] = (np.random.random() * np.pi * 2 - np.pi)
        else:
            q[i] = (np.random.random() * (l[1] - l[0]) + l[0])
    return np.array(q)


def generate_point_cloud(id: str, seed: int = None, dataset_dir='../../dataset', render_collision=False, obj_scale=1):
    with open(os.path.join(os.path.dirname(__file__), 'icosphere2.vertices'), 'r') as f:
        sphere_points = np.array([[float(n) for n in line.strip().split()] for line in f])

    engine = sc.Engine()
    renderer = sc.VulkanRenderer(True)
    engine.set_renderer(renderer)
    config = sc.SceneConfig()
    config.gravity = [0, 0, 0]
    scene = engine.create_scene(config=config)

    scene.set_timestep(1 / 500)
    scene.set_ambient_light([0.2, 0.2, 0.2])
    scene.set_shadow_light([1, 1, -1], [1, 1, 1])

    urdf = os.path.join(dataset_dir, id, "mobility.urdf")
    loader = scene.create_urdf_loader()
    loader.scale = obj_scale
    loader.fix_root_link = True
    articulation = loader.load(urdf)
    if render_collision:
        # render collision instead of visual
        for link in articulation.get_links():
            link.render_collision()

    cm = scene.create_actor_builder().build(True)
    c = scene.add_mounted_camera('', cm, sc.Pose(), 512, 512, 0, 70 / 180 * np.pi, -10, 10)
    radius = 2
    c.set_mode_orthographic(2)

    if seed is not None:
        articulation.set_qpos(random_configuration(seed, articulation))
    else:
        articulation.set_qpos(initial_configuration(articulation))

    articulation.set_qvel([0] * articulation.dof)
    for i in range(5):
        scene.step()

    link_info = {}
    print(articulation.get_qpos())
    link_info['qpos'] = articulation.get_qpos().tolist()
    for link, joint in zip(articulation.get_links(), articulation.get_joints()):
        if link.name == 'base':
            continue
        joint_pose = joint.get_global_pose()
        joint_name = joint.name

        if joint.type == sc.ArticulationJointType.PRISMATIC:
            joint_type = 'prismatic'
        elif joint.type == sc.ArticulationJointType.REVOLUTE:
            joint_type = 'revolute'
        else:
            joint_type = 'none'
        joint_type = joint.type
        link_id = link.get_id()
        link_name = link.name

        link_info[int(link_id)] = {
            'joint_pose': [[float(x) for x in joint_pose.p], [float(x) for x in joint_pose.q]],
            'joint_name': str(joint.name),
            'link_name': str(link.name),
            'link_id': int(link_id),
        }

    import json
    dirname = '{}/{}_{}'.format(OUTPUT_DIR, id, seed if seed is not None else 'init')
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, 'info.json'), 'w') as f:
        json.dump({'seed': seed, 'scale': obj_scale, 'info': link_info}, f)

    for l in articulation.get_links():
        l.unhide_visual()

    all_cloud = None
    for i, sphere_point in tqdm(list(enumerate(sphere_points))):
        sphere_point = sphere_point / np.linalg.norm(sphere_point)
        cm.set_pose(sc.Pose(sphere_point * radius, direction2quat(-sphere_point)))
        scene.update_render()
        c.take_picture()

        rgba = c.get_albedo_rgba()
        depth = c.get_depth()
        normal = c.get_normal_rgba()
        points = c.get_position_rgba()
        segmentation = c.get_segmentation()

        points, normals = global_point_cloud(depth, points, normal, c.get_model_matrix())
        segmentation = segmentation[depth < 1]

        colors = np.zeros((len(segmentation), 3))
        colors[:, 0] = segmentation

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        cloud.normals = open3d.utility.Vector3dVector(normals)
        cloud.colors = open3d.utility.Vector3dVector(colors)

        if all_cloud is None:
            all_cloud = cloud
        else:
            all_cloud += cloud

    downsampled, _, indices = all_cloud.voxel_down_sample_and_trace(0.005, all_cloud.get_min_bound(),
                                                                    all_cloud.get_max_bound())
    print('Downsample finished')

    ss = []
    for idx in indices:
        segs = [int(color[0]) for color in np.asarray(all_cloud.colors)[idx]]
        seg = mode(segs).mode[0]
        ss.append(seg)

    np.random.seed(100)
    colors = np.random.random((100, 3))
    downsampled.colors = open3d.utility.Vector3dVector(colors[ss])
    # open3d.visualization.draw_geometries([downsampled])

    P = np.array(downsampled.points)
    N = np.array(downsampled.normals)
    S = np.array(ss)

    np.savez(os.path.join(dirname, 'all.npz'), points=P, normals=N, segmentation=S)

    # transform npz to pcd and remove the npz
    npz2pcd(os.path.join(dirname, 'all.npz'), os.path.join(dirname, 'all.pcd'))

    print('Done saving {}'.format(id))

    scene = None


import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", help="the directory to which object urdfs and other related files are saved")
    parser.add_argument("--output_to", help="the directory storing object files (urdf)")
    parser.add_argument("--object_id", help="the object id that grasps will be generated on")
    parser.add_argument("--pose_id", help="the pose random init seed for the object")
    parser.add_argument("--scale",
                        help="the scale of the loaded object. By default the object will be in 1-meter unit sphere",
                        default=1)

    args = parser.parse_args()
    OUTPUT_DIR = args.output_to
    print("Generating in folder: {}".format(args.output_to))

    pose_id = None if args.pose_id is None else int(args.pose_id)
    generate_point_cloud(args.object_id, pose_id, args.dataset_dir, obj_scale=float(args.scale), render_collision=True)
