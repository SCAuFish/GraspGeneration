import numpy as np
import open3d
import random

from evaluator import EnvFreeRevoluteEvaluator, WholeObjectEvaluator
from eval_util import get_paired_score

def get_grasps(grasp_file):
    lines = []
    
    # Take the candidate at the first index as line
    reader = open(grasp_file, "r")
    
    contained_in_grasp = set()
    for line in reader.readlines():
        current_point = int(line.split("#")[0])
        
        if current_point in contained_in_grasp:
            continue
            
        grasp_pair    = line.split("#")[-1]
        if "|" in grasp_pair:
            grasp_pair = grasp_pair.split("|")[0]
            grasp_pair = grasp_pair.strip().split(" ")[0]
            
        grasp_pair     = int(grasp_pair)
        contained_in_grasp.add(grasp_pair)
        lines.append([current_point, grasp_pair])
        
    return lines

def visualize_grasps(pcd_file, grasp_file):
    pcd = open3d.io.read_point_cloud(pcd_file)

    lines = get_grasps(grasp_file)
    print(len(lines))
    lines = random.choices(lines, k=max(100000, len(lines)))
    points = np.asarray(pcd.points)
    lineset = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    colors = [[0, 0, 1] for i in range(len(lines))]
    lineset.colors = open3d.utility.Vector3dVector(colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    open3d.visualization.draw_geometries([pcd, lineset, mesh_frame])

def visualize_grasps_quality(pcd_file, grasp_file, obj_id, joint_name, friction_coef=0.7):
    pcd = open3d.io.read_point_cloud(pcd_file)

    lines = get_grasps(grasp_file)
    print(len(lines))
    lines = random.choices(lines, k=max(100000, len(lines)))
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    lineset = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    # evaluate grasps and color them
    evaluator = WholeObjectEvaluator(str(obj_id))
    _, xyz, axis = evaluator.find_axis(joint_name)
    # axis = np.array([0, 0, 1])
    # xyz  = np.array([-0.52, 0.38, 0])
    print("xyz, axis: {}, {}".format(xyz, axis))
    unit_torque_force, force_along_normal, force_ortho = evaluator.eval_grasp(points, normals, xyz, axis)
    # If force_along_normal is negative, it is valid, meaning pushing inward. If it is positive, it is not valid in
    # soft-finger contact model
    scores = []
    mean_scores = []
    for proposal in lines:
        p1, p2 = proposal
        
        score = get_paired_score(p1, p2, unit_torque_force, force_along_normal, force_ortho, friction_coef=friction_coef)
        scores.append(score)
        mean_scores.append(score[0] + score[1])

    max_score = max(mean_scores)
    colors = [[0, mean_scores[i] / 10, 1-(mean_scores[i] / 10)] for i in range(len(lines))]
    lineset.colors = open3d.utility.Vector3dVector(colors)

    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    open3d.visualization.draw_geometries([pcd, lineset, mesh_frame])


import sys
if __name__ == "__main__":
    point_cloud_file = sys.argv[1]
    grasp_file       = sys.argv[2]
    if len(sys.argv) == 3:
        visualize_grasps(point_cloud_file, grasp_file)
    else:
        obj_id     = sys.argv[3]
        joint_name = sys.argv[4]
        visualize_grasps_quality(point_cloud_file, grasp_file, obj_id, joint_name, 0.1)
    # visualize_grasps("../35059_link0.pcd", "../eval_src/filtered_proposal.out")
    # visualize_grasps("../assets/smoothed_35059_link0.pcd", "../outputs/smoothed_grasps_100cm.out")
    # visualize_grasps("../assets/smoothed_35059_link0.pcd", "../outputs/local_robust_smoothed_grasps_100cm.out")
    # visualize_grasps("../assets/smoothed_35059_link0.pcd", "../outputs/smoothed_grasps_10cm.out")
    # visualize_grasps("../assets/smoothed_35059_link0.pcd", "../outputs/local_robust_smoothed_grasps_10cm.out")
    # visualize_grasps("../35059_0/pointcloud.pcd", file)
    # visualize_grasps("../assets/smoothed_35059_link0.pcd", file)

