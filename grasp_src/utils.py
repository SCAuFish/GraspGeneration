# Some functions for further filtering grasps/visualization
import sapien.core as sapien
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm as norm
import open3d
import json
from collections import defaultdict

class WholeObjectEvaluator():
    def __init__(self, objectId, ASSET_DIR):
        object_urdf_path = ASSET_DIR + "/" + objectId
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                object_urdf_path+"/mobility.urdf")
        self.objectID    = objectId
        self.object_root = ET.parse(filepath)

    def eval_grasp(self, contact_point, contact_normal, axisPoint, axisDir, joint:str):
        ''' Take a one contact point, evaluate the contact point

            ? Maybe should also evaluate how much to exert along the contact axis to achieve unit
            force along surface normal

            params: contact_point - 3D coordinates of contact point
                    contact_normal - the normal direction at the contact point
                    axis - axis of the joint, should be in the same frame as contact_point and contact_normal
                        axisPoint is the point through which the axis passes
                        axisDir is the direction of the axis (3x1 vector)
                    joint - joint type ("revolute", "prismatic", "continuous")

            return: force to generate unit torque, projection of unit force on contact normal,
                    projection of unit force on contact tangent
        '''
        # 0. Check dimensions, each row is a point/normal
        assert contact_normal.shape[1] == 3
        assert contact_point.shape[1]  == 3
        assert joint in ['prismatic', 'revolute', 'continuous']
        axisPoint     = axisPoint.reshape(1, 3)
        axisDir       = axisDir.reshape(1, 3)
        # 1. use origin_in_world and direction_in_world to calculate force arm to the contact points
        v1 = axisDir
        v2 = (axisPoint - contact_point)
        # axis=1 so that the norm is calculated per row
        force_arm = norm(np.cross(v1, v2), axis=1) / norm(v1)

        # 2. use force arm to calculate unit force on contact point
        unit_torque_force = 1 / force_arm
        if joint == 'prismatic':
            unit_torque_force *= force_arm

        # 3. Evaluate how much force is required along and orthogonal to contact normal to create unit force
        # orthogonal to force_arm
        # 3.1 Find force-arm vector
        unit_direction = axisDir / norm(axisDir)
        force_arm_vector_origin = axisPoint + ((contact_point - axisPoint.reshape(1, 3)) @ unit_direction.transpose()) * unit_direction
        force_arm_vector        = contact_point - force_arm_vector_origin
        if joint == 'prismatic':
            # For prismatic joint, all force_arms are considered to have same lengths
            force_arm_vector /= force_arm_vector

        # 3.2 Calculate unit force orthogonal to force-arm, on the same plane as contact normal and force-arm
        ortho_force  = np.cross(force_arm_vector, axisDir)

        ortho_force  = ortho_force / norm(ortho_force, axis=1).reshape((ortho_force.shape[0], 1))

        # 3.3 Calculate the force required along and orthogonal to contact normal to create unit_ortho_force
        force_along_normal = np.sum(ortho_force * (contact_normal / norm(contact_normal, axis=1).reshape((contact_normal.shape[0], 1))), axis=1)
        force_ortho_normal = np.sqrt(1 - np.clip(force_along_normal, -1, 1) ** 2)

        return unit_torque_force, force_along_normal, force_ortho_normal

    def eval_required_normal(self, unit_torque_force, force_along_normal, force_ortho_normal, friction_coef=0.7):
        ''' Takes output from eval_grasp as input and a friction coefficient so that the requried normal force
            may be calculated

            return - normal force required at each contact point to create unit torque
        '''
        normal_force = unit_torque_force / (np.abs(force_along_normal) + friction_coef * np.abs(force_ortho_normal))

        return normal_force

    def eval_and_visualize(self, part_name, friction_coef=0.7):
        pcd_file_path = "{}/{}/{}_downsample.pcd".format(CLOUD_DIR, self.objectID, part_name)
        pcd_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            pcd_file_path)
        pcd = open3d.io.read_point_cloud(pcd_file_path)
        joint, xyz, axis = self.find_axis(part_name)
        if joint:
            points = [xyz - axis * 5, xyz + axis * 5]
            lines = [[0, 1]]
            lineset = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(points),
                lines=open3d.utility.Vector2iVector(lines),
            )
            colors = [[1, 0, 0]] if joint == 'prismatic' else [[0, 1, 0]]
            lineset.colors = open3d.utility.Vector3dVector(colors)
            mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
            
            unit_torque_force, force_along_normal, force_ortho = \
                self.eval_grasp(np.asarray(pcd.points), np.asarray(pcd.normals), xyz, axis)
            normal_force_result = self.eval_required_normal(unit_torque_force, force_along_normal, force_ortho, friction_coef)
            colors = np.zeros((normal_force_result.shape[0], 3))
            normal_force_result = np.log(normal_force_result)
            normal_force_result = normal_force_result / np.max(normal_force_result)
            colors[:, 0] = normal_force_result
            pcd.colors = open3d.utility.Vector3dVector(colors)
            open3d.visualization.draw_geometries([pcd, lineset, mesh_frame])

    def find_axis(self, joint_name):
        ''' Get axis point and axis direction from URDF file.
        '''
        robot_root = self.object_root
        for joint in robot_root.findall("joint"):
            if joint.get('type') in ['prismatic', 'revolute', 'continuous']:
                if joint.get('name') == joint_name:
                    axis = joint.find('axis')
                    if axis is None:
                        axis = '1 0 0'
                    else:
                        axis = axis.get('xyz')
                    axis = np.array([float(x) for x in axis.split()])
                    assert axis.shape == (3, )

                    child_link_name = joint.find('child').get('link')
                    link = [l for l in robot_root.findall('link') if l.get('name') == child_link_name][0]
                    col = link.find('collision')
                    origin = col.find('origin')
                    xyz = np.array([float(x) for x in origin.get('xyz').split()])
                    assert xyz.shape == (3, )
                    assert origin.get('rpy') is None
                    if joint.get('type') == 'prismatic':
                        xyz = np.array([0, 0, 0])
                    return joint.get('type'), -xyz, axis
        return None, None, None
    
def get_paired_score(unit_torque_force: np.ndarray, force_along_normal: np.ndarray, force_ortho: np.ndarray, friction_coef=0.7):
    ''' Hypothesis: if two contact points are exerting force, the sum of two minimum normal forces are always greater than the
                    minimum normal force of only using one of the contact point.
    '''
    # Consider first rotation direction: require negative force along normal (rotating along axis)
    dir1nf = unit_torque_force / (np.abs(np.clip(force_along_normal, -10000, 0)) + friction_coef * np.abs(force_ortho))

    # Consider the second dirction: require positive force along normal (rotating anti axis)
    dir2nf = unit_torque_force / (np.clip(force_along_normal, 0, 10000) + friction_coef * np.abs(force_ortho))

    return (dir1nf, dir2nf)


def filter_and_score(object_npz, objectId, info_json_file, grasp_proposal, output_file, ASSET_DIR, friction_coef=0.7):
    # 1. Read in grasp proposals
    pointcloud = np.load(object_npz)
    points  = pointcloud['points']
    normals = pointcloud['normals']
    segmentation = pointcloud['segmentation']
    print("Loaded {} points from {}".format(len(points), object_npz))

    proposal_reader = open(grasp_proposal)
    proposals = []
    for line in proposal_reader.readlines():
        parts  = line.split("#")
        p1_idx = int(parts[0])
        others_str = parts[-1].strip(" |\n")
        others = [int(otherIdx.strip().split(" ")[0]) for otherIdx in others_str.split("|")]
        all_info = [otherIdx for otherIdx in others_str.split("|")]
        proposals += [(p1_idx, other_idx, all_info) for other_idx, all_info in zip(others, all_info)]
            
    # 2. Use info_json to map joint name, link segmentation id to joint and link name
    movable_segments = []
    info_json = json.load(open(info_json_file))['info']
    seg2link_joint = {}
    joint2pose = {}
    print(info_json)
    for key in info_json:
        if key == "qpos":
            continue
        seg2link_joint[int(key)] = (info_json[key]['link_name'], info_json[key]['joint_name'])
        joint2pose[info_json[key]['joint_name']] = sapien.Pose(info_json[key]['joint_pose'][0], info_json[key]['joint_pose'][1])
        
    # 3. Create evaluator
    evaluator = WholeObjectEvaluator(str(objectId), ASSET_DIR)
    
    # 4. Evaluate based on each link
    proposal_writer = open(output_file, "w")
    part2grasps     = defaultdict(list)
    # 4.1 group grasps based on the segmentation they are on
    for proposal in proposals:
        p1, p2, all_info = proposal
        if segmentation[p1] == segmentation[p2]:
            seg = segmentation[p1]
            part2grasps[seg].append(proposal)

    # 4.2 Evaluate and filter by segment

    for seg in seg2link_joint:
        # 4.1 find information about the current joint
        joint, xyz, axis = evaluator.find_axis(seg2link_joint[seg][1])
        if joint == None:
            continue
            
        # The x axis in joint frame is its rotation axis
        xyz_dir    = (joint2pose[seg2link_joint[seg][1]] * sapien.Pose([1, 0, 0])).p
        xyz_origin = (joint2pose[seg2link_joint[seg][1]] * sapien.Pose([0, 0, 0])).p

        axis = xyz_dir - xyz_origin
        xyz  = xyz_origin
        # 4.2 Get grasps that is related to current joint
        grasps  = part2grasps[seg]
        proposal_dict = defaultdict(list)
        for proposal in grasps:
            p1, p2, all_info = proposal

            proposal_dict[p1].append(all_info)

        # Calculate required normal force on each point
        unit_torque_force, force_along_normal, force_ortho = evaluator.eval_grasp(points, normals, xyz, axis, joint)
        dir1nf, dir2nf = get_paired_score(unit_torque_force, force_along_normal, force_ortho, friction_coef=friction_coef)

        for p in proposal_dict:
            output_str = "{}###".format(p)
            pairs = proposal_dict[p]

            pointIdx = int(p)
            for pair in pairs:
                pairIdx = int(pair.strip().split(" ")[0])
                # score = get_paired_score(pointIdx, pairIdx, unit_torque_force, force_along_normal, force_ortho, friction_coef=friction_coef)
                score = (min(dir1nf[pointIdx], dir1nf[pairIdx]), min(dir2nf[pointIdx], dir2nf[pairIdx]))

                output_str += "{} {}| ".format(pair, score)
                

            proposal_writer.write(output_str + "\n")


    print("Done score calculating")            
    proposal_writer.close()

# import argparse
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root", default="../clouds")
#     args = parser.parse_args()
#
#     for obj_dir in os.listdir(args.root):
#         try:
#             obj_id = int(obj_dir.split("_")[0])
#         except Exception as e:
#             print(e)
#             print(f"{obj_dir} is not a valid object dir")
#
#         print(f"processing {obj_dir}")
#         try:
#             filter_and_score(f"{args.root}/{obj_dir}/all.npz",
#                             obj_id,
#                             f"{args.root}/{obj_dir}/info.json",
#                             f"{args.root}/{obj_dir}/raw_grasp.out",
#                             f"{args.root}/{obj_dir}/filtered.out",
#                             "../../dataset/")
#         except Exception as e:
#             print(e)
#             print(f"Failed grasp filtering on {obj_dir}")


filter_and_score("../7310_init/all.npz", 7310,
                 "../7310_init/info.json",
                 "../7310_init/raw_grasp.out",
                 "../7310_init/filtered.out",
                 "../../dataset/")