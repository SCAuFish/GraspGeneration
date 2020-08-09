# Given information of an object's joints and links, calculate the
# quality of a grasp

import sapien.core as sapien
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm as norm
import open3d

class BaseEvaluator:
    def __init__(self):
        pass

    def eval_grasp(self, contact_points):
        pass

ASSET_DIR = "../../dataset/"
CLOUD_DIR = "../../pc_gen/clouds/"
class EnvFreeRevoluteEvaluator(BaseEvaluator):
    def __init__(self, objectId):
        object_urdf_path = ASSET_DIR + objectId
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                object_urdf_path+"/mobility.urdf")
        self.objectID    = objectId
        self.object_root = ET.parse(filepath)

    def eval_grasp(self, contact_point, contact_normal, axisPoint, axisDir):
        ''' Take a one contact point, evaluate the contact point

            ? Maybe should also evaluate how much to exert along the contact axis to achieve unit
            force along surface normal

            params: contact_point - 3D coordinates of contact point
                    contact_normal - the normal direction at the contact point
                    axis - axis of the joint, should be in the same frame as contact_point and contact_normal
                        axisPoint is the point through which the axis passes
                        axisDir is the direction of the axis (3x1 vector)
                    grasp_link_name - the id of the link that the contact points are on

            return: force to generate unit torque, projection of unit force on contact normal,
                    projection of unit force on contact tangent
        '''
        # 0. Check dimensions, each row is a point/normal
        assert contact_normal.shape[1] == 3
        assert contact_point.shape[1]  == 3
        axisPoint     = axisPoint.reshape(1, 3)
        axisDir       = axisDir.reshape(1, 3)
        # 1. use origin_in_world and direction_in_world to calculate force arm to the contact points
        v1 = axisDir
        v2 = (axisPoint - contact_point)
        # axis=1 so that the norm is calculated per row
        force_arm = norm(np.cross(v1, v2), axis=1) / norm(v1)

        # 2. use force arm to calculate unit force on contact point
        unit_torque_force = 1 / force_arm

        # 3. Evaluate how much force is required along and orthogonal to contact normal to create unit force
        # orthogonal to force_arm
        # 3.1 Find force-arm vector
        unit_direction = axisDir / norm(axisDir)
        force_arm_vector_origin = axisPoint + ((contact_point - axisPoint.reshape(1, 3)) @ unit_direction.transpose()) * unit_direction
        force_arm_vector        = contact_point - force_arm_vector_origin

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

    def find_axis(self, part):
        ''' Get axis point and axis direction from URDF file.
        '''
        robot_root = self.object_root
        for joint in robot_root.findall("joint"):
            if joint.get('type') in ['prismatic', 'revolute', 'continuous']:
                if joint.find('child').get('link') == part:
                    axis = joint.find('axis')
                    if axis is None:
                        axis = '1 0 0'
                    else:
                        axis = axis.get('xyz')
                    axis = np.array([float(x) for x in axis.split()])
                    assert axis.shape == (3, )
                    link = [l for l in robot_root.findall('link') if l.get('name') == part][0]
                    col = link.find('collision')
                    origin = col.find('origin')
                    xyz = np.array([float(x) for x in origin.get('xyz').split()])
                    assert xyz.shape == (3, )
                    assert origin.get('rpy') is None
                    if joint.get('type') == 'prismatic':
                        xyz = np.array([0, 0, 0])
                    return joint.get('type'), -xyz, axis
        return None, None, None


class WholeObjectEvaluator(BaseEvaluator):
    def __init__(self, objectId):
        object_urdf_path = ASSET_DIR + objectId
        filepath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                object_urdf_path+"/mobility.urdf")
        self.objectID    = objectId
        self.object_root = ET.parse(filepath)

    def eval_grasp(self, contact_point, contact_normal, axisPoint, axisDir):
        ''' Take a one contact point, evaluate the contact point

            ? Maybe should also evaluate how much to exert along the contact axis to achieve unit
            force along surface normal

            params: contact_point - 3D coordinates of contact point
                    contact_normal - the normal direction at the contact point
                    axis - axis of the joint, should be in the same frame as contact_point and contact_normal
                        axisPoint is the point through which the axis passes
                        axisDir is the direction of the axis (3x1 vector)
                    grasp_link_name - the id of the link that the contact points are on

            return: force to generate unit torque, projection of unit force on contact normal,
                    projection of unit force on contact tangent
        '''
        # 0. Check dimensions, each row is a point/normal
        assert contact_normal.shape[1] == 3
        assert contact_point.shape[1]  == 3
        axisPoint     = axisPoint.reshape(1, 3)
        axisDir       = axisDir.reshape(1, 3)
        # 1. use origin_in_world and direction_in_world to calculate force arm to the contact points
        v1 = axisDir
        v2 = (axisPoint - contact_point)
        # axis=1 so that the norm is calculated per row
        force_arm = norm(np.cross(v1, v2), axis=1) / norm(v1)

        # 2. use force arm to calculate unit force on contact point
        unit_torque_force = 1 / force_arm

        # 3. Evaluate how much force is required along and orthogonal to contact normal to create unit force
        # orthogonal to force_arm
        # 3.1 Find force-arm vector
        unit_direction = axisDir / norm(axisDir)
        force_arm_vector_origin = axisPoint + ((contact_point - axisPoint.reshape(1, 3)) @ unit_direction.transpose()) * unit_direction
        force_arm_vector        = contact_point - force_arm_vector_origin

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
                print(joint.find('child').get('link'))
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
        print("here")
        return None, None, None
