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
        filepath = os.path.join(os.path.abspath(''),
                                object_urdf_path+"/mobility.urdf")
        self.objectID    = objectId
        self.object_root = ET.parse(filepath)

    def find_axis(self, link_name, joint2pose):
        ''' Get axis point and axis direction from URDF file.
            if the link is connected to a fixed link, return None
        '''
        robot_root = self.object_root
        for joint in robot_root.findall("joint"):
            if joint.get('type') in ['prismatic', 'revolute', 'continuous']:
                if joint.find('child').get('link') == link_name:
                    axis = joint.find('axis')
                    if axis is None:
                        axis = '1 0 0'
                    else:
                        axis = axis.get('xyz')
                    axis = np.array([float(x) for x in axis.split()])
                    assert axis.shape == (3, )
                    link = [l for l in robot_root.findall('link') if l.get('name') == link_name][0]
                    col = link.find('collision')
                    origin = col.find('origin')
                    xyz = np.array([float(x) for x in origin.get('xyz').split()])
                    assert xyz.shape == (3, )
                    assert origin.get('rpy') is None
                    if joint.get('type') == 'prismatic':
                        xyz = np.array([0, 0, 0])
                        
                    return joint.get('type'), -xyz, axis
        return None, None, None
    

def filter_fixed_parts(object_npz, objectId, info_json_file, grasp_proposal, output_file, ASSET_DIR):
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
    joint2pose     = {}
    for key in info_json:
        seg2link_joint[int(key)] = (info_json[key]['link_name'], info_json[key]['joint_name'])
        joint2pose[info_json[key]['joint_name']] = sapien.Pose(info_json[key]['joint_pose'][0], info_json[key]['joint_pose'][1])
        
    # 3. Create evaluator
    evaluator = WholeObjectEvaluator(str(objectId), ASSET_DIR)
    
    # 4. Evaluate based on each link
    proposal_writer = open(output_file, "w")
    for seg in seg2link_joint:
        joint, xyz, axis = evaluator.find_axis(seg2link_joint[seg][0], joint2pose)
        axis = np.array([0, 0, 1])
        if joint == None:
            continue
            
        print("Done score calculating")
        # Output to proposal format
        proposal_dict = defaultdict(list)
        for proposal in proposals:
            p1, p2, all_info = proposal

            proposal_dict[p1].append(all_info)

        total_selected = 0
        for p in proposal_dict:
            if segmentation[p] != seg:
                continue
            output_str = "{}###".format(p)
            pairs = proposal_dict[p]
            has_pair = False
            for pair in pairs:
                pairIdx = int(pair.strip().split(" ")[0])
                if segmentation[pairIdx] != seg:
                    continue
                output_str += "{} | ".format(pair)
                has_pair = True
                total_selected += 1

            if has_pair:
                proposal_writer.write(output_str + "\n")
            
    proposal_writer.close()