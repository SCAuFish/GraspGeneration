import numpy as np
import sapien.core as sapien
from evaluator import EnvFreeRevoluteEvaluator
import open3d
import os
from typing import List
from collections import defaultdict
import json

def eval_in_pointcloud():
    evaluator = EnvFreeRevoluteEvaluator("35059")
    evaluator.eval_and_visualize("link_0", 0.2)

def get_paired_score(p1: int, p2: int, unit_torque_force: np.ndarray, force_along_normal: np.ndarray, force_ortho: np.ndarray, friction_coef=0.7):
    ''' Hypothesis: if two contact points are exerting force, the sum of two minimum normal forces are always greater than the
                    minimum normal force of only using one of the contact point.
    '''
    # Consider first rotation direction: require negative force along normal (rotating along axis)
    nf1, nf2 = None, None
    nf1 = unit_torque_force[p1] / (np.abs(np.clip(force_along_normal[p1], -10000, 0)) + friction_coef * np.abs(force_ortho[p1]))
    nf2 = unit_torque_force[p2] / (np.abs(np.clip(force_along_normal[p2], -10000, 0)) + friction_coef * np.abs(force_ortho[p2]))

    dir1nf = min(nf1, nf2)

    # Consider the second dirction: require positive force along normal (rotating anti axis)
    nf1, nf2 = None, None
    nf1 = unit_torque_force[p1] / (np.clip(force_along_normal[p1], 0, 10000) + friction_coef * np.abs(force_ortho[p1]))
    nf2 = unit_torque_force[p2] / (np.clip(force_along_normal[p2], 0, 10000) + friction_coef * np.abs(force_ortho[p2]))

    dir2nf = min(nf1, nf2)

    return (dir1nf, dir2nf)


def filter_grasp_candidate(pcd_file, proposal_file, obj_id, link_name, friction_coef=0.7, filter_ratio=0.1):
    ''' Use this method to filter proposals by feeding them through an object evaluator
        And only keep the best ones. The number of grasps outputted is calculated as: output_num = input_num * filter_ratio
    '''
    # Create evaluator for the object
    evaluator = EnvFreeRevoluteEvaluator(str(obj_id))

    # 1. Load all proposed grasps
    pcd = open3d.io.read_point_cloud(pcd_file)
    points  = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    print("Loaded {} points from {}".format(len(points), pcd_file))

    proposal_reader = open(proposal_file)
    proposals = []
    for line in proposal_reader.readlines():
        parts  = line.split("#")
        p1_idx = int(parts[0])
        others = parts[-1].strip(" ,\n")
        others = [int(otherIdx) for otherIdx in others.split(",")]

        proposals += [(p1_idx, other_idx) for other_idx in others]

    # 2. Evaluate all these proposals
    joint, xyz, axis = evaluator.find_axis(link_name)
    unit_torque_force, force_along_normal, force_ortho = evaluator.eval_grasp(points, normals, xyz, axis)
    # If force_along_normal is negative, it is valid, meaning pushing inward. If it is positive, it is not valid in
    # soft-finger contact model
    scores = []
    mean_scores = []
    for proposal in proposals:
        p1, p2 = proposal
        
        score = get_paired_score(p1, p2, unit_torque_force, force_along_normal, force_ortho, friction_coef=friction_coef)
        scores.append(score)
        mean_scores.append(score[0] + score[1])

    best_ones = list(np.argsort(mean_scores)[:int(len(mean_scores) * filter_ratio)])
    print("Done score calculating")
    # Output to proposal format
    proposal_dict = defaultdict(list)
    for idx in best_ones:
        p1, p2 = proposals[idx]

        if p1 > p2:
            p1, p2 = p2, p1

        proposal_dict[p1].append(p2)

    proposal_writer = open("filtered_proposal.out", "w")
    for p in proposal_dict:
        proposal_writer.write("{}###".format(p))
        pairs = proposal_dict[p]
        for pair in pairs:
            proposal_writer.write("{}, ".format(pair))

        proposal_writer.write("\n")

    proposal_writer.close()

def filter_grasp_candidate_full_object(object_npy, objectId, grasp_proposal, info_json_file):
    # 1. Read in grasp proposals
    pointcloud = np.load(object_npy)
    points  = pointcloud['points']
    normals = pointcloud['normals']
    segmentation = pointcloud['segmentation']
    print("Loaded {} points from {}".format(len(points), object_npy))

    proposal_reader = open(grasp_proposal)
    proposals = []
    for line in proposal_reader.readlines():
        parts  = line.split("#")
        p1_idx = int(parts[0])
        others = parts[-1].strip(" ,\n")
        others = [int(otherIdx) for otherIdx in others.split(",")]

        proposals += [(p1_idx, other_idx) for other_idx in others]
    
    # 2. Use info_json to map joint name, link name to segmentation id
    movable_segments = []
    info_json = json.load(open(info_json_file))['info']


        
import sys
if __name__ == "__main__":
    pcd_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "../assets/35059_link0.pcd")
    proposal_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                sys.argv[1])
    filter_grasp_candidate(pcd_file_path, proposal_path, 35059, "link_0", filter_ratio=0.3, friction_coef=1.5)

    # pcd_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #                             "../35059_link0.pcd")
    # proposal_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #                             "../grasps_100cm.out")
    # filter_grasp_candidate(pcd_file_path, proposal_path, 35059, "link_0",friction_coef=0.3, filter_ratio=0.1)