# This is the start point of grasp generation pipeline
import os
from pcloud_src.data_generation import generate_point_cloud
from grasp_src.utils import filter_fixed_parts

# 1. Compile grasp generation program
print('Generation Started')
os.system('cmake ./grasp_src/CMakeLists.txt')
os.system('make -C ./grasp_src grasp_gen')
print('Done compilation')

ASSET_DIR = '../dataset'
objects = os.listdir(ASSET_DIR)

for i in range(len(objects)):
    objId = objects[i]
    print("Generating on ID {} -- {}/{}".format(objects[i], i+1, len(objects)))

    # 2. Generate pcd file for a point cloud
    # add more pose with other seeds here
    seeds = [1]
    for i in range(len(seeds)):
        generate_point_cloud(objId, seeds[i], dataset_dir = '../dataset')

    # 3. Generate raw grasps
    objDirs = ["{}_{}".format(objId, seed) for seed in seeds]
    for i in range(len(objDirs)):
        os.system('./grasp_src/grasp_gen ./clouds/{}/all.pcd ./clouds/{}/raw_grasp.out'.format(objDirs[i], objDirs[i]))

    print("Done raw grasp generation")

    # 4. Filter grasps that are on fixed links. Back to 2 
    for i in range(len(objDirs)):
        token = objDirs[i]
        filter_fixed_parts("./clouds/{}/all.npz".format(token), objId, 
                           "./clouds/{}/info.json".format(token), 
                           "./clouds/{}/raw_grasp.out".format(token), 
                           "./clouds/{}/filtered.out".format(token),
                           ASSET_DIR)

    break