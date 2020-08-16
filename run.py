# This is the start point of grasp generation pipeline
import os, sys
from pcloud_src.data_generation import generate_point_cloud, CLOUD_DIR
from grasp_src.utils import filter_and_score

ASSET_DIR = '/cephfs/chs091/dataset'
if len(sys.argv) > 1:
    # when running locally
    ASSET_DIR = sys.argv[1]
objects = os.listdir(ASSET_DIR)

for i in range(len(objects)):
    objId = objects[i]
    print("Generating on ID {} -- {}/{}".format(objects[i], i+1, len(objects)))

    # 2. Generate pcd file for a point cloud
    # add more pose with other seeds here
    seeds = [1]
    for i in range(len(seeds)):
        objDir = "{}_{}".format(objId, seeds[i])
        if os.path.isdir("{}/{}".format(CLOUD_DIR, objDir)):
            # skip generated grasps
            continue
        # generate_point_cloud(objId, seeds[i], dataset_dir = ASSET_DIR, render_collision=True)
        exit_code = os.system("python3 pcloud_src/data_generation.py {} {} {}".format(objId, seeds[i], ASSET_DIR))
        if exit_code != 0:
            print("Failed pointcloud generation on {}".format(objId))
            break

        # 3. Generate raw grasps
        exit_code = os.system(f'./grasp_src/grasp_gen {CLOUD_DIR}/{objDir}/all.pcd {CLOUD_DIR}/{objDir}/raw_grasp.out')
        if exit_code != 0:
            print("Failed grasp generation on {}".format(objId))
            break

        print("Done raw grasp generation")

        # 4. Filter grasps that are on fixed links. Back to 2 
        try:
            filter_and_score("{}/{}/all.npz".format(CLOUD_DIR, objDir), objId, 
                           "{}/{}/info.json".format(CLOUD_DIR, objDir), 
                           "{}/{}/raw_grasp.out".format(CLOUD_DIR, objDir), 
                           "{}/{}/filtered.out".format(CLOUD_DIR, objDir),
                           ASSET_DIR)
        except:
            print("Failed grasp filtering on {}".format(objId))
            break
