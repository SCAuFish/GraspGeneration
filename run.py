# This is the start point of grasp generation pipeline
import os, sys
from pcloud_src.data_generation import generate_point_cloud
from grasp_src.utils import filter_fixed_parts

ASSET_DIR = '/cephfs/chs091/datasets'
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
        if os.path.isdir("/cephfs/chs091/clouds/{}".format(objDir)):
            # skip generated grasps
            continue
        generate_point_cloud(objId, seeds[i], dataset_dir = ASSET_DIR)

        # 3. Generate raw grasps
        os.system('./grasp_src/grasp_gen /cephfs/chs091/clouds/{}/all.pcd /cephfs/chs091/clouds/{}/raw_grasp.out'.format(objDir, objDir))
        print("Done raw grasp generation")

        # 4. Filter grasps that are on fixed links. Back to 2 
        filter_fixed_parts("/cephfs/chs091/clouds/{}/all.npz".format(objDir), objId, 
                           "/cephfs/chs091/clouds/{}/info.json".format(objDir), 
                           "/cephfs/chs091/clouds/{}/raw_grasp.out".format(objDir), 
                           "/cephfs/chs091/clouds/{}/filtered.out".format(objDir),
                           ASSET_DIR)