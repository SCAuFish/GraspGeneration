# This is the start point of grasp generation pipeline
import os, sys
from pcloud_src.data_generation import generate_point_cloud, CLOUD_DIR
from grasp_src.utils import filter_and_score

ASSET_DIR = '/cephfs/chs091/dataset'
if len(sys.argv) > 1:
    # when running locally
    ASSET_DIR = sys.argv[1]
objects = os.listdir(ASSET_DIR)

# 4. Filter grasps that are on fixed links. Back to 2 
objDir = "39628_1"
objId  = 39628
filter_and_score("{}/{}/all.npz".format(CLOUD_DIR, objDir), objId, 
    "{}/{}/info.json".format(CLOUD_DIR, objDir), 
    "{}/{}/raw_grasp.out".format(CLOUD_DIR, objDir), 
    "{}/{}/filtered.out".format(CLOUD_DIR, objDir),
    ASSET_DIR)
