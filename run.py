# This is the start point of grasp generation pipeline
import os, sys
from pcloud_src.data_generation import generate_point_cloud, CLOUD_DIR
from grasp_src.utils import filter_and_score
from categories import safe, storage_mixed, storage_prismatic, storage_revolute, dishwasher
import argparse
import socket

ASSET_DIR = '/cephfs/chs091/dataset'
if socket.gethostname() == 'AuFish2020':
    ASSET_DIR = '/home/aufish/Documents/grasp_data/dataset/'

parser = argparse.ArgumentParser()
parser.add_argument("--objects", help="the objects to generate grasps on. Can be categories predefined")
parser.add_argument("--skip_pcd", help="Skip the generation of pointcloud. Turn on only if pcd already exists.",
                    action="store_true")
args = parser.parse_args()

# if len(sys.argv) > 1:
#     # when running locally
#     ASSET_DIR = sys.argv[1]
# print("Using objects in {}".format(ASSET_DIR))
# objects = os.listdir(ASSET_DIR)

object_list = []
if args.objects == 'safe':
    object_list = safe
elif args.objects == 'dishwasher':
    object_list = dishwasher
elif args.objects == 'storage_revolute':
    object_list = storage_revolute
elif args.objects == 'storage_prismatic':
    object_list = storage_prismatic
elif args.objects == 'storage_mixed':
    object_list = storage_mixed

print("Skipping pcd configure: {}".format(args.skip_pcd))
for i in range(len(object_list)):
    objId = object_list[i].split("_")[0]
    print("Generating on ID {} -- {}/{}".format(object_list[i], i + 1, len(object_list)))

    # 2. Generate pcd file for a point cloud
    # add more pose with other seeds here
    seeds = [1]
    for i in range(len(seeds)):
        objDir = "{}_{}".format(objId, seeds[i])
        if os.path.exists("{}/{}/raw_grasp.out".format(CLOUD_DIR, objDir)):
            # skip generated grasps
            print(f"skipped generation of {objDir}")
            continue
        if not args.skip_pcd:
            # generate_point_cloud(objId, seeds[i], dataset_dir = ASSET_DIR, render_collision=True)
            command = "python3 pcloud_src/data_generation.py --object_id {} " \
                      "--pose_id {} --dataset_dir {} --output_to {} " \
                      "--scale {}".format(objId, seeds[i], ASSET_DIR, CLOUD_DIR, 1)
            exit_code = os.system(command)
            if exit_code != 0:
                print("Failed pointcloud generation on {}".format(objId))
                break

        # 3. Generate raw grasps
        os.system(f'touch {CLOUD_DIR}/{objDir}/raw_grasp.out')
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
