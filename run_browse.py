import os, sys
import random
import json
from collections import defaultdict
from vis_src.run import visualize_grasps

if __name__ == "__main__":
    print("Starting server connection")
    started = os.system("kubectl create -f server_connecting_pod.yaml")
    if not os.path.isdir("nautilus_buffer"):
        os.mkdir("nautilus_buffer")
    print()

    print("Preparing dataset info")
    categories = defaultdict(list)
    for folder in os.listdir("../dataset"):
        if folder.isnumeric():
            f = json.load(open("../dataset/{}/meta.json".format(folder)))
            categories[f['model_cat'].lower()].append(folder)
    print()
    
    print("Here is a list of avaiable categories")
    print(list(categories.keys()))
    while True:
        print()
        user_input = input("Type in an object ID or a category: ")

        if user_input[0].lower() == 'q':
            break

        objId = -1
        if user_input.isnumeric():
            objId = int(user_input)
        else:
            if user_input in categories.keys():
                candidate_num = len(categories[user_input])
                objId = categories[user_input][random.randint(0, candidate_num-1)]
            else:
                print(f"{user_input} is not a valid id or category name")

        copied = os.system(f"kubectl cp chs091-connection:/cephfs/chs091/clouds/{objId}_1/all.pcd "
                           f"./nautilus_buffer/{objId}.pcd -n ucsd-haosulab")
        copied = os.system(f"kubectl cp chs091-connection:/cephfs/chs091/clouds/{objId}_1/filtered.out "
                           f"./nautilus_buffer/{objId}.out -n ucsd-haosulab")
        copied = os.system(f"kubectl cp chs091-connection:/cephfs/chs091/clouds/{objId}_1/raw_grasp.out "
                           f"./nautilus_buffer/{objId}_raw.out -n ucsd-haosulab")
        if copied != 0:
            print(f"Object {objId} does not exist") 

        print("Displaying")
        visualize_grasps(f"./nautilus_buffer/{objId}.pcd", f"./nautilus_buffer/{objId}_raw.out")
        visualize_grasps(f"./nautilus_buffer/{objId}.pcd", f"./nautilus_buffer/{objId}.out")


    print()
    print("Shutting down connection")
    # os.system("kubectl delete pod chs091-connection -n ucsd-haosulab")