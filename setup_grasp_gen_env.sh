# Takes input as the folder to work on
echo "Hello world" > /workspace/hello.txt

cp -r /cephfs/chs091/grasp_gen_conda/ /opt/conda/envs/
cp -r /cephfs/chs091/GraspGeneration/ /workspace/
cp -r /cephfs/chs091/vulkan/ /usr/share/
cd /workspace/GraspGeneration

echo "Hello again" >> /workspace/hello.txt

DATASET_PREFIX="/cephfs/chs091/dataset_part_"
conda run -n grasp_gen_conda python3 run.py $DATASET_PREFIX/$1 > /cephfs/chs091/log_$1.txt 2>&1
