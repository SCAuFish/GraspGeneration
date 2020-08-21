# Takes input as the folder to work on
echo "Hello world" > /workspace/hello.txt

cp -r /cephfs/chs091/grasp_gen_conda/ /opt/conda/envs/
cp -r /cephfs/chs091/GraspGeneration/ /workspace/
cp -r /cephfs/chs091/vulkan/ /usr/share/
cd /workspace/GraspGeneration

echo "Hello again" >> /workspace/hello.txt

conda run -n grasp_gen_conda python3 run.py $1
#> /cephfs/chs091/log.txt 2>&1
