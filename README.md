# Intro
This folder contains key files that will be compiled and run to generate grasps.

# Compilation on Docker
Due to some reason, cmake on Docker keeps adding a `-fPIC` flag when compiling with nvcc, which is not accepted. Therefore, instead of using `cmake` and `make`, the compilation on Docker will be done with commands given in `grasp_src/compile.sh`

# Features to add in next version
1. Generate with collision body instead of visual body
2. Save qpose instead of joint global pose in info.json
3. Filter grasps on each part and save them seperately (Some related work is going on in eval_src)