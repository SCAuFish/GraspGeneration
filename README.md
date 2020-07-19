# Intro
This folder contains key files that will be compiled and run to generate grasps.

# Compilation on Docker
Due to some reason, cmake on Docker keeps adding a `-fPIC` flag when compiling with nvcc, which is not accepted. Therefore, instead of using `cmake` and `make`, the compilation on Docker will be done with commands given in `grasp_src/compile.sh`