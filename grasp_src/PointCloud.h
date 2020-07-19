#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <vector>
#include <cuda.h>
#include <cuda_profiler_api.h>

class Point{
    public:
    float x, y, z;
    float nx, ny, nz;
    float normal_length_square;
    int   gridX, gridY, gridZ;
    int   index;
    int   generated_grasp, filteredGraspNum;

    // Generated antipodal points and their scores
    int*  antiPoints;
    float* scores;
    float3* directions;
    float  worst_score;
    
    CUDA_HOSTDEV void addAntipodal(int index, float score, int candidateNum, float3 noCollisionDir);
    ~Point();
};


/**
 * A class that separates points into grids
 */ 
class PointCloud{
    public:
    Point* cloud;
    int    size;
    int    xDim, yDim, zDim;
    float  xMin, yMin, zMin, xMax, yMax, zMax;
    int    candidateNum;

    PointCloud(Point* pointList, int size, int candidateNum);

    void generateGraspsBrute(float friction_coef, float jaw_span);
    void filterGraspsByNeighbor();

    ~PointCloud();
};
#endif