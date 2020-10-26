#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#include <vector>
#include <tuple>
#include <stdexcept>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cstring>
//#include <Eigen/LU>

class Point{
    public:
    float x, y, z;
    float nx, ny, nz;
    int index;
    int   generated_grasp, filteredGraspNum, candidateNum;

    // Generated antipodal points and their scores
    int*  antiPoints;
    float* scores;
    float3* directions;
    float  worst_score;
    
    CUDA_HOSTDEV void addAntipodal(int index, float score, int candidateNum, float3 noCollisionDir);
    std::vector<int> getAntiPoints();
    std::vector<float> getAntiScores();
//    std::vector<Eigen::Vector3f> getAntiDirs();
    Point(float x, float y, float z, float nx, float ny, float nz, int candidateNum);
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
    PointCloud(std::vector<Point*>& pointList, int candidateNum);

    void generateGraspsBrute(float friction_coef, float jaw_span);
    void generateGraspsSinglePoint(float friction_coef, float jaw_span, Point& p);
    void filterGraspsByNeighbor();

    ~PointCloud();
};
#endif