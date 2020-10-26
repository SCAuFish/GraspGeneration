#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include "PointCloud.h"
#include <limits>
#include <algorithm>
#include "cublas_v2.h"
#include "math_constants.h"
#include <queue>
#include <utility>
#include <assert.h>

typedef std::tuple<int, float, float3> p_score;
/**
 * Given the two points: p1, p2 and axis info represented by axisOrigin and direction
 * x, y, z, calculate the score of the grasp
 * All points and axis info should be in the same frame
 */
 __device__
 float dotProduct(float3 v1, float3 v2){
     return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; 
 }
 
 __device__
 float3 crossProduct(float3 v1, float3 v2){
     float x = v1.y * v2.z - v1.z * v2.y;
     float y = v1.z * v2.x - v1.x * v2.z;
     float z = v1.x * v2.y - v1.y * v2.x;
 
     return make_float3(x, y, z);
 }

 __device__
 float norm(float3 v){
     return norm3df(v.x, v.y, v.z);
 }

 __device__
 float3 operator+(const float3 &a, const float3 &b){
     return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
 }

 __device__
 float3 operator/(const float3 &a, const float &b){
     return make_float3(a.x / b, a.y / b, a.z / b);
 }

 __device__
 float3 operator*(const float3 &a, const float &b){
     return make_float3(a.x * b, a.y * b, a.z * b);
 }


/**
 * points: points with generated grasps
 * pointnum: size of poitns
 * threshold: each point should have at least this number of neighbors with valid grasps
 */
__global__
void filterGraspsByNeighborBrute(Point* points, int point_num, float threshold){
    // In a brute force way, it is an O(k^2*n^2) algorithm, looping through all grasps, then testing all neighbors' grasps
    int point_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride      = blockDim.x * gridDim.x;

    for (int i = point_index; i < point_num; i += stride){
        Point& curr = points[i];

        int normalSimilarGrasps = 0;
        int neighborNum = 0;
        for (int j = 0; j < point_num; j++) {
            Point& another = points[j];

            float diff_x, diff_y, diff_z;
            diff_x = another.x - curr.x;
            diff_y = another.y - curr.y;
            diff_z = another.z - curr.z;

            if (norm3df(diff_x, diff_y, diff_z) > 0.01) continue;

            neighborNum += 1;
            for (int currIter = 0; currIter < curr.generated_grasp; currIter++){
                for (int anotherIter = 0; anotherIter < another.generated_grasp; anotherIter++){
                    Point& currGraspPair    = points[curr.antiPoints[currIter]];
                    Point& anotherGraspPair = points[another.antiPoints[anotherIter]];
                    float3 currGraspDir     = make_float3(currGraspPair.x-curr.x, currGraspPair.y-curr.y, currGraspPair.z-curr.z);
                    float3 anotherGraspDir  = make_float3(anotherGraspPair.x-another.x, anotherGraspPair.y-another.y, anotherGraspPair.z-another.z);

                    // Count the grasp as a back-up if the cosine between them is larger than 0.7
                    if (dotProduct(currGraspDir, anotherGraspDir) / (norm(currGraspDir) * norm(anotherGraspDir)) > 0.9){
                        normalSimilarGrasps += 1;
                    }
                }
            }
        }        

        // curr.filteredGraspNum = normalSimilarGrasps > threshold ? curr.generated_grasp : 0;
        if (normalSimilarGrasps < threshold) {
            curr.filteredGraspNum = 0;
        } 
    }
}

__device__
// Return the direction if there is a direction without collision, or return (0, 0, 0)
float3 collidedWithGripper(Point* points, int size, Point& p1, Point& p2, float aabbInnerRadius, float aabbOuterRadius, float gripperHeight){
    // Find the mid point between two contacts
    float3 mid  = make_float3((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);
    float3 zAxis= make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    // Define a grasp coordinate, with mid-point as origin, grasp axis (p1-p2) as positive z, arbitrarily
    // define an axis as x, and calculate y
    float3 temp = make_float3(zAxis.x + 1, zAxis.y, zAxis.z);
    if (zAxis.y < 0.0000001 && zAxis.z == 0.0000001){
        temp.y = temp.y + 1;
    }
    float3 xAxis= crossProduct(zAxis, temp);

    float3 yAxis= crossProduct(zAxis, xAxis);
    
    xAxis = xAxis / norm(xAxis);
    yAxis = yAxis / norm(yAxis);
    zAxis = zAxis / norm(zAxis);

    // Loop through all points and find if at least one slice is free of collision
    bool collisionSlice[8];
    // bool collisionSliceInner[4];
    for (int i = 0; i < 8; i++){
        collisionSlice[i] = false;
    }
    for (int i = 0; i < size; i++){
        // Calculate the length from point to the axis
        Point& p = points[i];
        float3 pToMid = make_float3(p.x - mid.x, p.y - mid.y, p.z - mid.z);

        float pToMidOnX = dotProduct(pToMid, xAxis);
        float pToMidOnY = dotProduct(pToMid, yAxis);

        // If the point is out of outer radius, it won't collide with the gripper
        if (norm3df(pToMidOnX, pToMidOnY, 0) > aabbOuterRadius){
            continue;
        }

        float pToMidOnZ = dotProduct(pToMid, zAxis);
        pToMidOnZ = abs(pToMidOnZ);
        if (pToMidOnZ > gripperHeight) {
            // If the point is too far from the grasp, ignore it
            continue;
        }

        // If the point is inside of the inner radius, check whether it collides with the finger, by calculating the
        // distance from origin along axis direction. It has to be within the distance of two contact points
        if (norm3df(pToMidOnX, pToMidOnY, 0) < aabbInnerRadius){
            // float checkRadius = 0.01;
            float epsilon     = 0.005;
            float dist_between_contacts = norm(make_float3(p2.x-p1.x, p2.y-p1.y, p2.z-p1.z));
            if ( pToMidOnZ < (dist_between_contacts/2 + epsilon) ) continue;

            float dist1 = norm3df(p.x - p1.x, p.y - p1.y, p.z - p1.z);
            float dist2 = norm3df(p.x - p2.x, p.y - p2.y, p.z - p2.z);

            if (dist1 > gripperHeight && dist2 > gripperHeight) continue;
        }

        // Divide into 10 regions, if one of them is collision-free, treat the grasp as valid
        // This number '10' should vary according to the depth of the gripper's collision body and the distance from
        // the collision body to the grasp center
        float cosAngleX = pToMidOnX / norm3df(pToMidOnX, pToMidOnY, 0);

        float angle = acosf(cosAngleX);
        if (pToMidOnY < 0) angle = 2 * CUDART_PI_F - angle;
        angle = angle >= 2 * CUDART_PI_F ? 0 : angle;

        collisionSlice[(int) (8 * angle / (2 * CUDART_PI_F))] = true;
    }

    int noCollisionSlice = -1;
    for (int i = 0; i < 8; i++){
        if ( collisionSlice[i] == false ){
            noCollisionSlice = i;
            break;
        }
    }
    if (noCollisionSlice == -1) return make_float3(0, 0, 0);
    else {
        float angle_floor = ((2 * CUDART_PI_F) * noCollisionSlice) / 8;
        float angle_ceil = ((2 * CUDART_PI_F) * (noCollisionSlice + 1)) / 8;
        float angle = (angle_ceil + angle_floor) / 2;
        float3 dir  = ( xAxis * cosf(angle) + yAxis * sinf(angle) );
        return dir;
    }
}

// util functions for generating and filtering grasps
/**
 * aabbInnerRadius and aabbOuterRadius defines a revoluted body generated by an AABB rotating along the
 * grasp axis (axis connecting two end effectors).
 */
__global__
void generateGraspBrute(Point* points, float friction_coef, float jaw_span, int point_num, int candidateNum,
    float aabbInnerRadius, float aabbOuterRadius, float gripperHeight){
    int point_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride      = blockDim.x * gridDim.x;
    for (int i = point_index; i < point_num; i+=stride){
        Point& curr = points[i];
        // Brute force loop through all points and check
        for (int j = i+1; j < point_num; j++){
            Point& another =  points[j];

            float diff_x, diff_y, diff_z;
            diff_x = another.x - curr.x;
            diff_y = another.y - curr.y;
            diff_z = another.z - curr.z;

            float square_norm = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
            if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > jaw_span * jaw_span){
                // too far for the jaw
                continue;
            }

            float angle1     = -diff_x * curr.nx - diff_y * curr.ny - diff_z * curr.nz;
            float cos_angle1 = angle1 / sqrtf(square_norm);
            float tan_angle1 = tanf(acosf(cos_angle1));

            float angle2     = diff_x * another.nx + diff_y * another.ny + diff_z * another.nz;
            float cos_angle2 = angle2 / sqrtf(square_norm);
            float tan_angle2 = tanf(acosf(cos_angle2));

            if ( cos_angle1 < 0.0001 || tan_angle1 > friction_coef || cos_angle2 < 0.0001 || tan_angle2 > friction_coef) {
                // Out of friction cone
                continue;
            } 
            else {
                float3 noCollisionDir = collidedWithGripper(points, point_num, curr, another, aabbInnerRadius, aabbOuterRadius, gripperHeight);
                if (norm(noCollisionDir) < 0.01){
                    continue;
                }
                else {
                    // Use the maximum angle as score -- the larger the worse
                    float score = tan_angle1 > tan_angle2 ? tan_angle1 : tan_angle2;
                    curr.addAntipodal(j, score, candidateNum, noCollisionDir);
                }
            }
        }
    }
}


/**
 * generate grasp on a single point
 * The scores for each point will be mapped into scores array
 * Then the reduction will be completed in the CPU
 */
__global__
void generateGraspSinglePoint(float* scores, float3* directions, Point* points, float friction_coef, float jaw_span, int point_num, int candidateNum,
    float aabbInnerRadius, float aabbOuterRadius, float gripperHeight, Point* p){
    int point_index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = point_index; i < point_num; i+= stride) {
        Point& another = points[i];
        scores[i]     = 100000;
 
        float diff_x, diff_y, diff_z;
        diff_x = another.x - p -> x;
        diff_y = another.y - p -> y;
        diff_z = another.z - p -> z;
 
        float square_norm = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        if (square_norm > jaw_span * jaw_span){
            // too far for the jaw
            continue;
        }
 
        float angle1     = -diff_x * p -> nx - diff_y * p -> ny - diff_z * p -> nz;
        float cos_angle1 = angle1 / sqrtf(square_norm);
        float tan_angle1 = tanf(acosf(cos_angle1));
 
        float angle2     = diff_x * another.nx + diff_y * another.ny + diff_z * another.nz;
        float cos_angle2 = angle2 / sqrtf(square_norm);
        float tan_angle2 = tanf(acosf(cos_angle2));
 

        if (square_norm < 0.001 && abs(angle1 + angle2) < 0.001) {
            //same point, or if they are too close
            continue;
        }
        else if ( cos_angle1 < 0.1 || tan_angle1 > friction_coef || cos_angle2 < 0.1 || tan_angle2 > friction_coef) {
            // Out of friction cone
            continue;
        } 
        else {
            float3 noCollisionDir = collidedWithGripper(points, point_num, *p, another, aabbInnerRadius, aabbOuterRadius, gripperHeight);
            if (norm(noCollisionDir) < 0.01){
                continue;
            }
            else {
                // Use the maximum angle as score -- the larger the worse
                float score = tan_angle1 > tan_angle2 ? tan_angle1 : tan_angle2;
                scores[i] = score;
                directions[i] = noCollisionDir;
            }
        }

        // if ( cos_angle1 < 0.0001 || tan_angle1 > friction_coef || cos_angle2 < 0.0001 || tan_angle2 > friction_coef) {
        //     // Out of friction cone
        //     continue;
        // } 
        // else {
        //     float3 noCollisionDir = collidedWithGripper(points, point_num, curr, another, aabbInnerRadius, aabbOuterRadius, gripperHeight);
        //     if (norm(noCollisionDir) < 0.01){
        //         continue;
        //     }
        //     else {
        //         // Use the maximum angle as score -- the larger the worse
        //         float score = tan_angle1 > tan_angle2 ? tan_angle1 : tan_angle2;
        //         curr.addAntipodal(j, score, candidateNum, noCollisionDir);
        //     }
        // }
    } 
}


__host__ __device__
void Point::addAntipodal(int index, float score, int candidateNum, float3 noCollisionDir){
    // Manage the priority queue here
    // Score is the larger angle
    // printf("min score: %f, current score: %f\n", this->worst_score, score);

    if (generated_grasp == 0){
        this->antiPoints[this->generated_grasp] = index;
        this->scores    [this->generated_grasp] = score;
        this->directions[this->generated_grasp] = noCollisionDir;
        this->generated_grasp += 1;

        this->worst_score = score;
    }

    else if (this->generated_grasp < candidateNum) {
        this->antiPoints[this->generated_grasp] = index;
        this->scores    [this->generated_grasp] = score;
        this->directions[this->generated_grasp] = noCollisionDir;
        this->generated_grasp += 1;

        this->worst_score = score > this->worst_score ? score : this->worst_score;
    }

    else if (score < this->worst_score){
        float new_worst_score = score;
        for (int i = 0; i < candidateNum; i++){
            if (this->scores[i] == this->worst_score) {
                // Insert point to replace worst score
                this->scores[i] = score;
                this->antiPoints[i] = index;
                this->directions[i] = noCollisionDir;
            } else {
                // Find new worst score
                new_worst_score = this->scores[i] > new_worst_score ? this->scores[i] : new_worst_score;
            }
        }

        this->worst_score = new_worst_score;
    }

    this->filteredGraspNum = this->generated_grasp;
}

/**
 * A wrapper function to return antipodal points saved on this point
 */
std::vector<int> Point::getAntiPoints() {
    std::vector<int> result;

    int* candidate_iter = this -> antiPoints;
    for (int i = 0; i < this -> generated_grasp; i++) {
        result.push_back(candidate_iter[i]);
    }

    return result;
}

/**
 * A wrapper function to return antipodal scores saved on this point
 */
 std::vector<float> Point::getAntiScores() {
    std::vector<float> result;

    float* candidate_iter = this -> scores;
    for (int i = 0; i < this -> generated_grasp; i++) {
        result.push_back(candidate_iter[i]);
    }

    return result;
}

/**
 * A wrapper function to return antipodal directions saved on this point
 */
 std::vector<Eigen::Vector3f> Point::getAntiDirs() {
    std::vector<Eigen::Vector3f> result;

    float3* candidate_iter = this -> directions;
    for (int i = 0; i < this -> generated_grasp; i++) {
        Eigen::Vector3f dir;
        dir << candidate_iter[i].x, candidate_iter[i].y, candidate_iter[i].z;
        result.push_back(dir);
    }

    return result;
}

Point::Point(float x, float y, float z, float nx, float ny, float nz, int candidateNum) : 
    x(x), y(y), z(z), nx(nx), ny(ny), nz(nz), generated_grasp(0), filteredGraspNum(0), candidateNum(candidateNum){
        this -> antiPoints = (int*) malloc(sizeof(int) * candidateNum);
        this -> scores = (float*) malloc(sizeof(float) * candidateNum);
        this -> directions = (float3*) malloc(sizeof(float3) * candidateNum);
    }

Point::~Point(){
    if (this->antiPoints != nullptr)
        free(this->antiPoints);
    if (this->scores != nullptr)
        free(this->scores);
    if (this->directions != nullptr)
        free(this->directions);
}

PointCloud::PointCloud(Point* points, int size, int candidateNum){
    this->size = size;
    this->candidateNum = candidateNum;
    // 1. constructor: put points on to cuda memory
    cudaMallocManaged(&(this->cloud), size * sizeof(Point));
    cudaMemcpy(this->cloud, points, size * sizeof(Point), cudaMemcpyHostToDevice);
    for (int i = 0; i < size; i++){
        // Allocate space for 10 points to store antipodal grasp
        cudaMallocManaged( &(this->cloud[i].antiPoints), candidateNum * sizeof(int));
        cudaMallocManaged( &(this->cloud[i].scores), candidateNum * sizeof(float));
        cudaMallocManaged( &(this->cloud[i].directions), candidateNum * sizeof(float3));
    }

    // 2. Create grid depending on points distribution
    this->xMin = std::numeric_limits<float>::max();
    this->yMin = std::numeric_limits<float>::max();
    this->zMin = std::numeric_limits<float>::max();
    this->xMax = std::numeric_limits<float>::min();
    this->yMax = std::numeric_limits<float>::min();
    this->zMax = std::numeric_limits<float>::min();
    
    for (int i = 0; i < size; i++){
        Point& p = points[i];

        this->xMin = std::min(this->xMin, p.x);
        this->yMin = std::min(this->yMin, p.y);
        this->zMin = std::min(this->zMin, p.z);

        this->xMax = std::max(this->xMax, p.x);
        this->yMax = std::max(this->yMax, p.y);
        this->zMax = std::max(this->zMax, p.z);
    }
    
    std::cout << "Populated points: " << size << std::endl;
}

/**
 *  An overloaded version of constructor that may work with pybind using vector
 * instead of raw array
 */
PointCloud::PointCloud(std::vector<Point*>& pointList, int candidateNum){
    // create space for antiPoints for each point passed in
    for (int i = 0; i < pointList.size(); i++) {
        if (pointList[i] -> candidateNum != candidateNum){
            throw std::length_error("Candidate number mismatch for point and point cloud");
        }
    }
    std::cout << "here" << std::endl;
    Point* points = (Point*) malloc(sizeof(Point) * pointList.size());
    for (int i = 0; i < pointList.size(); i++){
        std::memcpy(&points[i], pointList[i], sizeof(Point));
    }
    int size = pointList.size();

    this->size = size;
    this->candidateNum = candidateNum;
    // 1. constructor: put points on to cuda memory
    cudaMallocManaged(&(this->cloud), size * sizeof(Point));
    cudaMemcpy(this->cloud, points, size * sizeof(Point), cudaMemcpyHostToDevice);
    for (int i = 0; i < size; i++){
        // Allocate space for 10 points to store antipodal grasp
        cudaMallocManaged( &(this->cloud[i].antiPoints), candidateNum * sizeof(int));
        cudaMallocManaged( &(this->cloud[i].scores), candidateNum * sizeof(float));
        cudaMallocManaged( &(this->cloud[i].directions), candidateNum * sizeof(float3));
    }

    // 2. Create grid depending on points distribution
    this->xMin = std::numeric_limits<float>::max();
    this->yMin = std::numeric_limits<float>::max();
    this->zMin = std::numeric_limits<float>::max();
    this->xMax = std::numeric_limits<float>::min();
    this->yMax = std::numeric_limits<float>::min();
    this->zMax = std::numeric_limits<float>::min();
    
    for (int i = 0; i < size; i++){
        Point& p = points[i];

        this->xMin = std::min(this->xMin, p.x);
        this->yMin = std::min(this->yMin, p.y);
        this->zMin = std::min(this->zMin, p.z);

        this->xMax = std::max(this->xMax, p.x);
        this->yMax = std::max(this->yMax, p.y);
        this->zMax = std::max(this->zMax, p.z);
    }
    
    free(points);
    std::cout << "Populated points: " << size << std::endl;
}

void PointCloud::generateGraspsBrute(float friction_coef, float jaw_span){
    generateGraspBrute<<<128, 256>>>(this->cloud, friction_coef, jaw_span, this->size, this->candidateNum, 0.1, 0.2, 0.2);
    cudaDeviceSynchronize();
}

void PointCloud::generateGraspsSinglePoint(float friction_coef, float jaw_span, Point& p) {
    float* scores;
    float3* directions;
    Point* givenPoint;

    cudaMallocManaged(&scores, this->size * sizeof(float));
    cudaMallocManaged(&directions, this->size * sizeof(float3));
    cudaMallocManaged(&givenPoint, sizeof(Point));
    cudaMemcpy(givenPoint, &p, sizeof(Point), cudaMemcpyHostToDevice);
    generateGraspSinglePoint<<<128, 256>>>(scores, directions, this -> cloud, friction_coef, jaw_span, 
        this->size, this->candidateNum, 0.03, 0.2, 0.2, givenPoint);
    cudaDeviceSynchronize();

    // max heap but we want the points with smallest score
    auto compare = [](p_score& s1, p_score& s2) {return std::get<1>(s1) > std::get<1>(s2);};
    std::priority_queue<p_score, std::vector<p_score>, decltype(compare)> sorted_scores(compare);
    for (int i = 0; i < this->size; i++){
        sorted_scores.push(std::make_tuple(i, scores[i], directions[i]));
    }

    p.generated_grasp = 0;
    while (p.generated_grasp < this->candidateNum && sorted_scores.size() > 0){
        p_score curr_best = sorted_scores.top();
        sorted_scores.pop();

        if (&this->cloud[std::get<0>(curr_best)] == &p) {
            // The candidate is the query point itself
            continue;
        }

        if (abs(std::get<1>(curr_best)-100000) < 1) break;
        p.antiPoints[p.generated_grasp] = std::get<0>(curr_best);
        p.scores[p.generated_grasp] = std::get<1>(curr_best);
        p.directions[p.generated_grasp] = std::get<2>(curr_best);
        p.generated_grasp += 1;
    }
}


// Filter grasps, such that neighbors of current points should also have valid grasp
void PointCloud::filterGraspsByNeighbor(){
    filterGraspsByNeighborBrute<<<128, 256>>>(this->cloud, this->size, 1000);
    cudaDeviceSynchronize();
}

PointCloud::~PointCloud(){
    for (int i = 0; i < this-> size; i++) {
        cudaFree(this->cloud[i].antiPoints);
        cudaFree(this->cloud[i].scores);
        cudaFree(this->cloud[i].directions);
    }
    cudaFree(this->cloud);
}
