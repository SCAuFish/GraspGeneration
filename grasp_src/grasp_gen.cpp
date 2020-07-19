#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "PointCloud.h"

// class PointCloud{
//     public:
//     int size;
//     Point* cloud;
// }

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointNormal>* cloud = new pcl::PointCloud<pcl::PointNormal>;
  std::cout << "Reading file: " << argv[1] << std::endl;
  if (pcl::io::loadPCDFile<pcl::PointNormal>(argv[1], *cloud) == -1){
    PCL_ERROR("couldn't read file test.pcd\n");
    return -1;
  }

  Point* pCloud = (Point*) malloc(cloud->points.size() * sizeof(Point));
  for (int i = 0; i < cloud->points.size(); i++){
    Point p;
    p.x  = cloud->points[i].x;
    p.y  = cloud->points[i].y;
    p.z  = cloud->points[i].z;
    p.nx = cloud->points[i].normal_x;
    p.ny = cloud->points[i].normal_y;
    p.nz = cloud->points[i].normal_z;
    p.index = i;
    p.generated_grasp = 0;
    p.filteredGraspNum = 0;

    pCloud[i] = p;
  }

  PointCloud p(pCloud, cloud->points.size(), 10);
  p.generateGraspsBrute(0.7, 0.1);
  std::cout << "done grasp generation" << std::endl;

  // Write to file
  std::ofstream output_file;
  std::cout << "Writing to :" << argv[2] << std::endl;
  output_file.open(argv[2]);
  for (int i = 0; i < p.size; i++){
    if (p.cloud[i].filteredGraspNum == 0){
      continue;
    }
    Point& point = p.cloud[i];
    output_file << i << "#(" << point.x << ", " << point.y << ", " << point.z << ")#";
    output_file << "(" << point.nx << ", " << point.ny << ", " << point.nz << ")";
    output_file << "# ";
    for (int j = 0; j < point.filteredGraspNum; j++){
      output_file << point.antiPoints[j] << " (" << point.directions[j].x << ", " << point.directions[j].y << ", " << point.directions[j].z << ")" << " | ";
    } 
    output_file << std::endl;
  }
  output_file.close();

  return (0);
}