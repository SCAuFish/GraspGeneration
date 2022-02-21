#include "PointCloud.h"
#include <vector>
#include <iostream>
void foo(std::vector<Point*>& v) {
    int s = v.size();
    return;
}
int main() {
    Point a(0, 0, 0, 1, 0, 0, 2);
    Point b(-1, 0, 0, -1, 0, 0, 2);
    Point c(1, 1, 1, 1, 0, 0, 2);

    std::vector<Point*> pList({&a, &b, &c});
    PointCloud pcd(pList, 2);

    pcd.generateGraspsSinglePoint(.7, 10, a);

    std::cout << "generated grasp: " << a.generated_grasp << std::endl;
    for (int index : a.getAntiPoints())
        std::cout << index << std::endl;
}