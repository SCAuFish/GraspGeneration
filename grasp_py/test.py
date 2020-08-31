from AntiGrasp import *

a = Point(0,0,0,1,0,0,1)
b = Point(-1,0,0,-1,0,0,1)

pcd = PointCloud([a, b], 1)

pcd.generate_grasp_single_point(0.7, 2, a)

print("finished generating")
print(a.generated_grasp)
# print(a.get_antipoints())
