from AntiGrasp import *

a = Point(0,0,0,1,0,0,5)
b = Point(-1,0,0,-1,0,0,5)

pcd = PointCloud([a, b], 5)

pcd.generate_grasp_single_point(0.7, 2, a)

print("finished generating")
# print(a.generated_grasp)
print(a.get_antipoints())
