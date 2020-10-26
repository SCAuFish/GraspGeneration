#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "PointCloud.h"

#include <string>
namespace py = pybind11;

PYBIND11_MODULE(antipodal, m) {
    py::class_<Point>(m, "Point")
        .def(py::init<float, float, float, float, float, float, int>())
        .def("get_antipoints", &Point::getAntiPoints)
        .def("get_antiscores", &Point::getAntiScores)
        .def("get_antidirs", &Point::getAntiDirs)
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z)
        .def_readwrite("nx", &Point::nx)
        .def_readwrite("ny", &Point::ny)
        .def_readwrite("nz", &Point::nz)
        .def_readonly("generated_grasp", &Point::generated_grasp)
        .def_readonly("filtered_grasp_num", &Point::filteredGraspNum);

    py::class_<PointCloud>(m, "PointCloud")
        .def(py::init<std::vector<Point*>&, int>())
        // .def("generate_grasp", &PointCloud::generateGraspsBrute)
        .def("generate_grasp_single_point", &PointCloud::generateGraspsSinglePoint)
        .def("__repr__", [](const PointCloud &pcd) {
            return std::string("<PointCloud with ") + std::to_string(pcd.size) + " points>";
        })
        .def_readwrite("cloud", &PointCloud::cloud)
        .def_readwrite("size",  &PointCloud::size)
        .def_readwrite("candidate_num", &PointCloud::candidateNum);    
}