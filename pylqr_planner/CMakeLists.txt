
cmake_minimum_required(VERSION 3.5)
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(pylqr_planner)

find_package(TinyURDFParser REQUIRED)
find_package(ilqr_planner REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)

if(DEFINED ENV{ROS_PYTHON_VERSION})
  SET(PYTHON_VERSION $ENV{ROS_PYTHON_VERSION} CACHE STRING "Python Version")
else()
  SET(PYTHON_VERSION 3 CACHE STRING "Python Version")
endif()

set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION} CACHE STRING "Python version used by PyBind11")

find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)

set(LIBRARY_NAME "PyLQR")

set(PYTHON_SITE_PACKAGES_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages")  # This might be overridden below if built with catkin or if conda is detected

if(DEFINED ENV{CONDA_PREFIX})
   message(STATUS "CONDA INSTALLATION DETECTED")
   set(PYTHON_SITE_PACKAGES_INSTALL_DIR "$ENV{CONDA_PREFIX}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")  # This might be overridden below if built with catkin.
endif()

if(${USE_CATKIN})
  message(STATUS "WILL USE CATKIN")
  find_package(catkin QUIET)
  catkin_package(
    SKIP_CMAKE_CONFIG_GENERATION
    SKIP_PKG_CONFIG_GENERATION
  )
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR})
  set(PYTHON_SITE_PACKAGES_INSTALL_DIR "${PYTHON_INSTALL_DIR}")
endif()

# Build the module
include_directories(include)

SET(PYTHON_MODULE_EXTENSION ".so")
add_subdirectory(pybind11)
pybind11_add_module(${LIBRARY_NAME}
  src/bindings.cpp
  src/PythonCallbackMessage.cpp)

message(STATUS "Python package will be installed in: ${PYTHON_SITE_PACKAGES_INSTALL_DIR}")
target_link_libraries(${LIBRARY_NAME} PRIVATE ${Eigen3_LIBRARIES} ilqr_planner::ilqr_planner ${orocos_kdl_LIBRARIES} TinyURDFParser::TinyURDFParser)
install(TARGETS ${LIBRARY_NAME} DESTINATION "${PYTHON_SITE_PACKAGES_INSTALL_DIR}")
