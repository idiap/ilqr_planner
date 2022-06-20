cmake_minimum_required(VERSION 3.5)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(ilqr_planner)

set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)
set(${PROJECT_NAME}_VERSION_PATCH 0)

set(${PROJECT_NAME}_VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(${PROJECT_NAME}_PREFIX_PACKAGE share/${PROJECT_NAME}/cmake)
set(${PROJECT_NAME}_PREFIX_INCLUDE include/${PROJECT_NAME})

set(DEBUG_MODE true)

add_compile_options(
    "-Wall"
    "-Wextra"
    "-Werror=vla"
    "-Wno-unused-function"
    "-Wno-missing-braces"
    "-Wno-unknown-pragmas"
    "-Wno-parentheses"
    "-pedantic"
    "-Wconversion"
    "-Werror=pedantic"
    "-O3"
    "-fPIC"
)

add_subdirectory(${PROJECT_NAME})

export(EXPORT ${PROJECT_NAME}-config-targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-targets.cmake"
  NAMESPACE ${PROJECT_NAME}::
)

install(EXPORT ${PROJECT_NAME}-config-targets
  FILE ${PROJECT_NAME}-config-targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${${PROJECT_NAME}_PREFIX_PACKAGE}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/${PROJECT_NAME}-config.cmake "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake" COPYONLY)
configure_file(cmake/${PROJECT_NAME}-packages.cmake "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-packages.cmake")

install(
  FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake" 
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-packages.cmake"
  DESTINATION 
    ${${PROJECT_NAME}_PREFIX_PACKAGE}
  COMPONENT 
    Devel
)

