# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

add_subdirectory(lib/tinyurdfparser)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)
#find_package(TinyURDFParser REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")
file(GLOB_RECURSE PROJECT_HEADERS "include/*.h")

#include_directories(include)

add_library(${PROJECT_NAME}
	STATIC
    ${PROJECT_SOURCES}
)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${${PROJECT_NAME}_VERSION}
  SOVERSION 1
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION 1
)

target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		${orocos_kdl_INCLUDE_DIRS}

	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/tinyurdfparser/include>
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
		${orocos_kdl_LIBRARIES}
	PRIVATE
		TinyURDFParser
)

install(TARGETS ${PROJECT_NAME} EXPORT ${CMAKE_PROJECT_NAME}-config-targets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

file(GLOB_RECURSE SIM_HEADERS "${CMAKE_CURRENT_LIST_DIR}/include/ilqr_planner/sim/*.h")
install(
	FILES
		${SIM_HEADERS}
	DESTINATION
		"include/${PROJECT_NAME}/sim"
	COMPONENT
		Devel
)

file(GLOB_RECURSE SOLVER_HEADERS "${CMAKE_CURRENT_LIST_DIR}/include/ilqr_planner/solver/*.h")
install(
	FILES
		${SOLVER_HEADERS}
	DESTINATION
		"include/${PROJECT_NAME}/solver"
	COMPONENT
		Devel
)


file(GLOB_RECURSE SYSTEM_HEADERS "${CMAKE_CURRENT_LIST_DIR}/include/ilqr_planner/system/*.h")
install(
	FILES
		${SYSTEM_HEADERS}
	DESTINATION
		"include/${PROJECT_NAME}/system"
	COMPONENT
		Devel
)

file(GLOB_RECURSE UTILS_HEADERS "${CMAKE_CURRENT_LIST_DIR}/include/ilqr_planner/utils/*.h")
install(
	FILES
		${UTILS_HEADERS}
	DESTINATION
		"include/${PROJECT_NAME}/utils"
	COMPONENT
		Devel
)
