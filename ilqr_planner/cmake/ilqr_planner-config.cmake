# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

include(CMakeFindDependencyMacro)

# Capturing values from configure (optional)
# set(my-config-var @my-config-var@)

# Same syntax as find_package
find_dependency(Eigen3 REQUIRED)
find_dependency(TinyURDFParser REQUIRED)
find_dependency(orocos_kdl REQUIRED)

# Any extra setup

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/ilqr_planner-config-targets.cmake")
