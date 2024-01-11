<!--
SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>

SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>

SPDX-License-Identifier: GPL-3.0-only
-->

# CHANGELOG

## V1.1.1 TO 2022-07-06

* Added the possibility to create joint space tasks.
  * ``AngularKeypoint`` or ``AngularTimeKeypoint`` can be used for keypoints.
  * ``JointSpacePlannerSys`` or ``JointSpaceTimePlannerSys`` can be used as system.
* Updated ``SequentialSystem`` to accept systems that does not share the same target space (as long as the state and control space are the same).

## V1.1.0 2022-06-02

* Deal properly state limits, now this feature is used in all the solvers (batch and iterative).
* For systems, added a new constructor that allows to set qMax and qMin only.
* Removed dependecies to ``kdl_parser`` and ``urdf``. ROS is not needed anymore.

### BREAKING

* ``fpBatch`` , and ``forwardPassWithLimits`` does not return command constraints anymore (command constraints have been removed).
* ``SequentialSystem`` does not take the joint limits as constructor parameters anymore.
* Modified the way keypoints are handled by ``System`` object.
  * Instead of passing raw lists, now a list of ``Keypoint`` objects is required.
  * For this, implemented classes: ``Keypoint`` (abstract), ``PosOrnKeypoint`` for ``PosOrnPlannerSys`` , ``SpacetimeKeypoint`` for ``PosOrnTimePlannerSys``
* Modified constructors of batch solvers:
  * Can not pass a ``mu`` vector anymore, you can still pass a custom ``Q`` matrix that the solver will use with ``System::keypoints``
  * For the same reason, removed ``vpIndexes`` argument, solvers will use the ones provided by ``System``
* ``solve`` method for solvers now take an additional argument: ``early_stop``.

## V1.0.3 2022-03-09

* Added a normal batch iLQR solver
* Modified sparse computation of Batch solvers
  * Instead of using Eigen's sparse computation do it manually by removing unneeded lines in Su,Q, and J.

### Breaking

* Constructor of batch solver changed (u0 is not passed in the constructor anymore).
* ``solve`` method of batch solver changed (take u0 as parameter).
* For conveniance, class ``BatchSparseItLQRCP`` has been renamed to ``BatchILQRCP``.

## V1.0.2 2021-12-20

* Fixed a bug in ``ILQRRecursive`` with the computation of gains.
* For ``BatchSparseItLQRCP`` and ``BatchSparseILQR`` added a constructor that does not take Q and mu as parameter (use the one provided by the system)
* For the solvers, reset the system at the end of the optimization.
* Cleaned build scenario:
  * Bring back catkin possibility
  * Replaced global cmake by a ``build.sh`` file and ``uninstall.sh`` file.

## V1.0.1 2021-11-29

* Added a batch ilqr (without control primitives)
* For Spacetime system: Time is now dealed inside the Simulation interface object that provide a ``getTime`` and ``setTime`` methods. (allow compatibility for sequential systems)
* ``SimulationInterface::setConfiguration(...)`` now takes an extra parameters ``reset-time``

### Python

* Added a multiple systems example for second order systems.
* Added a multiple systems example with time optimization.

### C++

* ``System::fpBatch(...)`` and ``System::diffBatch(...)`` are now using const references instead of raw pointer.
* Implemented a generic ``System::diffBatch(...)`` that can be used by all systems.

## V1.0.0 2021-11-25

* Restructuration of the project. Official initial release.

### BREAKING

* Source moved from [here](https://gitlab.idiap.ch/rli/ros-ilqr) to [here](https://gitlab.idiap.ch/rli/ilqr_planner)
* As the project evolved a lot since the last realease. We should consider this package as a new one.
