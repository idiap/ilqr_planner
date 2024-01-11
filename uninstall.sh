# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

sudo xargs rm < build/ilqr_planner/install_manifest.txt
sudo xargs rm < build/pylqr_planner/install_manifest.txt
rm -rd build
