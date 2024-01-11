// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <string>

namespace ilqr_planner {
class CallBackMessage {
public:
    virtual ~CallBackMessage() {}
    virtual void notify(const std::string& message) = 0;
};
}  // namespace ilqr_planner
