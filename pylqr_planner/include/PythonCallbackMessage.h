// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#pragma once

#include <ilqr_planner/utils/CallbackMessage.h>
#include <string>

class PythonCallbackMessage : public ilqr_planner::CallBackMessage {
public:
    PythonCallbackMessage();
    virtual void notify(const std::string& msg) override;

private:
};
