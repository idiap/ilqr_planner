// SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
//
// SPDX-FileContributor: Jeremy Maceiras  <jeremy.maceiras@idiap.ch>
//
// SPDX-License-Identifier: GPL-3.0-only

#include "PythonCallbackMessage.h"
#include <pybind11/embed.h>

namespace py = pybind11;

PythonCallbackMessage::PythonCallbackMessage() {}

void PythonCallbackMessage::notify(const std::string& msg) {
    // py::scoped_interpreter guard{};
    py::print(msg);
}
