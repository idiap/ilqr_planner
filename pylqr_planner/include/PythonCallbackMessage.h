#pragma once

#include <ilqr_planner/utils/CallbackMessage.h>
#include <string>

class PythonCallbackMessage : public ilqr_planner::CallBackMessage {
public:
    PythonCallbackMessage();
    virtual void notify(const std::string& msg) override;

private:
};
