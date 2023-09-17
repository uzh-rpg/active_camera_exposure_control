#pragma once

#include <memory>
#include <ros/ros.h>

namespace auto_exposure
{
class AutoCameraSettings;
}

namespace auto_exposure
{
std::shared_ptr<AutoCameraSettings> makeAutoCameraSettings(
    const ros::NodeHandle& pnh);
}
