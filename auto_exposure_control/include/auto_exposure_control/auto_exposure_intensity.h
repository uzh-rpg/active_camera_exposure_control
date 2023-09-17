#pragma once

#include <memory>

namespace cv
{
class Mat;
}

namespace auto_exposure
{
class AutoExposureIntensity
{
public:
  AutoExposureIntensity() {}
  ~AutoExposureIntensity() {}
  int computeDesiredExposureTimeIntensity(const cv::Mat& img,
                                          const int last_exp_us);
private:
};
using AutoExposureIntensityPtr = std::shared_ptr<AutoExposureIntensity>;
}
