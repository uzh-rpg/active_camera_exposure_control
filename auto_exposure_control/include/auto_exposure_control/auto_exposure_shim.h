#pragma once

#include <memory>

namespace cv
{
class Mat;
}

namespace auto_exposure
{
struct AutoExposureShimOptions
{
  bool shim_display = false;
  double shim_delta = 0.11;
  double shim_lambda = 10.0;
  double lowpass_desired_intensity_gain = 0.3;
};

// Shim, Inwook, Joon-Young Lee, and In So Kweon.
// "Auto-adjusting camera exposure for outdoor robotics using gradient
// information."
class AutoExposureShim
{
public:
  AutoExposureShim() = delete;
  AutoExposureShim(const AutoExposureShim&) = delete;
  void operator=(const AutoExposureShim&) = delete;

  AutoExposureShim(const AutoExposureShimOptions& options) : options_(options)
  {
  }
  ~AutoExposureShim() {}

  int computeDesiredIntensityExposureShim(const cv::Mat& img,
                                          const int last_exp_time_us);

  float computeGradientInformationShim(const cv::Mat& img);

  AutoExposureShimOptions options_;

private:
};
using AutoExposureShimPtr = std::shared_ptr<AutoExposureShim>;
}
