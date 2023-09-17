#include "auto_exposure_control/auto_exposure_intensity.h"

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace auto_exposure
{
int AutoExposureIntensity::computeDesiredExposureTimeIntensity(
    const cv::Mat& img, const int last_exp_us)
{
  int mean_intensity = static_cast<int>(*(cv::mean(img).val));
  if (mean_intensity < 1)
  {
    mean_intensity = 1;
  }

  VLOG(45) << "Mean intensity: " << mean_intensity;

  float damping = 0.2f; // this helps avoid oscillation

  int exp_delta =
      static_cast<int>(last_exp_us * ((255.0 * 0.5) / mean_intensity - 1.0));

  return static_cast<int>(last_exp_us + damping * exp_delta);
}
}
