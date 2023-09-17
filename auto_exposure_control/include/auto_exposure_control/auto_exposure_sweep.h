#pragma once

#include <memory>

#include <photometric_camera/photometric_model.h>

#include "auto_exposure_control/auto_exp_utils.h"

namespace auto_exposure
{
struct AutoExposureSweepOptions
{
  // tuning: should mostly be fixed
  double update_damping = 0.2;
  double over_comp_thresh = 0.4;
  double under_comp_thresh = 0.4;

  // the following parameters should be fixed
  bool use_compensation_captured = true;
  bool use_compensation_predicted = true;

  int max_exp_us = 10000;
  int min_exp_us = 50;

  int num_samples = 30;
  int sample_max_range = 10000;

  // common options about weighted gradient calculation
  double weighted_gradient_order = 5.0;
  double weighted_gradient_pivotal = 0.7;
  double weighted_gradient_deriv_thresh = 10;
};

class AutoExposureSweep
{
public:
  AutoExposureSweep() = delete;
  AutoExposureSweep(const AutoExposureSweep&) = delete;
  void operator=(const AutoExposureSweep&) = delete;

  AutoExposureSweep(
      const AutoExposureSweepOptions& options,
      const auto_exposure_utils::OverUnderExposureCompOptions& comp_options,
      const photometric_camera::PhotometricModel::Ptr& photo_model)
    : options_(options)
    , comp_options_(comp_options)
    , photometric_model_(photo_model)
    , log_cnt_(0)
  {
    logger_.add("index");
    logger_.add("gain_x");
    logger_.add("exp_us");
    logger_.add("med_irad");
    logger_.add("mean_irad");
    logger_.add("wg");
    logger_.add("wg_deriv");
    logger_.add("exp_update");
  }

  ~AutoExposureSweep()
  {
//    std::ofstream log;
//    log.open("sweep_log.txt");
//    std::cout << logger_;
//    log << logger_;
//    log.close();
  }

  int computeDesiredExposureSweep(const cv::Mat& img,
                                  const int last_exp_us,
                                  const double gain_x);

  AutoExposureSweepOptions options_;

  // TO REMOVE LATER
  auto_exposure_utils::OverUnderExposureCompOptions comp_options_;

private:
  int clampExposureTime(int desired_exp)
  {
    // leave some space for the next sampling
    int new_desired_exp_us = desired_exp;
    if (new_desired_exp_us < options_.min_exp_us)
    {
      new_desired_exp_us = options_.min_exp_us + 50;
    }

    if (new_desired_exp_us > options_.max_exp_us)
    {
      new_desired_exp_us = options_.max_exp_us - 50;
    }

    return new_desired_exp_us;
  }

  photometric_camera::PhotometricModel::Ptr photometric_model_;
  std::vector<double> percentile_weights_;

  auto_exposure_utils::Logger logger_;
  int log_cnt_;
};
using AutoExposurSweepPtr = std::shared_ptr<AutoExposureSweep>;
}
