#pragma once

#include <memory>
#include <vector>

#include <photometric_camera/photometric_model.h>

#include "auto_exposure_control/auto_exp_utils.h"
#include "auto_exposure_control/fixed_size_buffer.h"

namespace cv
{
class Mat;
}

namespace auto_exposure
{
struct AutoExposurePercentileOptions
{
  // options for weighted gradient method
  // sine weights order
  double weighted_gradient_order = 5.0;

  // the percentile we use
  double weighted_gradient_pivotal = 0.7;

  // basic gradient settings
  int min_update = 1;
  double ga_rate = 0.7;

  // automatic gradeint ascent rate tuning
  bool gain_scaling = true;
  bool auto_ga_rate = true;
  double auto_ga_rate_multiplier = 1.0;
  std::string ga_rate_profile;

  // switch to intensity based method for acceleration
  int use_intensity_low_bound = 70;
  int use_intensity_high_bound = 190;
  double rate_over_comp_thresh = 0.5;

  // accelerate long monotonic update
  bool use_slope_comp = true;
  double slope_inc_rate = 1.2;
  double slope_inc_thresh = 0.2;

  // EXPERIMENTAL: one shot optimization
  bool oneshot = false;
  int oneshot_maxiter = 2;
};

// Calculate the exposure time to maximize the percentile gradient
class AutoExposurePercentile
{
public:
  AutoExposurePercentile() = delete;
  AutoExposurePercentile(const AutoExposurePercentile&) = delete;
  void operator=(const AutoExposurePercentile&) = delete;

  AutoExposurePercentile(
      const AutoExposurePercentileOptions& options,
      const auto_exposure_utils::OverUnderExposureCompOptions& comp_options,
      const photometric_camera::PhotometricModel::Ptr& photo_model)
    : options_(options)
    , comp_options_(comp_options)
    , photometric_model_(photo_model)
    , weighted_grad_derivs_(15u)
#ifdef DEBUG_OUTPUT
    , past_states_(15u)
#endif
    , percentile_weights_()
    , slope_scale_(1.0)
  {
    init();
  }

  ~AutoExposurePercentile() {}

  int computeDesiredExposureWeightedGradient(const cv::Mat& img,
                                             const int last_exp_us,
                                             const float gain_x = 1.0);
  int computeDesiredExposureWeightedGradientInc(const cv::Mat& img,
                                                const int last_exp_us,
                                                const float gain_x = 1.0);
  int computeDesiredExposureWeightedGradientOneShot(const cv::Mat& img,
                                                    const int last_exp_us,
                                                    const float gain_x = 1.0);
  double computeDWeightedGradientDExposure(const cv::Mat& img,
                                           const int exp_us,
                                           const float gain_x = 1.0);

  AutoExposurePercentileOptions options_;
  auto_exposure_utils::OverUnderExposureCompOptions comp_options_;

private:

  void init();
  double computeDesiredGARate(const double median_irad,
                              const bool gain_scaling,
                              const float gain_x,
                              const double mutiplier);

  std::vector<double> med_irad_;
  std::vector<double> ga_rate_;

  double kl_;
  double kh_;
  std::vector<double> km_;
  std::vector<double> bm_;


  struct InternalStates
  {
    InternalStates(const size_t buffer_size)
      : raw_ga_rates(buffer_size)
      , comp_ga_rates(buffer_size)
      , exp_updates(buffer_size)
      , weighted_grad(buffer_size)
      , weighted_grad_derivs(buffer_size)
      , over_ratios(buffer_size)
      , under_ratios(buffer_size)
      , median_grad(buffer_size)
      , median_irad(buffer_size)
      , mean_intensities(buffer_size)
      , med_intensities(buffer_size)
      , hist_size(buffer_size)
    {  }

    void check() const
    {
      CHECK_EQ(raw_ga_rates.cnt(), comp_ga_rates.cnt());
      CHECK_EQ(comp_ga_rates.cnt(), exp_updates.cnt());
      CHECK_EQ(exp_updates.cnt(), weighted_grad.cnt());
      CHECK_EQ(weighted_grad.cnt(), weighted_grad_derivs.cnt());
      CHECK_EQ(weighted_grad_derivs.cnt(), over_ratios.cnt());
      CHECK_EQ(over_ratios.cnt(), under_ratios.cnt());
      CHECK_EQ(under_ratios.cnt(), median_grad.cnt());
      CHECK_EQ(median_grad.cnt(), median_irad.cnt());
    }

    bool isFilled() const
    {
      return comp_ga_rates.isFilled();
    }

    size_t size() const
    {
      return raw_ga_rates.size();
    }

    ae_utils::FixedSizeBuffer<double> raw_ga_rates;
    ae_utils::FixedSizeBuffer<double> comp_ga_rates;
    ae_utils::FixedSizeBuffer<double> exp_updates;
    ae_utils::FixedSizeBuffer<double> weighted_grad;
    ae_utils::FixedSizeBuffer<double> weighted_grad_derivs;

    ae_utils::FixedSizeBuffer<double> over_ratios;
    ae_utils::FixedSizeBuffer<double> under_ratios;
    ae_utils::FixedSizeBuffer<double> median_grad;
    ae_utils::FixedSizeBuffer<double> median_irad;

    ae_utils::FixedSizeBuffer<double> mean_intensities;
    ae_utils::FixedSizeBuffer<double> med_intensities;

    size_t hist_size;
  };


  photometric_camera::PhotometricModel::Ptr photometric_model_;
  ae_utils::FixedSizeBuffer<double> weighted_grad_derivs_;

#ifdef DEBUG_OUTPUT
  InternalStates past_states_;
#endif

  std::vector<double> percentile_weights_;
  double slope_scale_ = 1.0;
};
using AutoExposurePercentilePtr = std::shared_ptr<AutoExposurePercentile>;
}
