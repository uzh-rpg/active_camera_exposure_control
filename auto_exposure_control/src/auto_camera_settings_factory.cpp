#include "auto_exposure_control/auto_camera_settings_factory.h"

#include <ros/package.h>
#include <glog/logging.h>
#include <photometric_camera/photometric_model.h>

#include "auto_exposure_control/auto_camera_settings.h"
#include "vikit/params_helper.h"

namespace auto_exposure
{
void loadAutoCameraSettingsOptions(const ros::NodeHandle& pnh,
                                   AutoCameraOptions* options)
{
  CHECK_NOTNULL(options);

  std::string ae_method = vk::param<std::string>(pnh, "ae_method", "Intensity");
  if (ae_method == "Fixed")
  {
    options->ae_method = AutoExposureMethod::kFixed;
  }
  else if (ae_method == "Percentile")
  {
    options->ae_method = AutoExposureMethod::kPercentile;
  }
  else if (ae_method == "Shim")
  {
    options->ae_method = AutoExposureMethod::kShim;
  }
  else if (ae_method == "Sweep")
  {
    options->ae_method = AutoExposureMethod::kSweep;
  }
  else if (ae_method == "Intensity")
  {
    options->ae_method = AutoExposureMethod::kMeanIntensity;
  }
  else if (ae_method == "PatchScore")
  {
    options->ae_method = AutoExposureMethod::kPatchScore;
  }
  else if (ae_method == "BlurPatchScore")
  {
    options->ae_method = AutoExposureMethod::kBlurPatchScore;
  }
  else
  {
    LOG(FATAL) << "Unknown AE method.";
  }

  // basic settings
  options->use_parallel_thread = vk::param<bool>(pnh, "use_parallel_thread", false);
  options->pyramid_lvl = vk::param<int>(pnh, "pyramid_level", 3);

  // exposure time range
  options->max_exp_time_us = vk::param<int>(pnh, "max_exp_time_us", 15000);
  options->min_exp_time_us = vk::param<int>(pnh, "min_exp_time_us", 20);

  // gain changing
  options->auto_gain = vk::param<bool>(pnh, "auto_gain", true);
  options->inc_gain_exp_us = vk::param<int>(pnh, "inc_gain_max_exp_us", 10000);
  options->dec_gain_exp_us = vk::param<int>(pnh, "dec_gain_min_exp_us", 2000);
  options->gain_change_step =
      static_cast<float>(vk::param<double>(pnh, "gain_change_step", 0.5));
  options->max_gain =
      static_cast<float>(vk::param<double>(pnh, "max_gain", 4.0));
  options->min_gain =
      static_cast<float>(vk::param<double>(pnh, "min_gain", 1.0));

  // Method specific settings
  // options for Shim
  options->shim_options_.shim_display =
      vk::param<bool>(pnh, "shim_display", false);
  options->shim_options_.shim_lambda =
      vk::param<double>(pnh, "shim_lambda", 10.0);
  options->shim_options_.shim_delta =
      vk::param<double>(pnh, "shim_delta", 0.11);
  options->shim_options_.lowpass_desired_intensity_gain =
      vk::param<double>(pnh, "shim_lowpass_gain", 0.3);

  // weighted gradient method
  options->perc_options.weighted_gradient_order =
      vk::param<double>(pnh, "weighted_gradient_order", 5.0);
  options->perc_options.weighted_gradient_pivotal =
      vk::param<double>(pnh, "weighted_gradient_pivotal", 0.7);
  // gradient ascent settings
  options->perc_options.ga_rate =
      vk::param<double>(pnh, "percentile_ga_rate", 0.7);
  options->perc_options.min_update =
      vk::param<int>(pnh, "percentile_min_update", 1);
  // automatic ga rate
  options->perc_options.auto_ga_rate =
      vk::param<bool>(pnh, "percentile_auto_ga_rate", true);
  options->perc_options.auto_ga_rate_multiplier =
      vk::param<double>(pnh, "percentile_auto_ga_rate_multiplier", 1.0);
  options->perc_options.gain_scaling =
      vk::param<bool>(pnh, "percentile_gain_scaling", true);
  options->perc_options.ga_rate_profile =
      vk::param<std::string>(pnh, "percentile_ag_profile",
                             ros::package::getPath("auto_exposure_control") +
                             "/params/ga_profile.txt");
  // combine intensity based method
  options->perc_options.use_intensity_high_bound =
      vk::param<int>(pnh, "percentile_intensity_high_bound", 190);
  options->perc_options.use_intensity_low_bound =
      vk::param<int>(pnh, "percentile_intensity_low_bound", 70);
  options->perc_options.rate_over_comp_thresh =
      vk::param<double>(pnh, "percentile_rate_over_comp_thresh", 0.5);
  // accelerate monotonic slope
  options->perc_options.use_slope_comp =
      vk::param<bool>(pnh, "percentile_use_slope_comp", true);
  options->perc_options.slope_inc_thresh =
      vk::param<double>(pnh, "percentile_slope_inc_thresh", 0.2);
  options->perc_options.slope_inc_rate =
      vk::param<double>(pnh, "percentile_slope_inc_rate", 1.2);
  // oneshot speed up
  options->perc_options.oneshot =
      vk::param<bool>(pnh, "percentile_oneshot", false);
  options->perc_options.oneshot_maxiter =
      vk::param<int>(pnh, "percentile_oneshot_maxiter", 2);

  // sweep method
  options->sweep_options_.update_damping =
      vk::param<double>(pnh, "sweep_update_damping", 0.2);
  options->sweep_options_.use_compensation_captured =
      vk::param<bool>(pnh, "sweep_use_compensation_captured", true);
  options->sweep_options_.use_compensation_predicted =
      vk::param<bool>(pnh, "sweep_use_compensation_predicted", true);
  options->sweep_options_.max_exp_us =
      vk::param<int>(pnh, "max_exp_time_us", 15000);
  options->sweep_options_.min_exp_us =
      vk::param<int>(pnh, "min_exp_time_us", 20);
  options->sweep_options_.over_comp_thresh =
      vk::param<double>(pnh, "sweep_over_comp_thresh", 0.4);
  options->sweep_options_.under_comp_thresh =
      vk::param<double>(pnh, "sweep_under_comp_thresh", 0.4);

  options->sweep_options_.weighted_gradient_order =
      vk::param<double>(pnh, "weighted_gradient_order", 5.0);
  options->sweep_options_.weighted_gradient_pivotal =
      vk::param<double>(pnh, "weighted_gradient_pivotal", 0.5);
  options->sweep_options_.weighted_gradient_deriv_thresh =
      vk::param<double>(pnh, "weighted_gradient_deriv_thresh", 10.0);

  // over and under exposure compensation
  options->comp_options_.overexposure_active_thresh =
      vk::param<double>(pnh, "over_compensation_active_thresh", 0.0);
  options->comp_options_.overexposure_compensation_factor =
      vk::param<double>(pnh, "over_compensation_factor", -3.0);
  options->comp_options_.overexposure_compensation_g_thresh =
      vk::param<double>(pnh, "over_compensation_g_thresh", 0.5);
  options->comp_options_.underexposure_active_thresh =
      vk::param<double>(pnh, "under_compensation_active_thresh", 1.0);
  options->comp_options_.underexposure_compensation_factor =
      vk::param<double>(pnh, "under_compensation_factor", 3.0);
  options->comp_options_.underexposure_compensation_g_thresh =
      vk::param<double>(pnh, "under_compensation_g_thresh", 0.8);
  options->comp_options_.over_intensity_thresh = static_cast<uint8_t>(
      vk::param<double>(pnh, "over_intensity_thresh", 252.0));
  options->comp_options_.under_intensity_thresh = static_cast<uint8_t>(
      vk::param<double>(pnh, "under_intensity_thresh", 10.0));


// patch/feature based method, need to use tracking information
#ifdef USING_TRACKING_INFO
  options->blur_use_single_frame =
      vk::param<bool>(pnh, "blur_use_single_frame", false);

  options->radiometric_gradient_thresh =
      vk::param<double>(pnh, "gradient_thresh", 700.0);
  options->use_mean_gradient = vk::param<bool>(pnh, "use_mean_gradient", false);
  options->use_weighted_mean = vk::param<bool>(pnh, "use_weighted_mean", false);
  options->pivot_gradient_ratio =
      vk::param<double>(pnh, "pivot_gradient_ratio", 0.5);
  options->use_adaptive_ratio =
      vk::param<bool>(pnh, "use_adaptive_ratio", true);

  options->use_gradient_ascent =
      vk::param<bool>(pnh, "use_gradient_ascent", false);
  options->exposure_change_step =
      vk::param<double>(pnh, "exposure_change_step", 0.05f);
  options->gradient_ascent_rate =
      vk::param<double>(pnh, "gradient_ascent_rate", 0.1f);
#endif
}

std::shared_ptr<AutoCameraSettings>
makeAutoCameraSettings(const ros::NodeHandle& pnh)
{
  AutoCameraOptions options;
  loadAutoCameraSettingsOptions(pnh, &options);
  std::string radiometric_calib_dir = vk::param<std::string>(
      pnh,
      "photometric_dir",
      ros::package::getPath("photometric_calib") + "/camera_25000742");
  photometric_camera::PhotometricModel::Ptr photo_model =
      photometric_camera::PhotometricModel::loadModel(radiometric_calib_dir,
                                                      1.0);
  CHECK(photo_model);

  return std::make_shared<AutoCameraSettings>(options, photo_model);
}
}
