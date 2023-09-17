#include "auto_exposure_control/auto_camera_settings.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>
#include <vikit/timer.h>
#include <vikit/params_helper.h>

using namespace auto_exposure_utils;

// utility functions
namespace
{
#ifdef USING_TRACKING_INFO
// this function first normalizes the centripetal to [0, 1],
// then computes the sum-normalized weight
void computeWeightsFromCentripetal(const std::vector<float>& centripetal,
                                   std::vector<float>* weights)
{
  CHECK_NOTNULL(weights);
  CHECK(!centripetal.empty());
  CHECK_EQ(weights->size(), centripetal.size());

  float after_min = 0.0;
  float after_max = 1.0;
  float min_v = *(std::min_element(centripetal.begin(), centripetal.end()));
  float max_v = *(std::max_element(centripetal.begin(), centripetal.end()));

  if ((max_v - min_v) < 450)
  {
    VLOG(20) << "The range of centripetals is too small. No weight used.";
    return;
  }

  float ratio = (after_max - after_min) / (max_v - min_v);
  for (size_t i = 0; i < centripetal.size(); i++)
  {
    (*weights)[i] = (centripetal[i] - min_v) * ratio + after_min;
  }

  float normalized_sum = std::accumulate(weights->begin(), weights->end(), 0.0);
  std::for_each(weights->begin(),
                weights->end(),
                [normalized_sum](float& w)
                {
                  w /= normalized_sum;
                });
}

// this function calculates the new exposure time based on a flag value,
// and the flag value can be median, mean or based on an adaptive ratio
float calNewExposureTimeFlagValue(
    std::vector<float>& centripetal,
    std::vector<float>& grad_vec,     // will be sorted because of nth_element
    std::vector<float>& average_vec,  // will be sorted
    const float last_exposure_time_us,
    const auto_exposure::AutoCameraOptions options)
{
  CHECK_EQ(centripetal.size(), grad_vec.size());
  if (grad_vec.empty())
  {
    LOG(WARNING) << "No gradient information, exposure time unchanged!";
    return last_exposure_time_us;
  }

  // calculate the flag gradient
  float flag_gradient = 0.0;
  if (options.use_mean_gradient)
  {  // use the average of the gradients
    std::vector<float> weights(grad_vec.size(), 1.0 / grad_vec.size());
    if (options.use_weighted_mean)
    {
      computeWeightsFromCentripetal(centripetal, &weights);
    }
    for (size_t i = 0; i < grad_vec.size(); i++)
    {
      flag_gradient += grad_vec[i] * weights[i];
    }
    VLOG(45) << "Using mean Gradient: " << flag_gradient;
  }
  else
  {  // use a pivot gradient
    float ratio = options.pivot_gradient_ratio;
    if (options.use_adaptive_ratio && !average_vec.empty())
    {  // the ratio depends on the distribution of average intensities
      std::sort(average_vec.begin(), average_vec.end());
      int lower_than_middle =
          std::lower_bound(average_vec.begin(), average_vec.end(), 127.0) -
          average_vec.begin();
      ratio = (lower_than_middle * 1.0) / (average_vec.size());
    }
    VLOG(45) << "Using pivot ratio: " << ratio;

    auto it = grad_vec.begin() + std::floor(grad_vec.size() * ratio);
    std::nth_element(grad_vec.begin(), it, grad_vec.end());
    flag_gradient = *it;
    VLOG(45) << "Pivot Gradient: " << flag_gradient;
  }

  // change the exposure based on the gradient
  float next_exposure_time_us = last_exposure_time_us;
  if (std::fabs(flag_gradient) < options.radiometric_gradient_thresh)
  {
    VLOG(20) << "Flag gradient is too small, will not change the exposure.";
    return next_exposure_time_us;
  }
  if (options.use_gradient_ascent)
  {
    VLOG(45) << "Gradient Ascent rate: " << options.gradient_ascent_rate;
    next_exposure_time_us += options.gradient_ascent_rate * flag_gradient;
  }
  else
  {  // use a change rate w.r.t. current exposure time
    VLOG(45) << "Exposure Change Ratio" << options.exposure_change_step;
    if (flag_gradient > 0)
    {
      next_exposure_time_us *= (1.0 + options.exposure_change_step);
    }
    else
    {
      next_exposure_time_us *= (1.0 - options.exposure_change_step);
    }
  }
  VLOG(45) << "last exposure time = " << last_exposure_time_us;
  VLOG(45) << "next exposure time = " << next_exposure_time_us;

  return next_exposure_time_us;
}

bool calGradientAverageVectorFromImages(
    const svo::FramePtr& cur_frame,
    const int cur_level,
    const cv::Mat& cur_img_pyr,
    const svo::Keypoint& cur_px_pyr,
    const svo::FramePtr& ref_frame,
    const int ref_level,
    const cv::Mat& ref_img_pyr,
    const svo::PointPtr& landmark,
    const photometric_camera::PhotometricModel& rm,
    std::vector<float>* grad_vec,
    std::vector<float>* average_vec)
{
  CHECK(landmark);
  CHECK_NOTNULL(grad_vec);

  // create a patch out of the current feature
  active_camera_control::FeaturePatch::Ptr patch =
      active_camera_control::FeaturePatch::makePatch(cur_img_pyr,
                                                     cur_px_pyr.cast<int>());
  if (!patch)
  {
    LOG(WARNING) << "Fail to create patch from the current image!";
    return false;
  }

  // calculate dexposure_dt (actually irradiance)
  cv::Mat dexposure_dt;
  dexposure_dt.create(
      2 * patch->halfpatch_size_ + 2, 2 * patch->halfpatch_size_ + 2, CV_64F);
  Eigen::Vector3d c_p = cur_frame->T_f_w_ * landmark->pos();
  double depth_z = c_p(2);
  Eigen::Matrix<double, 6, 1> se3;
  se3.block<3, 1>(0, 0) = cur_frame->v_ms_ * (cur_frame->exp_time_us_ / 1000.0);
  se3.block<3, 1>(3, 0) = cur_frame->w_ms_ * (cur_frame->exp_time_us_ / 1000.0);
  svo::Transformation Tct = svo::Transformation::exp(se3);
  for (int ix = -patch->halfpatch_size_ - 1; ix < patch->halfpatch_size_ + 1;
       ix++)
  {
    for (int iy = -patch->halfpatch_size_ - 1; iy < patch->halfpatch_size_ + 1;
         iy++)
    {
      // 3D point in the frame (t) at the end of the exposure period
      svo::Keypoint kp_pyr(cur_px_pyr(0) + ix, cur_px_pyr(1) + iy);
      svo::Keypoint kp_img = kp_pyr * (1 << cur_level);
      Eigen::Vector3d t_pi;
      cur_frame->cam()->backProject3(kp_img, &t_pi);
      t_pi = t_pi / t_pi(2) * depth_z;
      // query the intensity from the reference frame (r)
      Eigen::Vector3d r_pi =
          ref_frame->T_cam_world() * cur_frame->T_world_cam() * Tct * t_pi;
      svo::Keypoint r_ui;
      ref_frame->cam()->project3(r_pi, &r_ui);
      svo::Keypoint r_ui_pyr = r_ui / (1 << ref_level);
      uint8_t ref_intensity = ref_img_pyr.at<uint8_t>(r_ui_pyr(1), r_ui_pyr(0));
      // from intensity to irradiance
      dexposure_dt.at<double>(iy + patch->halfpatch_size_ + 1,
                              ix + patch->halfpatch_size_ + 1) =
          std::exp(rm.g(ref_intensity)) / (ref_frame->exp_time_us_ / 1000.0);
    }
  }

  // calculate dScoredExposure
  cv::Mat exposure_patch;
  rm.getExposurePatch(cur_img_pyr,
                      cur_px_pyr(0),
                      cur_px_pyr(1),
                      patch->halfpatch_size_ + 1,
                      &exposure_patch);
  float dscore_dt = patch->computeDScoreDExposureBlur(
      rm,
      exposure_patch,
      dexposure_dt,
      Eigen::Vector2i(patch->halfpatch_size_ + 1, patch->halfpatch_size_ + 1));

  // save
  grad_vec->push_back(dscore_dt);
  if (average_vec)
  {
    average_vec->push_back(patch->computeAverage());
  }

  return true;
}
#endif

}  // namespace

namespace auto_exposure
{
AutoCameraSettings::AutoCameraSettings(
    const AutoCameraOptions& options,
    const photometric_camera::PhotometricModel::Ptr& photometric_model)
  : options_(options)
  , photometric_model_(photometric_model)
  , thread_(nullptr)
  , quit_thread_(false)
  , next_frame_condvar_()
  , next_frame_mutex_()
  , next_frame_(nullptr)
  , last_exp_time_us_(-1)
  , last_gain_x_(-1)
  , desired_values_mutex_()
  , desired_exposure_time_us_(-1)
  , auto_exp_perc_(nullptr)
  , auto_exp_shim_(nullptr)
  , auto_exp_sweep_(nullptr)
  , auto_exp_intensity_(nullptr)
{
  if (options_.ae_method == AutoExposureMethod::kPercentile)
  {
    auto_exp_perc_.reset(new auto_exposure::AutoExposurePercentile(
        options_.perc_options, options_.comp_options_, photometric_model_));
  }
  else if (options_.ae_method == AutoExposureMethod::kShim)
  {
    auto_exp_shim_.reset(
        new auto_exposure::AutoExposureShim(options_.shim_options_));
  }
  else if (options.ae_method == AutoExposureMethod::kSweep)
  {
    auto_exp_sweep_.reset(new auto_exposure::AutoExposureSweep(
        options_.sweep_options_, options.comp_options_, photometric_model_));
  }
  else if (options.ae_method == AutoExposureMethod::kMeanIntensity)
  {
    auto_exp_intensity_.reset(new auto_exposure::AutoExposureIntensity());
  }
  if (options_.use_parallel_thread)
    startThread();
}

void AutoCameraSettings::startThread()
{
  CHECK(options_.use_parallel_thread);
  if (thread_)
  {
    std::cout << "ERROR IntensityScheduler thread already started!"
              << std::endl;
    return;
  }
  thread_.reset(new std::thread(&AutoCameraSettings::threadLoop, this));
  std::cout << "IntensityScheduler thread started!" << std::endl;
}

void AutoCameraSettings::stopThread()
{
  CHECK(options_.use_parallel_thread);
  std::cout << "IntensityScheduler stop thread invoked." << std::endl;
  if (thread_ != nullptr)
  {
    std::cout << "IntensityScheduler stop thread invoked." << std::endl;
    quit_thread_ = true;
    thread_->join();
    thread_.reset();
  }
}

void AutoCameraSettings::setNextFrame(const auto_exposure::FramePtr& frame,
                                      const int exposure_time_us,
                                      const float gain_x)
{
  if (options_.use_parallel_thread)
  {
    std::lock_guard<std::mutex> lock(next_frame_mutex_);
    next_frame_ = frame;
    last_exp_time_us_ = exposure_time_us;
    last_gain_x_ = gain_x;
    next_frame_condvar_.notify_all();
  }
  else  // make sure the new settings are available after the function returns
  {
    last_exp_time_us_ = exposure_time_us;
    last_gain_x_ = gain_x;
    calculateNewSettings(frame);
  }
}

void AutoCameraSettings::setNextFrame(const cv::Mat& img,
                                      const int exposure_time_us,
                                      const float gain_x)
{
  auto_exposure::FramePtr new_frame_ptr =
      std::make_shared<auto_exposure::Frame>(
          exposure_time_us, gain_x, img, options_.pyramid_lvl + 1);
  setNextFrame(
      new_frame_ptr, new_frame_ptr->exp_time_us, new_frame_ptr->gain_x);
}

void AutoCameraSettings::threadLoop()
{
  while (!quit_thread_)
  {
    // get current frame
    {
      std::unique_lock<std::mutex> lock(next_frame_mutex_);
      while (next_frame_ == nullptr && !quit_thread_)
      {
        next_frame_condvar_.wait_for(lock, std::chrono::milliseconds(500));
      }
    }
    auto_exposure::FramePtr frame = nullptr;
    {
      std::lock_guard<std::mutex> lock(next_frame_mutex_);
      frame = next_frame_;
    }  // release lock

    // check frame availability before proceeding
    if (!frame)
    {
      desired_exposure_time_us_ = last_exp_time_us_;
      desired_gain_x_ = last_gain_x_;
      LOG(WARNING) << "No frame available!";
      return;
    }

    calculateNewSettings(frame);

    {
      std::lock_guard<std::mutex> lock(next_frame_mutex_);
      if (frame == next_frame_)
      {
        next_frame_ = nullptr;
      }
    }
  }  // thread loop
}

void AutoCameraSettings::calculateNewSettings(
    const auto_exposure::FramePtr& frame)
{
  VLOG(45) << "========== New Frame " << frame->id << " ==========";
  // working on subsampled image
  CHECK_GT(frame->img_pyr.size(), options_.pyramid_lvl);
  cv::Mat img = frame->img_pyr.at(options_.pyramid_lvl);

  // actual exposure control according to options
  desired_gain_x_ = last_gain_x_;
  vk::Timer t;
  int new_desired_exp_us = last_exp_time_us_;
  if (options_.ae_method == AutoExposureMethod::kFixed)
  {
  }
  else if (options_.ae_method == AutoExposureMethod::kPercentile)
  {
    new_desired_exp_us = auto_exp_perc_->computeDesiredExposureWeightedGradient(
        img, last_exp_time_us_, last_gain_x_);
  }
  else if (options_.ae_method == AutoExposureMethod::kSweep)
  {
    VLOG(45) << "Method: Sweep";

    new_desired_exp_us = auto_exp_sweep_->computeDesiredExposureSweep(
        img, last_exp_time_us_, last_gain_x_);
  }
  else if (options_.ae_method == AutoExposureMethod::kMeanIntensity)
  {
    VLOG(45) << "Average Intensity.";
    new_desired_exp_us =
        auto_exp_intensity_->computeDesiredExposureTimeIntensity(
            img, last_exp_time_us_);
  }
  else if (options_.ae_method == AutoExposureMethod::kShim)
  {
    VLOG(45) << "Method: Shim";
    new_desired_exp_us = auto_exp_shim_->computeDesiredIntensityExposureShim(
        img, last_exp_time_us_);
  }
  else if (options_.ae_method == AutoExposureMethod::kPatchScore)
  {  // change the exposure only based on the patch scores
#ifdef USE_TRACKING_INFO
    VLOG(45) << "Method: patch score";
    vk::Timer t;
    const int new_desired_exposure_us =
        computeDesiredExposureTime(frame, last_exposure_time_us_);
    VLOG(45) << "Computation Time: " << t.stop() * 1000 << "ms";
    VLOG(45) << "DESIRED EXPOSURE = " << new_desired_exposure_us;

    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    desired_exposure_time_us_ = new_desired_exposure_us;
#else
    LOG(FATAL) << "NOT SUPPORTED NOW.";
#endif
  }
  else if (options_.ae_method == AutoExposureMethod::kBlurPatchScore)
  {  // consider blur based on the tracking information
#ifdef USE_TRACKING_INFO
    VLOG(45) << "Method: patch score with blur modeling";
    vk::Timer t;
    int new_desired_exposure;
    if (options_.blur_use_single_frame)
    {
      new_desired_exposure = computeDesiredExposureTimeBlurSingleFrame(
          frame, last_exposure_time_us_);
    }
    else
    {
      new_desired_exposure =
          computeDesiredExposureTimeBlur(frame, last_exposure_time_us_);
    }
    VLOG(45) << "Computation Time: " << t.stop() * 1000 << "ms";
    VLOG(45) << "DESIRED EXPOSURE = " << new_desired_exposure;

    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    desired_exposure_time_us_ = new_desired_exposure;
#else
    LOG(FATAL) << "NOT SUPPORTED NOW";
#endif
  }
  VLOG(45) << "Computation Time: " << t.stop() * 1000 << "ms";
  VLOG(45) << "LAST EXPOSURE = " << last_exp_time_us_;
  VLOG(45) << "DESIRED EXPOSURE = " << new_desired_exp_us;

  // update setting values
  {
    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    desired_exposure_time_us_ = new_desired_exp_us;
    if (options_.auto_gain)
    {
      adjustGain();
    }
    if (desired_exposure_time_us_ < options_.min_exp_time_us)
    {
      VLOG(45) << "Desired exposure time too small: "
               << desired_exposure_time_us_ << " < " << options_.min_exp_time_us
               << ", clamping...";
      desired_exposure_time_us_ = options_.min_exp_time_us;
    }
    else if (desired_exposure_time_us_ >= options_.max_exp_time_us)
    {
      VLOG(45) << "Desired exposure time too big: " << desired_exposure_time_us_
               << " > " << options_.max_exp_time_us << "; clamping...";
      desired_exposure_time_us_ = options_.max_exp_time_us;
    }
  }
}

void AutoCameraSettings::adjustGain()
{
  if (desired_exposure_time_us_ < options_.dec_gain_exp_us)
  {
    desired_gain_x_ = last_gain_x_ - options_.gain_change_step;
    desired_gain_x_ = desired_gain_x_ < options_.min_gain ? options_.min_gain :
                                                            desired_gain_x_;
    desired_exposure_time_us_ = photometric_model_->getExpForNewGain(
        last_gain_x_, desired_gain_x_, desired_exposure_time_us_);
    VLOG(45) << "Decreasing gain from " << last_gain_x_ << " to "
             << desired_gain_x_;
    VLOG(45) << "New desired exposure time is  " << desired_exposure_time_us_;
  }
  else if (desired_exposure_time_us_ > options_.inc_gain_exp_us)
  {
    desired_gain_x_ = last_gain_x_ + options_.gain_change_step;
    desired_gain_x_ = desired_gain_x_ > options_.max_gain ? options_.max_gain :
                                                            desired_gain_x_;
    desired_exposure_time_us_ = photometric_model_->getExpForNewGain(
        last_gain_x_, desired_gain_x_, desired_exposure_time_us_);
    VLOG(45) << "Increasing gain from " << last_gain_x_ << " to "
             << desired_gain_x_;
    VLOG(45) << "New desired exposure time is  " << desired_exposure_time_us_;
  }
}

#ifdef USE_TRACKING_INFO
int IntensityScheduler::computeDesiredExposureTime(
    const svo::FramePtr& frame, const int last_exposure_time_us)
{
  CHECK_EQ(frame->numFeatures(), frame->centripetal_.size());

  cv::Mat img_rgb(frame->img().size(), CV_8UC3);
  cv::cvtColor(frame->img(), img_rgb, cv::COLOR_GRAY2RGB);
  std::vector<float> grad_vec;
  grad_vec.reserve(frame->numFeatures());
  std::vector<float> average_vec;
  average_vec.reserve(frame->numFeatures());
  std::vector<float> centripetal;
  centripetal.reserve(frame->numFeatures());
  for (size_t ft_index = 0; ft_index < frame->numFeatures(); ft_index++)
  {
    svo::FeatureWrapper ftr = frame->getFeatureWrapper(ft_index);
    FeaturePatch::Ptr patch = FeaturePatch::makePatch(
        frame->img_pyr_.at(ftr.level), (ftr.px / (1 << ftr.level)).cast<int>());
    if (patch)
    {
      float dScore_dExposure = patch->computeDScoreDExposure(
          *radiometric_model_, last_exposure_time_us / 1e3);

      // save
      grad_vec.push_back(dScore_dExposure);
      centripetal.push_back(frame->centripetal_[ft_index]);
      if (options_.use_adaptive_ratio)
      {
        average_vec.push_back(patch->computeAverage());
      }

      // visualization
      cv::Scalar color = (dScore_dExposure > 0.0f) ? cv::Scalar(0, 255, 0) :
                                                     cv::Scalar(255, 0, 0);
      cv::rectangle(img_rgb,
                    cv::Point2f(ftr.px[0] - 2, ftr.px[1] - 2),
                    cv::Point2f(ftr.px[0] + 2, ftr.px[1] + 2),
                    color,
                    -1);
    }
  }
  cv::imshow("img_rgb", img_rgb);
  cv::waitKey(10);

  return static_cast<int>(calNewExposureTimeFlagValue(
      centripetal, grad_vec, average_vec, last_exposure_time_us, options_));
}

int IntensityScheduler::computeDesiredExposureTimeBlur(
    const svo::FramePtr& cur_frame, const int last_exposure_time_us)
{
  cv::Mat img_rgb(cur_frame->img().size(), CV_8UC3);
  cv::cvtColor(cur_frame->img(), img_rgb, cv::COLOR_GRAY2RGB);
  std::vector<float> grad_vec;
  grad_vec.reserve(cur_frame->numFeatures());
  std::vector<float> average_vec;
  average_vec.reserve(cur_frame->numFeatures());
  std::vector<float>* average_vec_ptr =
      options_.use_adaptive_ratio ? &average_vec : nullptr;
  std::vector<float> centripetal;
  centripetal.reserve(cur_frame->numFeatures());
  for (size_t ft_index = 0; ft_index < cur_frame->numFeatures(); ft_index++)
  {
    svo::FeatureWrapper cur_ftr = cur_frame->getFeatureWrapper(ft_index);
    if (!cur_ftr.landmark)
    {
      VLOG(45) << "Current feature has no 3D landmark!";
      continue;
    }

    // get the reference frame
    svo::PointPtr landmark = cur_ftr.landmark;
    svo::FramePtr ref_frame = landmark->obs_[0].frame.lock();
    if (!ref_frame)
    {
      LOG(WARNING) << "cannot lock the reference frame!";
      continue;
    }
    size_t ref_id = landmark->obs_[0].keypoint_index_;
    int ref_level = ref_frame->getFeatureWrapper(ref_id).level;
    cv::Mat ref_img_pyr = ref_frame->img_pyr_.at(ref_level);

    cv::Mat cur_img_pyr = cur_frame->img_pyr_.at(cur_ftr.level);
    svo::Keypoint cur_px_pyr = (cur_ftr.px / (1 << cur_ftr.level));

    bool res = calGradientAverageVectorFromImages(cur_frame,
                                                  cur_ftr.level,
                                                  cur_img_pyr,
                                                  cur_px_pyr,
                                                  ref_frame,
                                                  ref_level,
                                                  ref_img_pyr,
                                                  landmark,
                                                  *radiometric_model_,
                                                  &grad_vec,
                                                  average_vec_ptr);

    // visualization
    if (res)
    {
      centripetal.push_back(cur_frame->centripetal_[ft_index]);
      cv::Scalar color = (grad_vec.back() > 0.0f) ? cv::Scalar(0, 255, 0) :
                                                    cv::Scalar(255, 0, 0);
      cv::rectangle(img_rgb,
                    cv::Point2f(cur_ftr.px[0] - 2, cur_ftr.px[1] - 2),
                    cv::Point2f(cur_ftr.px[0] + 2, cur_ftr.px[1] + 2),
                    color,
                    -1);
    }
  }
  cv::imshow("img_rgb", img_rgb);
  cv::waitKey(10);

  // based on the grad_vec, do the control
  return static_cast<int>(calNewExposureTimeFlagValue(
      centripetal, grad_vec, average_vec, last_exposure_time_us, options_));
}

// instead of query the reference frame for the radiance map,
// this function directly uses the current frame,
// this may be inaccurate due to the existing blur of the current frame
int IntensityScheduler::computeDesiredExposureTimeBlurSingleFrame(
    const svo::FramePtr& cur_frame, const int last_exposure_time_us)
{
  cv::Mat img_rgb(cur_frame->img().size(), CV_8UC3);
  cv::cvtColor(cur_frame->img(), img_rgb, cv::COLOR_GRAY2RGB);
  std::vector<float> grad_vec;
  std::vector<float> average_vec;
  std::vector<float>* average_vec_ptr =
      options_.use_adaptive_ratio ? &average_vec : nullptr;
  std::vector<float> centripetal;
  centripetal.reserve(cur_frame->numFeatures());
  for (size_t ft_index = 0; ft_index < cur_frame->numFeatures(); ft_index++)
  {
    svo::FeatureWrapper cur_ftr = cur_frame->getFeatureWrapper(ft_index);
    if (!cur_ftr.landmark)
    {
      VLOG(45) << "Current feature has no 3D landmark!";
      continue;
    }

    // create a patch out of the current feature
    cv::Mat cur_img_pyr = cur_frame->img_pyr_.at(cur_ftr.level);
    svo::Keypoint cur_px_pyr = cur_ftr.px / (1 << cur_ftr.level);

    bool res = calGradientAverageVectorFromImages(cur_frame,
                                                  cur_ftr.level,
                                                  cur_img_pyr,
                                                  cur_px_pyr,
                                                  cur_frame,
                                                  cur_ftr.level,
                                                  cur_img_pyr,
                                                  cur_ftr.landmark,
                                                  *radiometric_model_,
                                                  &grad_vec,
                                                  average_vec_ptr);

    // visualization
    if (res)
    {
      centripetal.push_back(cur_frame->centripetal_[ft_index]);
      cv::Scalar color = (grad_vec.back() > 0.0f) ? cv::Scalar(0, 255, 0) :
                                                    cv::Scalar(255, 0, 0);
      cv::rectangle(img_rgb,
                    cv::Point2f(cur_ftr.px[0] - 2, cur_ftr.px[1] - 2),
                    cv::Point2f(cur_ftr.px[0] + 2, cur_ftr.px[1] + 2),
                    color,
                    -1);
    }
  }
  cv::imshow("img_rgb", img_rgb);
  cv::waitKey(10);

  // based on the grad_vec, do the control
  return static_cast<int>(calNewExposureTimeFlagValue(
      centripetal, grad_vec, average_vec, last_exposure_time_us, options_));
}
#endif

#if 0
float IntensityScheduler::computeIntensityAtFeaturePos(
    const svo::Features& features, const cv::Mat& img, const int pyr_level,
    const int patch_halfsize)
{
  const float scale = 1.0f / (1 << pyr_level);
  const int patch_size = 2 * patch_halfsize;
  const int border = patch_size;
  const int stride = img.step;
  int total_intensity = 0;
  int n_fts = 0;
  for (const svo::FeatureWrapper& ftr : features)
  {
    // get coordinates at pixel
    const int u_cur_i = std::floor(ftr.px(0) * scale);
    const int v_cur_i = std::floor(ftr.px(1) * scale);

    // check if projection is within the image
    if (u_cur_i < 0 || v_cur_i < 0 || u_cur_i - border < 0 ||
        v_cur_i - border < 0 || u_cur_i >= img.cols - border ||
        v_cur_i >= img.rows - border)
      continue;

    for (int y = 0; y < patch_size; ++y)
    {
      uint8_t* img_ptr = (uint8_t*)img.data +
                         (v_cur_i - patch_halfsize + y) * stride +
                         (u_cur_i - patch_halfsize);

      for (int x = 0; x < patch_size; ++x, ++img_ptr)
      {
        total_intensity += *img_ptr;
      }
    }
    ++n_fts;
  }
  return static_cast<float>(total_intensity) /
         (n_fts * patch_size * patch_size);
}

float IntensityScheduler::computeImageStatistics(const svo::Features& features,
                                                 const svo::ImgPyr& img_pyr)
{
  // reset grid
  std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);

  // compute grid occupancy
  score_grid_vec_.resize(gamma_vec_.size());
  for (size_t i = 0; i < gamma_vec_.size(); ++i)
  {
    std::fill(score_grid_vec_[i].begin(), score_grid_vec_[i].end(), 0.0f);
    const float gamma = gamma_vec_.at(i);

    // set cells with features to value 1 in occupancy grid and compute feature
    // score
    for (const svo::FeatureWrapper& ftr : features)
    {
      if (ftr.px[0] < img_width_evaluated_ && ftr.px[1] < img_height_evaluated_)
      {
        const int cell_index =
            static_cast<int>(ftr.px[1] / grid_cell_size_) * grid_n_cols_ +
            static_cast<int>(ftr.px[0] / grid_cell_size_);

        score_grid_vec_[i][cell_index] = featureScore(img_pyr, ftr, gamma);
        occupancy_grid_[cell_index] = 1;
      }
    }

    // for all cells without a feature, compute the cell score
    const cv::Mat& img = img_pyr[pyr_level_];
    for (int y = 0; y < grid_n_rows_; ++y)
    {
      for (int x = 0; x < grid_n_cols_; ++x)
      {
        const int cell_index = grid_n_cols_ * y + x;
        if (occupancy_grid_[cell_index] == 0)
        {
          // compute cell score in cells that don't have a feature
          score_grid_vec_[i][cell_index] = cellScore(img, x, y, gamma);
        }
      }
    }
  }
}

float IntensityScheduler::featureScore(const svo::ImgPyr& img_pyr,
                                       const svo::FeatureWrapper& ftr,
                                       const float gamma)
{
  constexpr int halfpatch_size = 4;
  constexpr int patch_size = 2 * halfpatch_size;
  const cv::Mat& img = img_pyr[ftr.level];
  const int pitch = img.step;
  const int u = ftr.px[0] / (1 << ftr.level) - halfpatch_size;
  const int v = ftr.px[1] / (1 << ftr.level) - halfpatch_size;
  float a = 0;
  for (int y = 0; y < patch_size; ++y)
  {
    uint8_t* img_ptr = (uint8_t*)img.data + (v + y) * pitch + u;
    for (int x = 0; x < patch_size; ++x)
    {
      a += std::pow(
          std::pow(static_cast<float>(img_ptr[x + 1]) / 255.0, gamma) * 255 -
              std::pow(static_cast<float>(img_ptr[x - 1]) / 255.0, gamma) * 255,
          2);
    }
  }
  constexpr float image_noise = 25.0;
  constexpr float scale_factor = 1.0 / (2.0 * patch_size * patch_size);
  a *= scale_factor;
  double standard_deviation = std::sqrt(2.0 * image_noise / a);
  return standard_deviation;
}

void IntensityScheduler::saveImageStatistics(const svo::ImgPyr& img_pyr)
{
  ++statistics_id_;

  for (size_t g = 0; g < gamma_vec_.size(); ++g)
  {
    const float gamma = gamma_vec_[g];
    cv::Mat img = img_pyr[pyr_level_].clone();
    uint8_t* img_ptr = img.data;
    for (int i = 0; i < img.size().area(); ++i, ++img_ptr)
      *img_ptr = std::pow(static_cast<float>(*img_ptr) / 255.0, gamma) * 255;
    cv::imwrite("/tmp/active_" + std::to_string(statistics_id_) + "_" +
                    std::to_string(g) + ".png",
                img);
  }
}

void IntensityScheduler::setOverAndUnderexposedPixelsCount(const cv::Mat& img,
                                                           const int thresh)
{
  cv::Size size = img.size();
  if (img.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }
  int num_max = 0, num_min = 0;
  const int max_thresh = 255 - thresh;
  for (int i = 0; i < size.height; ++i)
  {
    const uchar* p = img.ptr<uchar>(i);
    for (int j = 0; j < size.width; ++j)
    {
      if (p[j] < thresh)
        ++num_min;
      else if (p[j] > max_thresh)
        ++num_max;
    }
  }
  // num_underexposed = num_min;
  // num_overexposed = num_max;
}

void IntensityScheduler::projectOccupancyXDir(
    std::vector<size_t>& proj_occupancy)
{
  proj_occupancy = std::vector<size_t>(grid_n_cols_, 0);
  for (int y = 0; y < grid_n_rows_; ++y)
  {
    for (int x = 0; x < grid_n_cols_; ++x)
    {
      if (occupancy_grid_.at(y * grid_n_cols_ + x) > 0)
        proj_occupancy[x] += 1;
    }
  }
}


float IntensityScheduler::cellScore(const cv::Mat& img,
                                    const int grid_x_coordinate,
                                    const int grid_y_coordinate,
                                    const float gamma)
{
  const int u = (grid_x_coordinate * grid_cell_size_pyr_);
  const int v = (grid_y_coordinate * grid_cell_size_pyr_);
  const int pitch = img.step;
  float a = 0;
  for (int y = 0; y < grid_cell_size_pyr_; ++y)
  {
    uint8_t* img_ptr = (uint8_t*)img.data + (v + y) * pitch + u;
    for (int x = 0; x < grid_cell_size_pyr_; ++x)
    {
      a += std::pow(
          std::pow(static_cast<float>(img_ptr[x + 1]) / 255.0, gamma) * 255 -
              std::pow(static_cast<float>(img_ptr[x]) / 255.0, gamma) * 255,
          2);
    }
  }
  constexpr float image_noise = 25.0;
  constexpr float scale_factor = 1.0 / (grid_cell_size_ * grid_cell_size_);
  a *= scale_factor;
  double standard_deviation = std::sqrt(2.0 * image_noise / a);
  return standard_deviation;
}
#endif

}  // namespace active_camera_control
