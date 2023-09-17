#pragma once

#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include <condition_variable>
#include <array>
#include <utility>  // std::pair
#include <cstdint>  // std::uint8_t

#include <ros/node_handle.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <photometric_camera/photometric_model.h>

#include "auto_exposure_control/auto_exposure_percentile.h"
#include "auto_exposure_control/auto_exposure_shim.h"
#include "auto_exposure_control/auto_exposure_sweep.h"
#include "auto_exposure_control/auto_exposure_intensity.h"
#include "auto_exposure_control/frame.h"

namespace auto_exposure
{
using namespace Eigen;

enum class AutoExposureMethod
{
  // do not change the exposure time
  kFixed,

  // use the derivative of weighted gradient to change the exposure time
  kPercentile,

  // sweep the parameter
  kSweep,

  // Shim, Inwook, Joon-Young Lee, and In So Kweon. "Auto-adjusting camera
  // exposure for outdoor robotics using gradient information."
  kShim,

  // Average Intensity
  kMeanIntensity,

  // tightly coupled with tracking information
  kPatchScore,
  kBlurPatchScore
};

struct AutoCameraOptions
{
  AutoExposureMethod ae_method;

  // useful if the calculation time is too long
  bool use_parallel_thread = false;

  // pyramid level to calculate image statistics for exposure control
  int pyramid_lvl = 3;

  // exposure time range
  // The maximum exposure time should be set based on the frame rate.
  // Use setMaxExposureTime(), which will also set the dec/inc_gain_exp_us.
  int max_exp_time_us = 15000;
  int min_exp_time_us = 20;

  // auto gain settings
  bool auto_gain = true;
  float max_gain = 4.0;
  float min_gain = 1.0;
  float gain_change_step = 0.5;
  int dec_gain_exp_us = 2000;
  int inc_gain_exp_us = 10000;

  // options for kShim
  auto_exposure::AutoExposureShimOptions shim_options_;

  // options for kWeightedGradient
  auto_exposure::AutoExposurePercentileOptions perc_options;

  // options for sweep method
  auto_exposure::AutoExposureSweepOptions sweep_options_;

  // shared option for over-/under-compensation
  auto_exposure_utils::OverUnderExposureCompOptions comp_options_;

#ifdef USING_TRACKING_INFO
  // options for patch score methods
  int patch_halfsize = 2;
  // only change when abs(gradient) is larger
  double radiometric_gradient_thresh = 200;
  // how we decide which gradient to look at
  bool use_mean_gradient = false;
  bool use_weighted_mean = false;
  float pivot_gradient_ratio = 0.5;
  bool use_adaptive_ratio = false;  // can help avoid overexposure
  // exposure step ratio
  bool use_gradient_ascent = true;
  float exposure_change_step = 0.03f;  // lower is more steady but slower
  float gradient_ascent_rate = 0.1f;   // learning rate for gradient ascent

  // when considering blur, use current frame only for simplification
  bool blur_use_single_frame = false;
#endif
};

class AutoCameraSettings
{
public:
  typedef std::shared_ptr<AutoCameraSettings> Ptr;

  AutoCameraSettings(
      const AutoCameraOptions& options,
      const photometric_camera::PhotometricModel::Ptr& radiometric_model);
  ~AutoCameraSettings()
  {
    if (options_.use_parallel_thread)
    {
      stopThread();
    }
  }


  void setNextFrame(const auto_exposure::FramePtr& frame,
                    const int exposure_time_us,
                    const float gain_x);

  void setNextFrame(const cv::Mat& frame,
                    const int exposure_time_us,
                    const float gain_x);

  bool isFrameSet()
  {
    std::lock_guard<std::mutex> lock(next_frame_mutex_);
    return next_frame_ != nullptr;
  }

  // get the exposure time set by the radiometric method
  int getDesiredExposureTimeUs() const
  {
    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    return desired_exposure_time_us_;
  }

  float getDesiredGainX() const
  {
    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    return desired_gain_x_;
  }

  void setDesiredExposureTimeUs(const int exp_us)
  {
    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    desired_exposure_time_us_ = exp_us;
  }

  void setDesiredGainX(const float gain_x)
  {
    std::lock_guard<std::mutex> lock(desired_values_mutex_);
    desired_gain_x_ = gain_x;
  }

  void setMaxExposureTime(const int max_exp_us)
  {
    options_.max_exp_time_us = max_exp_us;
    options_.inc_gain_exp_us = static_cast<int>(0.9 * max_exp_us);
    options_.dec_gain_exp_us = static_cast<int>(0.2 * max_exp_us);

    options_.sweep_options_.max_exp_us = max_exp_us;
    VLOG(45) << "Set max exposure time to " << max_exp_us
             << "; Threshold to increase gain is " << options_.inc_gain_exp_us
             << "; Threshold to decrease gain is " << options_.dec_gain_exp_us;
  }

  AutoCameraOptions options_;

private:

  void threadLoop();

  void startThread();

  void stopThread();

  void calculateNewSettings(const auto_exposure::FramePtr& frame);

  void adjustGain();

  photometric_camera::PhotometricModel::Ptr photometric_model_;

  // thread stuff
  std::shared_ptr<std::thread> thread_;
  std::atomic<bool> quit_thread_;

  // locked variables:
  // {
  std::condition_variable next_frame_condvar_;
  mutable std::mutex next_frame_mutex_;
  auto_exposure::FramePtr next_frame_;
  int last_exp_time_us_;
  float last_gain_x_;

  mutable std::mutex desired_values_mutex_;
  int desired_exposure_time_us_;
  float desired_gain_x_;
  // }

  auto_exposure::AutoExposurePercentilePtr auto_exp_perc_;
  auto_exposure::AutoExposureShimPtr auto_exp_shim_;
  auto_exposure::AutoExposurSweepPtr auto_exp_sweep_;
  auto_exposure::AutoExposureIntensityPtr auto_exp_intensity_;

#ifdef USE_TRACKING_INFO
  // use photometric model to predict exposure
  // for each patch, compute the dScore/dExposure
  // then use the MEDIAN value of all derivatives
  // to change the exposure
  int computeDesiredExposureTime(const svo::FramePtr& frame,
                                 const int last_exposure_time_us);

  // the following two functions also consider motion blur
  int computeDesiredExposureTimeBlur(const svo::FramePtr& cur_frame,
                                     const int last_exposure_time_us);

  int computeDesiredExposureTimeBlurSingleFrame(
      const svo::FramePtr& cur_frame, const int last_exposure_time_us);

#endif

// utility functions
#if 0
  static constexpr int img_width_ = 752;
  static constexpr int img_height_ = 480;
  static constexpr int pyr_level_ = 0;
  static constexpr int grid_cell_size_ = 32;
  static constexpr int grid_cell_size_pyr_ =
      grid_cell_size_ / (1 << pyr_level_);
  static constexpr int grid_n_cols_ =
      img_width_ / grid_cell_size_;  // round to floor
  static constexpr int grid_n_rows_ =
      img_height_ / grid_cell_size_;  // round to floor
  static constexpr int img_width_evaluated_ = grid_cell_size_ * grid_n_cols_;
  static constexpr int img_height_evaluated_ = grid_cell_size_ * grid_n_rows_;
  std::vector<float> gamma_vec_;
  typedef std::array<uint8_t, grid_n_cols_ * grid_n_rows_> OccupancyGrid;
  typedef std::array<float, grid_n_cols_ * grid_n_rows_> ScoreGrid;
  typedef std::array<float, grid_n_cols_> ScoreVec;

  // image statistics
  size_t statistics_id_ = 0;
  OccupancyGrid occupancy_grid_;
  std::vector<ScoreGrid> score_grid_vec_;
  static float computeIntensityAtFeaturePos(const svo::Features& features,
                                            const cv::Mat& img,
                                            const int pyr_level,
                                            const int patch_halfsize);

  // compute the feature score (for cells with features) and
  // cell score (for cells without feature)
  float computeImageStatistics(const svo::Features& features,
                               const svo::ImgPyr& img_pyr);
  static float featureScore(const svo::ImgPyr& img_pyr,
                            const svo::FeatureWrapper& ftr, const float gamma);
  void saveImageStatistics(const svo::ImgPyr& img_pyr);

  // TODO(zzc): no return value?
  void setOverAndUnderexposedPixelsCount(const cv::Mat& img,
                                         const int max_thresh);

  // TODO(zzc): why only the x direction?
  // TODO(zzc): what is the statistics used for?
  // the following two functions implement x-gradient based score
  // noise / accumulated gradient
  // should be positively correlated to match error

  static float cellScore(const cv::Mat& img, const int grid_x_coordinate,
                         const int grid_y_coordinate, const float gamma);



  void projectOccupancyXDir(std::vector<size_t>& proj_occupancy);
#endif
};

}  // namespace active_camera_control
