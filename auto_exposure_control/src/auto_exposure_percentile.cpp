#include "auto_exposure_control/auto_exposure_percentile.h"

#include <fstream>

#include <vikit/timer.h>

namespace ae_utils = auto_exposure_utils;

namespace auto_exposure
{
int AutoExposurePercentile::computeDesiredExposureWeightedGradient(
    const cv::Mat& img, const int last_exp_us, const float gain_x)
{
  int new_desired_exp_us;
  if (percentile_weights_.empty())
  {
    ae_utils::createSineWeights(static_cast<size_t>(img.cols * img.rows),
                                options_.weighted_gradient_order,
                                options_.weighted_gradient_pivotal,
                                &percentile_weights_);
  }
  if (options_.oneshot)
  {
    VLOG(45) << "Method: weighted gradient one shot";
    new_desired_exp_us =
        computeDesiredExposureWeightedGradientOneShot(img, last_exp_us, gain_x);
  }
  else
  {
    VLOG(45) << "Method: weighted gradient incremental";
    new_desired_exp_us =
        computeDesiredExposureWeightedGradientInc(img, last_exp_us, gain_x);
  }
  return new_desired_exp_us;
}

int AutoExposurePercentile::computeDesiredExposureWeightedGradientInc(
    const cv::Mat& img, const int last_exp_us, const float gain_x)
{
  vk::Timer timer;
  timer.start();
  int next_exposure_time_us = last_exp_us;

  std::vector<uint8_t> img_vec;
  ae_utils::cvMatToVector(img, &img_vec);

  VLOG(60) << "image vectorization took: " << timer.stop() * 1000.0 << " ms";
  timer.start();

  double mean_intensity =
      std::accumulate(img_vec.begin(), img_vec.end(), 0.0) / img_vec.size();
#ifdef DEBUG_OUTPUT
  std::sort(img_vec.begin(), img_vec.end());
  double med_intensity = img_vec[img_vec.size() / 2];
  past_states_.mean_intensities.add(mean_intensity);
  past_states_.med_intensities.add(med_intensity);
#endif
  VLOG(60) << "calcualte intensity statistics took: " << timer.stop() * 1000.0
           << " ms";
  timer.start();

  double over_ratio, under_ratio;
  ae_utils::overUnderExposedRatio(img_vec, 251, 4, &over_ratio, &under_ratio);
#ifdef DEBUG_OUTPUT
  past_states_.over_ratios.add(over_ratio);
  past_states_.under_ratios.add(under_ratio);
#endif

  VLOG(60) << "calcualte over/under exposure ratio took: "
           << timer.stop() * 1000.0 << " ms";
  timer.start();

  std::vector<double> irradiance_vec;
  photometric_model_->getExposureFromImageVec(img_vec, &irradiance_vec);
  std::transform(irradiance_vec.begin(),
                 irradiance_vec.end(),
                 irradiance_vec.begin(),
                 [&last_exp_us](double v)
                 {
                   return v / (1.0 * last_exp_us);
                 });

  VLOG(60) << "Calculating the irradiance took: " << timer.stop() * 1000.0
           << " ms";
  timer.start();

  std::nth_element(irradiance_vec.begin(),
                   static_cast<int>(irradiance_vec.size() * 0.6) +
                       irradiance_vec.begin(),
                   irradiance_vec.end());
  double perc_irrad =
      irradiance_vec[static_cast<size_t>(irradiance_vec.size() * 0.6)];
//  std::sort(irradiance_vec.begin(), irradiance_vec.end());
//  double perc_irrad = (irradiance_vec[irradiance_vec.size() * 0.6]);
#ifdef DEBUG_OUTPUT
  past_states_.median_irad.add(perc_irrad);
#endif

  VLOG(60) << "calcualte percentile irradiance took: " << timer.stop() * 1000.0
           << " ms";
  timer.start();

  // change the gradient ascent rate
  if (options_.auto_ga_rate)
  {
    double crt = perc_irrad;
    VLOG(45) << "Irradiance criteria: " << crt;
    options_.ga_rate = computeDesiredGARate(crt, options_.gain_scaling, gain_x,
                                            options_.auto_ga_rate_multiplier);
    VLOG(45) << "Changing gradient ascent rate to " << options_.ga_rate;
  }
#ifdef DEBUG_OUTPUT
  past_states_.raw_ga_rates.add(options_.ga_rate);
#endif

  VLOG(60) << "calcualte auto gradient ascent rate took: "
           << timer.stop() * 1000.0 << " ms";
  timer.start();

  // derivative
  std::vector<uint8_t> sorted_img_vec;
  std::vector<double> sorted_derivs_vec;
  std::vector<double> sorted_gradients_vec;
  double weighted_grad_deriv;
#ifdef DEBUG_OUTPUT
  ae_utils::computeSortedDGradientDExp(img,
                                       last_exp_us,
                                       photometric_model_,
                                       &sorted_derivs_vec,
                                       &sorted_gradients_vec,
                                       &sorted_img_vec);
#else
  ae_utils::computeSortedDGradientDExp(img,
                                       last_exp_us,
                                       photometric_model_,
                                       &sorted_derivs_vec,
                                       nullptr,
                                       &sorted_img_vec);
#endif
  VLOG(60) << "calcualte soft percentile derivatives took: "
           << timer.stop() * 1000.0 << " ms";
  timer.start();

  std::vector<double> sorted_derivs_comp(
      static_cast<size_t>(img.cols * img.rows));
  ae_utils::overUnderCompensateDerivs(
      sorted_derivs_vec, sorted_img_vec, comp_options_, &sorted_derivs_comp);

  VLOG(60) << "over/under exposure compensation took:  "
           << timer.stop() * 1000.0 << " ms";
  timer.start();
  weighted_grad_deriv =
      ae_utils::weightedSum(sorted_derivs_comp, percentile_weights_);
  weighted_grad_derivs_.add(weighted_grad_deriv);
#ifdef DEBUG_OUTPUT
  past_states_.weighted_grad_derivs.add(weighted_grad_deriv);
#endif

  VLOG(60) << "weighted sum took:  " << timer.stop() * 1000.0 << " ms";
  timer.start();

  // statistics
#ifdef DEBUG_OUTPUT
  double weighted_gradient =
      ae_utils::weightedSum(sorted_gradients_vec, percentile_weights_);
  past_states_.weighted_grad.add(weighted_gradient);

  double median_grad = sorted_gradients_vec[sorted_gradients_vec.size() / 2];
  past_states_.median_grad.add(median_grad);

  VLOG(45) << "calcualte gradient statitics took: "
           << timer.stop() * 1000.0 << " ms";
  timer.start();
#endif

  // acceleration extensions
  double comp_ga_rate = options_.ga_rate;
  // when there is heavy over-exposure and gradient derivative is negative
  // increase the gradient ascent rate
  if (weighted_grad_deriv < 0 && over_ratio > options_.rate_over_comp_thresh)
  {
    VLOG(45) << "Overexposure: increasing gradient ascent rate.";
    comp_ga_rate *= std::pow(50,
                             (1.0 / (1.0 - options_.rate_over_comp_thresh)) *
                                 (over_ratio - options_.rate_over_comp_thresh));
    VLOG(45) << "New gradient ascent rate: " << comp_ga_rate;
  }

  // gain momentum in case of mono
  if (options_.use_slope_comp && weighted_grad_derivs_.isFilled())
  {
    // detect monotonic increasing/descreasing
    std::vector<int> inc;
    for (size_t i = 0; i < weighted_grad_derivs_.size(); i++)
    {
      inc.push_back(weighted_grad_derivs_.get(i) > 0 ? 1 : 0);
    }
    int sum = std::accumulate(inc.begin(), inc.end(), 0);
    bool monotonic = (sum == 0 || sum == static_cast<int>(inc.size()));

    if (monotonic)
    {
      VLOG(45) << "Monotonic increasing/decreasing detected.";
      double wg_deriv_mean = weighted_grad_derivs_.mean();
      double wg_deriv_stdev = weighted_grad_derivs_.stdev();
      double ratio = wg_deriv_stdev / std::fabs(wg_deriv_mean);
      VLOG(45) << "weighted gradient derivative mean: " << wg_deriv_mean
               << ", stdev: " << wg_deriv_stdev << ", ratio: " << ratio;
      if (ratio < options_.slope_inc_thresh)
      {
        slope_scale_ *= options_.slope_inc_rate;

        // sanity check
        slope_scale_ = slope_scale_ > 1000 ? 1000 : slope_scale_;
        VLOG(45) << "Long monotonic update detected, "
                 << "increased slope scale to " << slope_scale_;
      }
      else if (ratio > options_.slope_inc_thresh + 0.1)
      {
        slope_scale_ /= (1.5 * (ratio - options_.slope_inc_thresh) / 0.1);

        slope_scale_ = slope_scale_ < 1.0 ? 1.0 : slope_scale_;
        VLOG(45) << "Restore slope scale to " << slope_scale_;
      }
      else
      {
        VLOG(45) << "Current slope scale " << slope_scale_;
      }
    }
    else
    {
      // reset when monotonic breaks
      slope_scale_ = 1.0;
      VLOG(45) << "Reset slope scale to " << slope_scale_;
    }

    comp_ga_rate *= slope_scale_;
  }

#ifdef DEBUG_OUTPUT
  past_states_.comp_ga_rates.add(comp_ga_rate);
#endif
  double cur_update = comp_ga_rate * weighted_grad_deriv;

  VLOG(60) << "compensate gradient rate using time history / over exposed area "
              "took " << timer.stop() * 1000.0 << " ms";
  timer.start();

  // avoid being stuck at total dark or bright images
  if (mean_intensity > options_.use_intensity_high_bound ||
      mean_intensity < options_.use_intensity_low_bound ||
      over_ratio > options_.rate_over_comp_thresh)
  {
    VLOG(45) << "Mean intensity too extreme, using desired intensity.";
    cur_update = last_exp_us * (127 / mean_intensity - 1.0);
    VLOG(45) << "Mean intensity too extreme, using desired intensity: "
             << cur_update;
  }

  VLOG(60) << "switch to intensity based method took: " << timer.stop() * 1000.0
           << " ms";
  timer.start();

  // check if update is too small or too large
  if (std::fabs(cur_update) < options_.min_update)
  {
    cur_update = 0;
    VLOG(45) << "update is too small, not going to change.";
  }

  if (std::fabs(cur_update * 1.0 / last_exp_us) > 0.1)
  {
    VLOG(45) << "Update too large, clamping.";
    cur_update = (cur_update > 0 ? 1 : -1) * 0.1 * last_exp_us;
  }
  next_exposure_time_us += static_cast<int>(cur_update);
#ifdef DEBUG_OUTPUT
  past_states_.exp_updates.add(cur_update);
#endif

  VLOG(60) << "clamping took: " << timer.stop() * 1000.0 << " ms";
  timer.start();

// verbose output
#ifdef DEBUG_OUTPUT
  past_states_.check();
  if (past_states_.isFilled())
  {
    VLOG(60) << "idx\t raw_ga\t comp_ga\t update\t w_deriv\t w_grad";
    for (size_t i = 0; i < past_states_.size(); i++)
    {
      VLOG(60) << i << "\t" << past_states_.raw_ga_rates.get(i) << std::setw(10)
               << "\t" << past_states_.comp_ga_rates.get(i) << std::setw(10)
               << "\t" << past_states_.exp_updates.get(i) << std::setw(10)
               << "\t" << past_states_.weighted_grad_derivs.get(i)
               << std::setw(10) << "\t" << past_states_.weighted_grad.get(i)
               << "\n";
    }
    VLOG(60) << "===================================================";
    VLOG(60) << "idx\t over\t under\t med_irad\t med_grad\t mean_int\t "
                "med_int\n";
    for (size_t i = 0; i < past_states_.size(); i++)
    {
      VLOG(60) << i << "\t" << past_states_.over_ratios.get(i) << std::setw(10)
               << "\t" << past_states_.under_ratios.get(i) << std::setw(10)
               << "\t" << past_states_.median_irad.get(i) << std::setw(10)
               << "\t" << past_states_.median_grad.get(i) << "\t"
               << past_states_.mean_intensities.get(i) << "\t"
               << past_states_.med_intensities.get(i) << "\n";
    }
  }

  VLOG(45) << "logging and output took: " << timer.stop() * 1000.0 << " ms";
  timer.start();
#endif

  return next_exposure_time_us;
}

int AutoExposurePercentile::computeDesiredExposureWeightedGradientOneShot(
    const cv::Mat& img, const int last_exp_us, const float gain_x)
{
  int new_exposure_time_us = last_exp_us;

  // otherwise we can do gradient ascent
  // fix iteration
  // TODO: adaptive depending on current frame rate
  const int max_iter = options_.oneshot_maxiter;

  int org_exp_us = last_exp_us;
  cv::Mat org_exposure;
  photometric_model_->getExposureFromImage(img, &org_exposure);
  cv::Mat org_irradiance_us = org_exposure / (1.0 * org_exp_us);

  // perform accelerated gradient ascent
  int cur_exp_us = org_exp_us;
  double last_update = 0.0;
  for (int i = 0; i < max_iter; i++)
  {
    int deriv_exp_us = cur_exp_us;

    // calcualte the point to evaluate gradient
    cv::Mat deriv_img;
    if (std::fabs(deriv_exp_us - org_exp_us) < 10)
    {
      deriv_img = img;
    }
    else
    {
      cv::Mat deriv_exposure = org_irradiance_us * (deriv_exp_us * 1.0);
      photometric_model_->getImageFromExposure(deriv_exposure, &deriv_img);
    }

    // derivative
    double deriv =
        computeDWeightedGradientDExposure(deriv_img, deriv_exp_us, gain_x);
    VLOG(60) << "derivative: " << deriv;

    // update
    double new_update = options_.ga_rate * deriv;

    cur_exp_us += static_cast<int>(new_update);
    last_update = new_update;
  }

  if (std::fabs(new_exposure_time_us - cur_exp_us) < options_.min_update)
  {
    VLOG(45) << "update is too small, not going to change.";
  }
  else
  {
    VLOG(45) << "Update: " << cur_exp_us - new_exposure_time_us;
    new_exposure_time_us = cur_exp_us;
  }

  return new_exposure_time_us;
}

double AutoExposurePercentile::computeDWeightedGradientDExposure(
    const cv::Mat& img, const int exp_us, const float gain_x)
{
  CHECK(photometric_model_);
  size_t px_cnt = static_cast<size_t>(img.cols * img.rows);
  CHECK_EQ(percentile_weights_.size(), px_cnt);

  std::vector<uint8_t> sorted_img;
  std::vector<double> raw_derivs_sorted;
  ae_utils::computeSortedDGradientDExp(img,
                                       exp_us,
                                       photometric_model_,
                                       &raw_derivs_sorted,
                                       nullptr,
                                       &sorted_img);

  // do over-/under-exposure compensation;
  std::vector<double> raw_derivs_sorted_comp(
      static_cast<size_t>(img.cols * img.rows));
  ae_utils::overUnderCompensateDerivs(
      raw_derivs_sorted, sorted_img, comp_options_, &raw_derivs_sorted_comp);

  double weighted_grad_deriv =
      ae_utils::weightedSum(raw_derivs_sorted_comp, percentile_weights_);

  return weighted_grad_deriv;
}

void AutoExposurePercentile::init()
{
  // load auto gradient ascent rate profile
  CHECK(!options_.ga_rate_profile.empty());

  std::ifstream ga_prof(options_.ga_rate_profile);
  CHECK(ga_prof.is_open());
  VLOG(45) << "Loading gradient ascent rate profile "
           << options_.ga_rate_profile;

  int num_samples;
  ga_prof >> num_samples;
  VLOG(45) << "Number of samples: " << num_samples;
  med_irad_.resize(static_cast<size_t>(num_samples));
  ga_rate_.resize(static_cast<size_t>(num_samples));

  ga_prof.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  for (int i = 0; i < num_samples; i++)
  {
    ga_prof >> med_irad_[static_cast<size_t>(i)];
  }

  ga_prof.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  for (int i = 0; i < num_samples; i++)
  {
    ga_prof >> ga_rate_[static_cast<size_t>(i)];
  }
  ga_prof.close();

  VLOG(45) << "Med irad\tGA rate:";
  for (int i = 0; i < num_samples; i++)
  {
    VLOG(45) << med_irad_[static_cast<size_t>(i)] << "\t"
             << ga_rate_[static_cast<size_t>(i)];
  }

  // first
  kl_ = ga_rate_[0] + std::log10(med_irad_[0]);
  VLOG(45) << "kl: " << kl_;

  // middle
  for (size_t i = 0; i < med_irad_.size() - 1; i++)
  {
    double k =
        (ga_rate_[i + 1] - ga_rate_[i]) / (med_irad_[i + 1] - med_irad_[i]);
    double b = ga_rate_[i] - k * med_irad_[i];
    km_.push_back(k);
    bm_.push_back(b);
  }

  // last
  kh_ = ga_rate_.back() / std::pow(10.0, -med_irad_.back());
  VLOG(45) << "kh: " << kh_;
}

double AutoExposurePercentile::computeDesiredGARate(const double med_irad,
                                                    const bool gain_scaling,
                                                    const float gain_x,
                                                    const double multiplier)
{
  double gain_multiplier = 1.0;
  if (gain_scaling)
  {
    gain_multiplier = 1.0 - 0.20 * (gain_x - 1.0);
    //    gain_multiplier = 1.0 / gain_x;
  }
  gain_multiplier *= multiplier;

  if (med_irad <= med_irad_[0])
  {
    return gain_multiplier * (kl_ - std::log10(med_irad));
  }
  else if (med_irad >= med_irad_.back())
  {
    return gain_multiplier * kh_ * std::pow(10.0, -med_irad);
  }
  else
  {
    for (size_t i = 0; i < med_irad_.size() - 1; i++)
    {
      if (med_irad <= med_irad_[i + 1])
      {
        return gain_multiplier * (km_[i] * med_irad + bm_[i]);
      }
    }
  }
  LOG(FATAL) << "Should not reach here.";
}
}
