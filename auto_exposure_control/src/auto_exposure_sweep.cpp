#include "auto_exposure_control/auto_exposure_sweep.h"

namespace
{
void linspace(const int num,
              const int start,
              const int end,
              std::vector<int>* samples)
{
  CHECK_NOTNULL(samples);
  CHECK_LT(start, end);
  samples->resize(static_cast<size_t>(num));

  double interval = ((end - start) * 1.0) / (num - 1);
  for (size_t i = 0; i < samples->size(); i++)
  {
    (*samples)[i] = static_cast<int>(start + i * interval);
  }
}

// y = kx + t
double computeZeroCrossingLinear(const double a,
                                 const double fa,
                                 const double b,
                                 const double fb)
{
  double k = (fb - fa) / (b - a);
  double t = fa - k * a;

  VLOG(45) << "k is " << k;
  VLOG(45) << "t is " << t;

  return -(t / k);
}
}

namespace auto_exposure
{
int AutoExposureSweep::computeDesiredExposureSweep(const cv::Mat& img,
                                                   const int last_exp_us,
                                                   const double gain_x)
{
  logger_.log("index", log_cnt_++);
  logger_.log("gain_x", gain_x);
  logger_.log("exp_us", last_exp_us);

  VLOG(45) << "Exposure time(us): " << last_exp_us;
  CHECK(!img.empty());
  size_t px_cnt = static_cast<size_t>(img.cols * img.rows);
  if (percentile_weights_.empty())
  {
    ae_utils::createSineWeights(px_cnt,
                                options_.weighted_gradient_order,
                                options_.weighted_gradient_pivotal,
                                &percentile_weights_);
  }

  // compute the derivative
  std::vector<double> sorted_derivs_vec;
  std::vector<double> sorted_gradients_vec;
  std::vector<uint8_t> sorted_img_vec;
  ae_utils::computeSortedDGradientDExp(img,
                                       last_exp_us,
                                       photometric_model_,
                                       &sorted_derivs_vec,
                                       &sorted_gradients_vec,
                                       &sorted_img_vec);

  // compensate
  std::vector<double> comp_sorted_derivs_vec(sorted_derivs_vec);
  if (options_.use_compensation_captured)
  {
    ae_utils::overUnderCompensateDerivs(sorted_derivs_vec,
                                        sorted_img_vec,
                                        comp_options_,
                                        &comp_sorted_derivs_vec);
  }
  else
  {
    VLOG(45) << "Will not use compensation for the captured image.";
  }

  // accumulate to get the derivative
  double weighted_grad_deriv =
      ae_utils::weightedSum(comp_sorted_derivs_vec, percentile_weights_);
  VLOG(45) << "Weighted gradient derivative: " << weighted_grad_deriv;
  double weighted_grad =
      ae_utils::weightedSum(sorted_gradients_vec, percentile_weights_);
  logger_.log("wg", weighted_grad);
  logger_.log("wg_deriv", weighted_grad_deriv);

  // statistics about over/under exposure
  double over_ratio, under_ratio;
  ae_utils::overUnderExposedRatio(
      sorted_img_vec, 251, 4, &over_ratio, &under_ratio);
  VLOG(45) << "Over/Under exposed pixel ratio: " << over_ratio << "/"
           << under_ratio;


  // create samples
  std::vector<int> samples(static_cast<size_t>(options_.num_samples));
  if (weighted_grad_deriv > 0)
  {
    VLOG(45) << "Derivative > 0";
    int max_sampled = options_.max_exp_us;
    if (max_sampled - last_exp_us > options_.sample_max_range)
    {
      max_sampled = last_exp_us + options_.sample_max_range;
    }
    linspace(options_.num_samples, last_exp_us, max_sampled, &samples);
  }
  else if (weighted_grad_deriv < 0)
  {
    VLOG(45) << "Derivative < 0";
    int min_sampled = options_.min_exp_us;
    if (last_exp_us - min_sampled > options_.sample_max_range)
    {
      min_sampled = last_exp_us - options_.sample_max_range;
    }
    linspace(options_.num_samples, min_sampled, last_exp_us, &samples);
  }

  // compute weighted gradient score/derivatives for sampled exposure times
  std::vector<double> pred_wg_scores(samples.size(), 0.0);
  std::vector<double> pred_wg_derivs(samples.size(), 0.0);
  std::vector<cv::Mat> predict_imgs(samples.size());

  // calculate the irradiance for once
  cv::Mat org_exposure;
  photometric_model_->getExposureFromImage(img, &org_exposure);
  cv::Mat org_irradiance_us = org_exposure / (1.0 * last_exp_us);

  std::vector<double> irradiance_vec;
  ae_utils::cvMatToVector(org_irradiance_us, &irradiance_vec);

  std::sort(irradiance_vec.begin(), irradiance_vec.end());
  double median_irrad = irradiance_vec[irradiance_vec.size() / 2];
  double mean_irrad =
      std::accumulate(irradiance_vec.begin(), irradiance_vec.end(), 0.0) /
      irradiance_vec.size();
  logger_.log("med_irad", median_irrad);
  logger_.log("mean_irad", mean_irrad);

  // predict images
  for (size_t i = 0; i < predict_imgs.size(); i++)
  {
    int test_exp_us = samples[i];
    cv::Mat pred_exposure = org_irradiance_us * (test_exp_us * 1.0);
    photometric_model_->getImageFromExposure(pred_exposure, &(predict_imgs[i]));
  }

  if (!options_.use_compensation_predicted)
  {
    VLOG(45) << "Will not use over/under-exposure compensation for predicted.";
  }
  // calcualte scores and derivatives
  for (size_t i = 0; i < samples.size(); i++)
  {
    int test_exp_us = samples[i];

    std::vector<double> pred_sorted_deriv_vec;
    std::vector<double> pred_sorted_gradients_vec;
    std::vector<uint8_t> pred_sorted_img_vec;
    ae_utils::computeSortedDGradientDExp(predict_imgs[i],
                                         test_exp_us,
                                         photometric_model_,
                                         &pred_sorted_deriv_vec,
                                         &pred_sorted_gradients_vec,
                                         &pred_sorted_img_vec);

    std::vector<double> pred_comp_sorted_deriv_vec(pred_sorted_deriv_vec);
    if (options_.use_compensation_predicted)
    {
      ae_utils::overUnderCompensateDerivs(pred_sorted_deriv_vec,
                                          pred_sorted_img_vec,
                                          comp_options_,
                                          &pred_comp_sorted_deriv_vec);
    }

    pred_wg_scores[i] =
        ae_utils::weightedSum(pred_sorted_gradients_vec, percentile_weights_);
    pred_wg_derivs[i] =
        ae_utils::weightedSum(pred_comp_sorted_deriv_vec, percentile_weights_);
  }

  // check consistency
  if ((weighted_grad_deriv > 0 && pred_wg_derivs[0] < 0) ||
      (weighted_grad_deriv < 0 && pred_wg_derivs.back() > 0))
  {
    VLOG(45) << "Derivative is close to zero (sign changes), not change.";
    logger_.log("exp_update", 0);
    return last_exp_us;
  }

  // get the sign of the derivatives
  std::vector<bool> is_positive(samples.size(), false);
  for (size_t i = 0; i < samples.size(); i++)
  {
    is_positive[i] = (pred_wg_derivs[i] > 0);
  }

  // verbose
  VLOG(60) << "EXP\t SCORE\t DERIV";
  for (size_t i = 0; i < samples.size(); i++)
  {
    VLOG(60) << samples[i] << "\t" << pred_wg_scores[i] << "\t"
             << pred_wg_derivs[i] << "\t" << is_positive[i];
  }

  // get the zero crossing index
  size_t before_zc = 0;
  for (size_t i = 1; i < is_positive.size(); i++)
  {
    if (is_positive[i] != is_positive[0])
    {
      break;
    }
    before_zc++;
  }

  // calculate desired exposure time
  int exp_us_change;
  if (before_zc == samples.size() - 1)
  {
    before_zc = 0;
    VLOG(45) << "No zero crossing found, set to limit.";

    if (weighted_grad_deriv > 0)
    {
      exp_us_change = options_.max_exp_us - last_exp_us;
    }
    else
    {
      exp_us_change = last_exp_us - options_.min_exp_us;
    }
  }
  else
  {
    VLOG(45) << "zero crossing index: " << before_zc;

    int a_exp_us = samples[before_zc];
    double a_exp_deriv = pred_wg_derivs[before_zc];
    int b_exp_us = samples[before_zc + 1];
    double b_exp_deriv = pred_wg_derivs[before_zc + 1];
    VLOG(45) << "a : " << a_exp_us << ", b: " << b_exp_us
             << ", a_deriv: " << a_exp_deriv << ", b_deriv: " << b_exp_deriv;

    if (std::fabs(a_exp_deriv) < 1e-4 || std::fabs(b_exp_deriv) < 1e-4)
    {
      VLOG(45) << "At least one of the zero crossing is close to zero:"
               << "\n ZC-index: " << before_zc << "\n a_exp: " << a_exp_us
               << ", a_exp_deriv: " << a_exp_deriv << "\n b_exp: " << b_exp_us
               << ", b_exp_deriv: " << b_exp_deriv;
      exp_us_change = 0;
    }
    else
    {
      double zc = computeZeroCrossingLinear(
          a_exp_us, a_exp_deriv, b_exp_us, b_exp_deriv);

      int raw_desired_exp_us = static_cast<int>(zc);
      VLOG(45) << "Desired exposure time based on zero crossing: "
               << raw_desired_exp_us;
      exp_us_change = raw_desired_exp_us - last_exp_us;
      VLOG(45) << "Desired exposure time change based on zero crossing: "
               << exp_us_change;
    }
  }

  // add damping, proportional control
  int damped_inc_exp_us =
      static_cast<int>(exp_us_change * options_.update_damping);
  if (exp_us_change > 0)
  {
    if (under_ratio > options_.under_comp_thresh)
    {
      damped_inc_exp_us *= (1.0 + under_ratio);
    }
  }
  else
  {
    if (over_ratio > options_.over_comp_thresh)
    {
      damped_inc_exp_us *= (1.0 + over_ratio);
    }
  }

  logger_.log("exp_update", damped_inc_exp_us);

  logger_.checkConsistency();
  logger_.incValuesCnt();

  return clampExposureTime(last_exp_us + damped_inc_exp_us);
}
}
