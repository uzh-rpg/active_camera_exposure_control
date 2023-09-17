#include "auto_exposure_control/auto_exp_utils.h"

#include <vikit/timer.h>

namespace auto_exposure_utils
{
std::ostream& operator<<(std::ostream& os, const Logger& st)
{
  const std::string delim(" ");

  // header
  os << "#" << delim;
  for (const std::string& name : st.names)
  {
    os << name << delim;
  }
  os << std::endl;

  // each line
  for (size_t i = 0; i < st.log_cnt; i++)
  {
    for (const std::vector<double>& s : st.data)
    {
      os << s[i] << delim;
    }
    os << std::endl;
  }

  return os;
}

void computeScharr(const cv::Mat& img, cv::Mat* img_dx, cv::Mat* img_dy)
{
  CHECK_NOTNULL(img_dx);
  CHECK_NOTNULL(img_dy);
  static constexpr double scharr_normalizer = 32.0;
  cv::Mat raw_img_dx;
  cv::Scharr(img, raw_img_dx, CV_64F, 1, 0, 3);
  *img_dx = raw_img_dx / scharr_normalizer;
  cv::Mat raw_img_dy;
  cv::Scharr(img, raw_img_dy, CV_64F, 0, 1, 3);
  *img_dy = raw_img_dy / scharr_normalizer;
}

void computeGradient(const cv::Mat& dx, const cv::Mat& dy, cv::Mat* gradient)
{
  CHECK(!dx.empty());
  CHECK(!dy.empty());
  CHECK_NOTNULL(gradient);
  CHECK_EQ(dx.cols, dy.cols);
  CHECK_EQ(dx.rows, dy.rows);

  gradient->create(dx.rows, dx.cols, CV_64F);

  for (int ix = 0; ix < dx.cols; ix++)
  {
    for (int iy = 0; iy < dx.rows; iy++)
    {
      gradient->at<double>(iy, ix) = std::pow(dx.at<double>(iy, ix), 2) +
                                     std::pow(dy.at<double>(iy, ix), 2);
    }
  }
}

void createSineWeights(const size_t num,
                       const double order,
                       const double percentile_ratio,
                       std::vector<double>* weights)
{
  CHECK_NOTNULL(weights);
  CHECK_GT(percentile_ratio, 0.0);
  CHECK_LT(percentile_ratio, 1.0);
  weights->resize(num);

  // first part: [0, pi/2]
  // second part: (pi/2, pi]
  size_t num_first = static_cast<size_t>(num * percentile_ratio);
  size_t num_second = num - num_first;
  // generate first half
  double step_first = M_PI_2 / (num_first - 1);
  for (size_t i = 0; i < num_first; i++)
  {
    (*weights)[i] = std::pow(std::sin(i * step_first), order);
  }
  (*weights)[num_first - 1] = 1.0;

  // generate second half
  double step_second = M_PI_2 / num_second;
  for (size_t i = num_first; i < num; i++)
  {
    (*weights)[i] =
        std::pow(std::sin(M_PI_2 - (i - num_first + 1) * step_second), order);
  }

  double sum = std::accumulate(weights->begin(), weights->end(), 0.0);
  std::for_each(weights->begin(),
                weights->end(),
                [&sum](double& e)
                {
                  e /= sum;
                });
}

void overUnderExposedRatio(const std::vector<uint8_t>& img_vec,
                           const uint8_t over_intensity_thresh,
                           const uint8_t under_intensity_thresh,
                           double* over_ratio,
                           double* under_ratio)
{
  CHECK_NOTNULL(over_ratio);
  CHECK_NOTNULL(under_ratio);
  size_t num_over = 0;
  size_t num_under = 0;
  for (size_t i = 0; i < img_vec.size(); i++)
  {
    if (img_vec[i] >= over_intensity_thresh)
    {
      num_over++;
    }
    else if (img_vec[i] <= under_intensity_thresh)
    {
      num_under++;
    }
  }
  (*over_ratio) = num_over / (1.0 * img_vec.size());
  (*under_ratio) = num_under / (1.0 * img_vec.size());
}

void computeSortedDGradientDExp(
    const cv::Mat& img,
    const int exp_us,
    const photometric_camera::PhotometricModel::Ptr& rm,
    std::vector<double>* sorted_derivs_vec,
    std::vector<double>* sorted_gradient_vec,
    std::vector<uint8_t>* sorted_img_vec)
{
  CHECK(rm);
  CHECK_NOTNULL(sorted_derivs_vec);
  size_t px_cnt = static_cast<size_t>(img.cols * img.rows);
  sorted_derivs_vec->resize(px_cnt);

  vk::Timer timer;
  timer.start();
  // compute image gradients
  cv::Mat DImgDx, DImgDy;
  auto_exposure_utils::computeScharr(img, &DImgDx, &DImgDy);
  cv::Mat gradient;
  auto_exposure_utils::computeGradient(DImgDx, DImgDy, &gradient);

  VLOG(60) << "Compute image gradients took " << timer.stop() * 1000 << " ms";
  timer.start();

  // compute DImgDExposureMs
  cv::Mat DImgDExpMS;
  rm->computeDImageDExposureMS(img, exp_us / 1000.0, &DImgDExpMS);
  cv::Mat DImgDExpMS_x, DImgDExpMS_y;
  auto_exposure_utils::computeScharr(DImgDExpMS, &DImgDExpMS_x, &DImgDExpMS_y);

  VLOG(60) << "Compute DImgDExposure took " << timer.stop() * 1000 << " ms";
  timer.start();

  // vectorize
  std::vector<double> DImgDx_vec, DImgDy_vec, DImgDExpMS_x_vec,
      DImgDExpMS_y_vec;
  std::vector<double> gradient_vec;
  std::vector<uint8_t> img_vec;
  auto_exposure_utils::cvMatToVector(img, &img_vec);
  //  auto_exposure_utils::cvMatToVector(DImgDx, &DImgDx_vec);
  //  auto_exposure_utils::cvMatToVector(DImgDy, &DImgDy_vec);
  //  auto_exposure_utils::cvMatToVector(DImgDExpMS_x, &DImgDExpMS_x_vec);
  //  auto_exposure_utils::cvMatToVector(DImgDExpMS_y, &DImgDExpMS_y_vec);
  auto_exposure_utils::cvMatToVector(gradient, &gradient_vec);

  VLOG(60) << "Vectorization took " << timer.stop() * 1000 << " ms";
  timer.start();

  std::vector<double> raw_derivs(static_cast<size_t>(img.cols * img.rows));
  size_t cnt = 0;
  for (int ix = 0; ix < img.cols; ix++)
  {
    for (int iy = 0; iy < img.rows; iy++)
    {
      raw_derivs[cnt++] =
          2.0 * (DImgDx.at<double>(iy, ix) * DImgDExpMS_x.at<double>(iy, ix) +
                 DImgDy.at<double>(iy, ix) * DImgDExpMS_y.at<double>(iy, ix));
    }
  }
  //  for (size_t i = 0; i < raw_derivs.size(); i++)
  //  {
  //    raw_derivs[i] = 2.0 * (DImgDx_vec[i] * DImgDExpMS_x_vec[i] +
  //                           DImgDy_vec[i] * DImgDExpMS_y_vec[i]);
  //  }

  VLOG(60) << "Raw derivatives calculation took " << timer.stop() * 1000 << " m"
                                                                            "s";
  timer.start();

  // sort the derivatives and pixel values by gradient
  std::vector<size_t> permutation;
  auto_exposure_utils::getPermutation(gradient_vec, &permutation);

  VLOG(60) << "Calculate Permutations took " << timer.stop() * 1000 << " ms";
  timer.start();

  auto_exposure_utils::applyPemutation(
      permutation, raw_derivs, sorted_derivs_vec);
  if (sorted_img_vec)
  {
    sorted_img_vec->resize(px_cnt);
    auto_exposure_utils::applyPemutation(permutation, img_vec, sorted_img_vec);
  }
  if (sorted_gradient_vec)
  {
    sorted_gradient_vec->resize(px_cnt);
    auto_exposure_utils::applyPemutation(
        permutation, gradient_vec, sorted_gradient_vec);
  }

  VLOG(60) << "Apply Permutations took " << timer.stop() * 1000 << " ms";
  timer.start();
}

void overUnderCompensateDerivs(const std::vector<double>& DGradientDExp_vec,
                               const std::vector<uint8_t>& img_vec,
                               const OverUnderExposureCompOptions& options,
                               std::vector<double>* DGradientDExp_comp_vec)
{
  CHECK_EQ(DGradientDExp_vec.size(), img_vec.size());
  size_t px_cnt = img_vec.size();
  CHECK_NOTNULL(DGradientDExp_comp_vec);
  DGradientDExp_comp_vec->resize(px_cnt);

  double ratio_over, ratio_under;
  auto_exposure_utils::overUnderExposedRatio(img_vec,
                                             options.over_intensity_thresh,
                                             options.under_intensity_thresh,
                                             &ratio_over,
                                             &ratio_under);
  VLOG(60) << "Overexposed ratio: " << ratio_over
           << ", underexposed ratio: " << ratio_under;

  if (ratio_over > options.overexposure_active_thresh)
  {
    VLOG(60) << "Going to compensate for overexposed pixels";
  }
  if (ratio_under > options.underexposure_active_thresh)
  {
    VLOG(60) << "Going to compensate for underexposed pixels";
  }

  size_t num_over_compen = 0;
  size_t num_under_compen = 0;
  for (size_t i = 0; i < px_cnt; i++)
  {
    double raw_deriv_i = DGradientDExp_vec[i];

    if (ratio_over > options.overexposure_active_thresh &&
        img_vec[i] >= options.over_intensity_thresh)
    {
      if (std::fabs(raw_deriv_i) < options.overexposure_compensation_g_thresh)
      {
        raw_deriv_i = options.overexposure_compensation_factor *
                      options.overexposure_compensation_g_thresh;
      }
      else
      {
        raw_deriv_i =
            std::fabs(raw_deriv_i) * options.overexposure_compensation_factor;
      }
      num_over_compen++;
    }

    if (ratio_under > options.underexposure_active_thresh &&
        img_vec[i] <= options.under_intensity_thresh)
    {
      if (std::fabs(raw_deriv_i) < options.underexposure_compensation_g_thresh)
      {
        raw_deriv_i = options.underexposure_compensation_factor *
                      options.underexposure_compensation_g_thresh;
      }
      else
      {
        raw_deriv_i =
            std::fabs(raw_deriv_i) * options.underexposure_compensation_factor;
      }
      num_under_compen++;
    }
    (*DGradientDExp_comp_vec)[i] = raw_deriv_i;
  }
  VLOG(60) << num_over_compen << " pixels are compensated for overexposure; "
           << num_under_compen << " pixels for underexposure.";
}
}
