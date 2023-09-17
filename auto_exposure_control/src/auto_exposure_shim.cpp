#include "auto_exposure_control/auto_exposure_shim.h"

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

namespace auto_exposure
{
int AutoExposureShim::computeDesiredIntensityExposureShim(
    const cv::Mat& img, const int last_exp_time_us)
{
  // compute gamma corrected images
  const std::vector<float> gamma_vec = { 0.1f, 0.5f, 0.8f, 1.0f,
                                         1.2f, 1.5f, 1.9f };

  std::vector<cv::Mat> corrected_images;
  for (size_t g = 0; g < gamma_vec.size(); ++g)
  {
    corrected_images.push_back(cv::Mat(img.size(), img.type()));
  }
  std::vector<float> gradient_info(gamma_vec.size());
  const int stride = img.cols;
  for (size_t g = 0; g < gamma_vec.size(); ++g)
  {
    cv::Mat& corr_img = corrected_images.at(g);
    const float& gamma = gamma_vec.at(g);
    for (int y = 0; y < img.rows; ++y)
    {
      const uchar* in = img.ptr<uchar>(y);
      uchar* out = corr_img.ptr<uchar>(y);
      for (int x = 0; x < stride; ++x)
      {
        out[x] = static_cast<uchar>(
            std::pow(static_cast<float>(in[x]) / 255.0f, gamma) * 255.0f);
      }
    }
    gradient_info.at(g) =
        computeGradientInformationShim(corrected_images.at(g));
  }

  // visualize
  if (options_.shim_display)
  {
    cv::Mat display_img(img.rows, gamma_vec.size() * img.cols, img.type());
    for (size_t g = 0; g < gamma_vec.size(); ++g)
      corrected_images[g].copyTo(
          display_img(cv::Rect(g * img.cols, 0, img.cols, img.rows)));
    cv::imshow("gamma_imgs", display_img);
    cv::waitKey(10);
    cv::imshow("img", img);
    cv::waitKey(10);
  }

  // get best gradient info
  // TODO (zzc): fit a 5th order polynomial and find the best gamma
  size_t best_gamma_index = 0;
  float best_info = gradient_info.at(0);
  for (size_t i = 0; i < gamma_vec.size(); ++i)
  {
    if (gradient_info[i] > best_info)
    {
      best_info = gradient_info[i];
      best_gamma_index = i;
    }
  }
  float best_gamma = gamma_vec.at(best_gamma_index);
  VLOG(45) << "Best gamma is " << best_gamma << std::endl;

  // compute the desired exposure time
  float alpha = 0.5;
  if (best_gamma < 1.0)
  {
    alpha = 1.0;
  }
  float kp = 0.2;
  return last_exp_time_us * (1 + alpha * kp * (1 - best_gamma));
}

float AutoExposureShim::computeGradientInformationShim(const cv::Mat& img)
{
  // design variables:
  float delta = options_.shim_delta;
  float lambda = options_.shim_lambda;
  float N_inv = 1 / (std::log10(lambda * (1 - delta) + 1));

  // get gradients
  cv::Mat grad_x, grad_y;
  cv::GaussianBlur(img, img, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
  cv::Sobel(img, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
  cv::Sobel(img, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

  // calculate gradient magnitude
  cv::Size size = grad_x.size();
  cv::Mat grad_mag(size, CV_32F);
  if (grad_x.isContinuous() && grad_y.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }
  const float* gx, *gy;
  float* out;
  // max value out of a sobel filter is 4*255
  float max = sqrt(16 * 255 * 255 + 16 * 255 * 255);
  for (int i = 0; i < size.height; ++i)
  {
    gx = grad_x.ptr<float>(i);
    gy = grad_y.ptr<float>(i);
    out = grad_mag.ptr<float>(i);
    for (int j = 0; j < size.width; ++j)
    {
      out[j] = std::sqrt(gx[j] * gx[j] + gy[j] * gy[j]);
      if (out[j] > max)
      {
        LOG(WARNING) << "Gradient exceeds the maximum value. Clamped.";
        out[j] = max;
      }
    }
  }
  // normalise
  grad_mag /= max;

  // calculate total gradient information
  float M = 0.0;
  size = grad_mag.size();
  if (grad_mag.isContinuous())
  {
    size.width *= size.height;
    size.height = 1;
  }
  const float* mag;
  for (int i = 0; i < size.height; ++i)
  {
    mag = grad_mag.ptr<float>(i);
    for (int j = 0; j < size.width; ++j)
    {
      if (mag[j] >= delta)
      {
        M += N_inv * std::log10(lambda * (mag[j] - delta) + 1);
      }
    }
  }
  return M;
}

}
