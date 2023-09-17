#include "photometric_camera/photometric_model.h"

#include <iostream>
#include <iomanip>  // for set precision
#include <sstream>
#include <fstream>
#include <limits>

#ifdef USE_DOUBLE
using real_t = double;
#define CV_REAL_T CV_64F
#else
using real_t = float;
#define CV_REAL_T CV_32F
#endif

namespace photometric_camera
{
namespace internal
{
bool loadValuesFromFile(const std::string& filename,
                        std::array<float, 256>& values)
{
  std::ifstream fs(filename.c_str());
  if (!fs.is_open())
  {
    LOG(WARNING) << "Could not open file " << filename;
    return false;
  }

  size_t i = 0;
  for (; i < 256 && fs.good() && !fs.eof(); ++i)
  {
    if (fs.peek() == '#')  // skip comments
      fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    float val;
    fs >> val;
    values.at(i) = val;
  }
  if (i != 256)
  {
    LOG(WARNING) << "Could not open file " << filename;
    return false;
  }
  return true;
}

bool loadValuesFromFile(const std::string& filename, std::vector<float>& poly)
{
  poly.clear();
  std::ifstream fs(filename.c_str());
  if (!fs.is_open())
  {
    LOG(WARNING) << "Could not open file " << filename;
    return false;
  }

  size_t i = 0;
  for (; fs.good() && !fs.eof(); ++i)
  {
    if (fs.peek() == '#')  // skip comments
      fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    float val;
    fs >> val;
    if (!fs.good() || fs.eof())
      break;
    poly.push_back(val);
  }

  return true;
}

float evaluatePolynomia(const std::vector<float>& coeffs, const float value)
{
  float x_power = 1.0;  // x^0
  float sum = 0.0;
  size_t poly_len = coeffs.size();
  for (size_t i = 0; i < poly_len; i++)
  {
    sum += coeffs[poly_len - 1 - i] * x_power;
    x_power *= value;
  }

  return sum;
}

}  // namespace

PhotometricModel::Ptr PhotometricModel::loadModel(const std::string& data_dir,
                                                  const float gain_x)
{
  PhotometricModel::Ptr model = std::make_shared<PhotometricModel>();

  std::string filename;
  std::string gain_string;
  if (gain_x > 0)  // otherwise empty gain string
  {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << gain_x;
    gain_string = "_gain" + ss.str();
  }

  filename = data_dir + "/g_fitted" + gain_string + ".txt";
  if (!internal::loadValuesFromFile(filename, model->g_))
    return nullptr;

  filename = data_dir + "/g_fitted_deriv_1" + gain_string + ".txt";
  if (!internal::loadValuesFromFile(filename, model->g_deriv_1_))
    return nullptr;

  filename = data_dir + "/g_fitted_deriv_2" + gain_string + ".txt";
  if (!internal::loadValuesFromFile(filename, model->g_deriv_2_))
    return nullptr;

  filename = data_dir + "/f_fitted" + gain_string + ".txt";
  if (!internal::loadValuesFromFile(filename, model->f_))
    return nullptr;

  filename = data_dir + "/f_fitted_deriv_1" + gain_string + ".txt";
  if (!internal::loadValuesFromFile(filename, model->f_deriv_1_))
    return nullptr;

  model->max_exposure_ = std::exp(model->g(255));
  model->min_exposure_ = std::exp(model->g(0));

  VLOG(1) << "Radiometric Model from " << data_dir << " loaded.";

  return model;
}

float PhotometricModel::g_float(const float intensity) const
{
  CHECK_LE(intensity, 255.0);
  CHECK_GE(intensity, 0.0);

  uint8_t upper = std::ceil(intensity);
  uint8_t lower = std::floor(intensity);

  if (upper == lower)
  {
    return g(static_cast<uint8_t>(intensity));
  }

  float w_upper = intensity - static_cast<float>(lower);
  float w_lower = static_cast<float>(upper) - intensity;

  return w_upper * g(upper) + w_lower * g(lower);
}

float PhotometricModel::f(const float exposure) const
{
  // clamp the exposure
  // for exposure get out of range, the fitted f can go crazy
  float res = 0.0;
  if (exposure > max_exposure_)
  {
    res = internal::evaluatePolynomia(f_, max_exposure_);
  }
  else if (exposure < min_exposure_)
  {
    res = internal::evaluatePolynomia(f_, min_exposure_);
  }
  else
  {
    res = internal::evaluatePolynomia(f_, exposure);
  }

  // clamp the value
  // since f is a fitted polynomial,
  // the result doesn't necessarily stays within [0, 255]
  if (res > 255.0)
  {
    res = 255.0;
  }
  if (res < 0.0)
  {
    res = 0.0;
  }

  return res;
}

float PhotometricModel::fDeriv1(const float exposure) const
{
  if (exposure > max_exposure_ || exposure < min_exposure_)
    return 0.0;
  return internal::evaluatePolynomia(f_deriv_1_, exposure);
}

void PhotometricModel::getIrradiancePatch(const cv::Mat& img, double exp_t,
                                          const int ux, const int uy,
                                          const size_t half_patch_size,
                                          cv::Mat* irradiance_patch) const
{
  CHECK_NOTNULL(irradiance_patch);
  CHECK(!img.empty()) << "Given image is empty.";

  size_t patch_size = 2 * half_patch_size;
  irradiance_patch->create(patch_size, patch_size, CV_REAL_T);

  // boundary
  int low_x = ux - half_patch_size;
  int low_y = uy - half_patch_size;
  int high_x = ux + half_patch_size - 1;
  int high_y = uy + half_patch_size - 1;

  // check border
  if (low_x < 0 || low_y < 0 || high_x >= img.cols || high_y >= img.cols)
  {
    VLOG(10) << "Patch is too near to the border. Abort.";
    return;
  }

  // sample the exposure
  for (int dx = 0; dx < static_cast<int>(patch_size); dx++)
  {
    for (int dy = 0; dy < static_cast<int>(patch_size); dy++)
    {
      real_t ln_exp = static_cast<real_t>(
          g(static_cast<size_t>(img.at<u_int8_t>(low_y + dy, low_x + dx))));
      irradiance_patch->at<real_t>(dy, dx) = std::exp(ln_exp) / exp_t;
    }
  }
}

void PhotometricModel::getExposurePatch(const cv::Mat& img, const int ux,
                                        const int uy,
                                        const size_t half_patch_size,
                                        cv::Mat* exposure_patch) const
{
  CHECK_NOTNULL(exposure_patch);
  CHECK(!img.empty()) << "Given image is empty.";

  // boundary
  int low_x = ux - half_patch_size;
  int low_y = uy - half_patch_size;
  int high_x = ux + half_patch_size - 1;
  int high_y = uy + half_patch_size - 1;

  // check border
  if (low_x < 0 || low_y < 0 || high_x >= img.cols || high_y >= img.cols)
  {
    VLOG(10) << "Patch is too near to the border. Abort.";
    return;
  }

  size_t patch_size = 2 * half_patch_size;
  exposure_patch->create(patch_size, patch_size, CV_REAL_T);
  // sample the exposure
  for (int dx = 0; dx < static_cast<int>(patch_size); dx++)
  {
    for (int dy = 0; dy < static_cast<int>(patch_size); dy++)
    {
      real_t ln_exp = static_cast<real_t>(
          g(static_cast<size_t>(img.at<u_int8_t>(low_y + dy, low_x + dx))));
      exposure_patch->at<real_t>(dy, dx) = std::exp(ln_exp);
    }
  }
}

void PhotometricModel::getImageFromExposure(const cv::Mat& exposure_map,
                                            cv::Mat* img) const
{
  CHECK(!exposure_map.empty());
  CHECK_NOTNULL(img);

  img->create(exposure_map.rows, exposure_map.cols, CV_8UC1);

  for (int ix = 0; ix < exposure_map.cols; ix++)
  {
    for (int iy = 0; iy < exposure_map.rows; iy++)
    {
      img->at<u_int8_t>(iy, ix) = static_cast<u_int8_t>(
          std::round(this->f(exposure_map.at<real_t>(iy, ix))));
    }
  }
}

void PhotometricModel::getExposureFromImage(const cv::Mat& img,
                                            cv::Mat* exposure) const
{
  CHECK(!img.empty());
  CHECK_NOTNULL(exposure);

  exposure->create(img.rows, img.cols, CV_REAL_T);

  for (int ix = 0; ix < img.cols; ix++)
  {
    for (int iy = 0; iy < img.rows; iy++)
    {
      exposure->at<real_t>(iy, ix) = std::exp(g(img.at<u_int8_t>(iy, ix)));
    }
  }
}

void PhotometricModel::computeDImageDExposureMS(const cv::Mat& img,
                                                const double exp_time_ms,
                                                cv::Mat* DImgDExpMs) const
{
  CHECK(!img.empty());
  CHECK_NOTNULL(DImgDExpMs);

  DImgDExpMs->create(img.rows, img.cols, CV_REAL_T);

  for (int ix = 0; ix < img.cols; ix++)
  {
    for (int iy = 0; iy < img.rows; iy++)
    {
      DImgDExpMs->at<real_t>(iy, ix) =
          1.0 / (exp_time_ms * this->gDeriv1(img.at<u_int8_t>(iy, ix)));
    }
  }
}

void PhotometricModel::compensateExposure(const double from_exp_ms,
                                          const double to_exp_ms,
                                          const uint8_t from_intensity,
                                          uint8_t* to_intensity) const
{
  CHECK_NOTNULL(to_intensity);

  real_t ln_from_exposure =
      static_cast<real_t>(g(static_cast<size_t>(from_intensity)));
  real_t to_exposure = to_exp_ms * (std::exp(ln_from_exposure) / from_exp_ms);
  *to_intensity = static_cast<uint8_t>(std::round(f(to_exposure)));
}

void PhotometricModel::compensateExposureUs(const double from_exp_us,
                                            const double to_exp_us,
                                            const uint8_t from_intensity,
                                            uint8_t* to_intensity) const
{
  double from_exp_ms = from_exp_us / 1000.0;
  double to_exp_ms = to_exp_us / 1000.0;
  compensateExposure(from_exp_ms, to_exp_ms, from_intensity, to_intensity);
}

void PhotometricModel::predictImage(const double ref_exp_us,
                                    const cv::Mat& ref_img,
                                    const double predict_exp_us,
                                    cv::Mat* predict_img) const
{
  cv::Mat ref_exposure;
  this->getExposureFromImage(ref_img, &ref_exposure);
  cv::Mat cur_exposure;
  cur_exposure.create(ref_exposure.rows, ref_exposure.cols, CV_REAL_T);
  double scale_factor = predict_exp_us / ref_exp_us;
  for (int ix = 0; ix < cur_exposure.cols; ix++)
  {
    for (int iy = 0; iy < cur_exposure.rows; iy++)
    {
      cur_exposure.at<real_t>(iy, ix) =
          static_cast<real_t>(ref_exposure.at<real_t>(iy, ix) * scale_factor);
    }
  }
  this->getImageFromExposure(cur_exposure, predict_img);
}

}  // namespace photometric_camera
