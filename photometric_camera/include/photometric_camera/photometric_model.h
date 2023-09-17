#pragma once

#include <string>
#include <array>
#include <memory>
#include <opencv2/core/core.hpp>

#include <glog/logging.h>

namespace photometric_camera
{
/// Radiometric Camera Calibration Object.
/// The exposure time used with this class should be of **ms**
class PhotometricModel
{
public:
  /// Private constructor. A new Radiometric model can only be created via the
  /// factory method makeRadiometricModel().
  PhotometricModel() = default;

  using Ptr = std::shared_ptr<PhotometricModel>;
  PhotometricModel(const PhotometricModel&) = delete;
  void operator=(const PhotometricModel&) = delete;

  ~PhotometricModel() = default;

  static PhotometricModel::Ptr loadModel(const std::string& data_dir,
                                         const float gain_x = -1.0);

  void getIrradiancePatch(const cv::Mat& img,
                          double exp_t,
                          const int ux,
                          const int uy,
                          const size_t half_patch_size,
                          cv::Mat* iiradiance) const;

  void getExposurePatch(const cv::Mat& img,
                        const int ux,
                        const int uy,
                        const size_t half_patch_size,
                        cv::Mat* exposure) const;

  void getImageFromExposure(const cv::Mat& exposure_map, cv::Mat* img) const;

  void getExposureFromImage(const cv::Mat& img, cv::Mat* exposure) const;

  void computeDImageDExposureMS(const cv::Mat& img,
                                const double exp_time_ms,
                                cv::Mat* DImgDExpMs) const;

  void predictImage(const double ref_exp_us,
                    const cv::Mat& ref_img,
                    const double predict_exp_us,
                    cv::Mat* predict_img) const;

  void compensateExposure(const double from_exp_ms,
                          const double to_exp_ms,
                          const uint8_t from_intensity,
                          uint8_t* to_intensity) const;

  void compensateExposureUs(const double from_exp_us,
                            const double to_exp_us,
                            const uint8_t from_intensity,
                            uint8_t* to_intensity) const;

  // vector version
  template <typename ValueType>
  void getExposureFromImageVec(const std::vector<uint8_t>& img,
                               std::vector<ValueType>* exposure) const
  {
    CHECK(!img.empty());
    CHECK_NOTNULL(exposure);
    exposure->resize(img.size());
    for (size_t i = 0; i < img.size(); i++)
    {
      (*exposure)[i] = static_cast<ValueType>(std::exp(g(img[i])));
    }
  }

  // just in case we will need more complicated model in future...
  int getExpForNewGain(const float from_gain_x,
                       const float to_gain_x,
                       const int from_exp_us)
  {
    return static_cast<int>(from_exp_us * (from_gain_x / to_gain_x));
  }

  /// Get log irradiant energy for measured intensity.
  inline const float& g(const size_t& intensity) const
  {
    return g_.at(intensity);
  }

  /// Get first derivative at g(intensity).
  inline const float& gDeriv1(const size_t& intensity) const
  {
    return g_deriv_1_.at(intensity);
  }

  /// Get second derivative at g(intensity).
  inline const float& gDeriv2(const size_t& intensity) const
  {
    return g_deriv_2_.at(intensity);
  }

  inline const float& fCoeff(const size_t ind) const { return f_[ind]; }

  inline size_t fSize() const { return f_.size(); }

  inline size_t fDerivSize() const { return f_deriv_1_.size(); }

  inline const float& fDerivCoeff(const size_t ind) const
  {
    return f_deriv_1_[ind];
  }

  // we can have float intensity in cases like interpolation
  float g_float(const float intensity) const;

  float f(const float exposure) const;

  float fDeriv1(const float exposure) const;

  float max_exposure_;
  float min_exposure_;

private:
  std::array<float, 256> g_;          ///< Radiometric calibration.
  std::array<float, 256> g_deriv_1_;  ///< First derivative of g.
  std::array<float, 256> g_deriv_2_;  ///< Second derivative of g.

  // polynomial model of the camera response function
  std::vector<float> f_;
  std::vector<float> f_deriv_1_;

  //
};

}  // namespace active_camera_control
