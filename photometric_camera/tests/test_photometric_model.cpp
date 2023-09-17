#include "photometric_camera/photometric_model.h"

#include <string>
#include <iostream>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>

using namespace photometric_camera;

namespace
{
void testPhotometricModel()
{
  // test load the model
  std::cout << "testPhotometricModel" << std::endl;

  //  std::string data_dir =
  //      ros::package::getPath("radiometric_camera")+
  //      "/radiometric_calib/camera_25000742";
  std::string data_dir =
      ros::package::getPath("photometric_calib") + "/camera_25000742";
  PhotometricModel::Ptr photometric_model =
      PhotometricModel::loadModel(data_dir, 1.0);

  if (photometric_model)
  {
    std::cout << "radiometric model loaded" << std::endl;
    std::cout << "I\t  g\t  g'\t  g''" << std::endl;
    for (int i = 0; i < 256; ++i)
    {
      std::cout << i << "\t  " << photometric_model->g(i) << "\t  "
                << photometric_model->gDeriv1(i) << "\t  "
                << photometric_model->gDeriv2(i) << std::endl;
    }

    std::cout << "interpolation for float intensities:" << std::endl;
    std::cout << "I\t  g" << std::endl;
    for (int i = 0; i < 255; ++i)
    {
      for (int j = 0; j < 5; ++j)
      {
        std::cout << i + 0.2 * j << "\t"
                  << photometric_model->g_float(i + 0.2 * j) << std::endl;
      }
    }

    std::cout << "The polynomial camera response function:\n";
    for (size_t i = 0; i < photometric_model->fSize(); i++)
    {
      std::cout << photometric_model->fCoeff(i) << ", ";
    }
    std::cout << std::endl;

    std::cout << "The polynomial camera response function derivative:\n";
    for (size_t i = 0; i < photometric_model->fDerivSize(); i++)
    {
      std::cout << photometric_model->fDerivCoeff(i) << ", ";
    }
    std::cout << std::endl;

    std::cout << "The min & max exposure: " << photometric_model->min_exposure_
              << ", " << photometric_model->max_exposure_ << std::endl;

    std::cout << "test for exposure 1.5: f: " << photometric_model->f(1.5)
              << ", f_deriv: " << photometric_model->fDeriv1(1.5) << std::endl;

    // get irradiance map
    std::string img_file = ros::package::getPath("active_camera_control") +
                           "/scripts/data/blur_synthetic/ref_img.png";
    cv::Mat img = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
    cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
    cv::imshow("img", img);

    cv::Mat irradiance_patch;
    photometric_model->getIrradiancePatch(
        img, 0.004, 387, 111, 20, &irradiance_patch);

    double min, max;
    cv::minMaxIdx(irradiance_patch, &min, &max);
    std::cout << "\nmin & max" << min << ", " << max << std::endl;
    cv::Mat disp_irradiance;
    cv::convertScaleAbs(irradiance_patch, disp_irradiance, 255 / max);
    cv::namedWindow("Patch Irradiance", cv::WINDOW_AUTOSIZE);
    cv::imshow("Patch Irradiance", disp_irradiance);
    cv::waitKey();
  }
}

}  // namespace

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  testPhotometricModel();
  return 0;
}
