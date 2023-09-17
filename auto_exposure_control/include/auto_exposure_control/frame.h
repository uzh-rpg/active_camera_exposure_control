#pragma once

#include <memory>

#include <opencv2/opencv.hpp>

namespace auto_exposure
{
class Frame
{
public:
  Frame(const int exp_time_us, const float gain_x, const cv::Mat& img,
        const size_t img_pyr_lvl = 3)
    : exp_time_us(exp_time_us), gain_x(gain_x), id(frame_cnt_++)
  {
    createImgPyramid(img, img_pyr_lvl);
  }

  int exp_time_us;
  float gain_x;
  std::vector<cv::Mat> img_pyr;
  size_t id;

private:
  void createImgPyramid(const cv::Mat& img, const size_t img_pyr_lvl);
  static size_t frame_cnt_;
};
using FramePtr = std::shared_ptr<Frame>;

}
