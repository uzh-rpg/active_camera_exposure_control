#include "auto_exposure_control/frame.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace utils
{
void createImgPyramid(const cv::Mat& img,
                      const size_t img_pyr_lvl,
                      std::vector<cv::Mat>* img_pyr)
{
  (*img_pyr)[0] = img;
  for(size_t i=1; i<img_pyr_lvl; ++i)
  {
    // NOTE: Size(width, height)
    cv::Size down_size((*img_pyr)[i-1].cols/2, (*img_pyr)[i-1].rows/2);
    (*img_pyr)[i] = cv::Mat(down_size, CV_8UC1);
    cv::pyrDown((*img_pyr)[i-1], (*img_pyr)[i], down_size);
  }
}
}

namespace auto_exposure
{
size_t Frame::frame_cnt_ = 0;
void Frame::createImgPyramid(const cv::Mat& img, const size_t img_pyr_lvl)
{
  CHECK(!img.empty());
  CHECK_GT(img_pyr_lvl, 0u);
  CHECK_GT(img.rows, 0);
  CHECK_GT(img.cols, 0);

  img_pyr.resize(img_pyr_lvl);

  if (img.type() == CV_8UC1)
  {
    utils::createImgPyramid(img, img_pyr_lvl, &img_pyr);
  }
  else if (img.type() == CV_8UC3)
  {
    cv::Mat gray_image;
    cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
    utils::createImgPyramid(gray_image, img_pyr_lvl, &img_pyr);
  }
}
}
