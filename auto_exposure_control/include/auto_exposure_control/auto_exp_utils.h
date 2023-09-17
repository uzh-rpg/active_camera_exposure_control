#pragma once

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include <glog/logging.h>

#include "photometric_camera/photometric_model.h"

namespace auto_exposure_utils
{
inline float gainDbToX(const float gain_db)
{
  return std::pow(10.0f, gain_db / 20.0f);
}

template <typename ValueType>
void getPermutation(const std::vector<ValueType>& vec,
                    std::vector<size_t>* permutation)
{
  CHECK_NOTNULL(permutation);
  permutation->resize(vec.size());
  std::iota(permutation->begin(), permutation->end(), 0);
  std::sort(permutation->begin(),
            permutation->end(),
            [&vec](size_t i, size_t j)
            {
              return vec[i] < vec[j];
            });
}

template <typename ValueType>
void applyPemutation(const std::vector<size_t>& permutation,
                     const std::vector<ValueType>& vec,
                     std::vector<ValueType>* sorted_vec)
{
  CHECK_NOTNULL(sorted_vec);
  CHECK_EQ(permutation.size(), vec.size());
  sorted_vec->resize(vec.size());
  std::transform(permutation.begin(),
                 permutation.end(),
                 sorted_vec->begin(),
                 [&vec](size_t i)
                 {
                   return vec[i];
                 });
}

template <typename ValueType>
void cvMatToVector(const cv::Mat& mat, std::vector<ValueType>* vec)
{
  CHECK(!mat.empty());
  CHECK_NOTNULL(vec);

  vec->resize(mat.rows * mat.cols);
  size_t cnt = 0;
  for (int ix = 0; ix < mat.cols; ix++)
  {
    for (int iy = 0; iy < mat.rows; iy++)
    {
      (*vec)[cnt++] = mat.at<ValueType>(iy, ix);
    }
  }
}

template <typename ValueType>
ValueType weightedSum(const std::vector<ValueType>& values,
                      const std::vector<ValueType>& weights)
{
  CHECK_EQ(values.size(), weights.size());
  ValueType prod_sum(0.0);
  for (size_t i = 0; i < values.size(); i++)
  {
    prod_sum += values[i] * weights[i];
  }
  return prod_sum;
}

struct Logger
{
  void add(const std::string& name)
  {
    index.insert(std::pair<std::string, size_t>(name, size));
    names.push_back(name);
    size++;
    data.push_back(std::vector<double>());
  }

  void log(const std::string& name, const double value)
  {
    auto it = index.find(name);
    CHECK(it != index.end());

    std::vector<double>& single_data = data[it->second];
    single_data.push_back(value);
  }

  void checkConsistency() const
  {
    CHECK_EQ(index.size(), size);
    CHECK_EQ(names.size(), size);
    CHECK_EQ(data.size(), size);

    // whether all the log items have the same length
    for (size_t i = 0; i < size - 1; i++)
    {
      CHECK_EQ(data[i].size(), data[i + 1].size());
    }
  }

  void incValuesCnt() { log_cnt++; }

  size_t size = 0;
  std::vector<std::string> names;       // to keep the order
  std::map<std::string, size_t> index;  // need iterating for output
  std::vector<std::vector<double> > data;

  size_t log_cnt = 0;

  friend std::ostream& operator<<(std::ostream& os, const Logger& st);
};

void computeScharr(const cv::Mat& img, cv::Mat* img_dx, cv::Mat* img_dy);

void computeGradient(const cv::Mat& dx, const cv::Mat& dy, cv::Mat* gradient);

void createSineWeights(const size_t num,
                       const double order,
                       const double percentile_ratio,
                       std::vector<double>* weights);

void overUnderExposedRatio(const std::vector<uint8_t>& img_vec,
                           const uint8_t over_intensity_thresh,
                           const uint8_t under_intensity_thresh,
                           double* over_ratio,
                           double* under_ratio);

void computeSortedDGradientDExp(
    const cv::Mat& img,
    const int exp_us,
    const photometric_camera::PhotometricModel::Ptr& rm,
    std::vector<double>* sorted_derivs_vec,
    std::vector<double>* sorted_gradient_vec,
    std::vector<uint8_t>* sorted_img_vec);

struct OverUnderExposureCompOptions
{
  // overexposure compensation
  double overexposure_compensation_factor = -3.0;
  double overexposure_compensation_g_thresh = 0.5;
  double overexposure_active_thresh = 0.0;
  uint8_t over_intensity_thresh = 252u;
  // underexposure compensation
  double underexposure_compensation_factor = 3.0;
  double underexposure_compensation_g_thresh = 0.8;
  double underexposure_active_thresh = 1.0;
  uint8_t under_intensity_thresh = 10u;
};
void overUnderCompensateDerivs(const std::vector<double>& DGradientDExp_vec,
                               const std::vector<uint8_t>& img_vec,
                               const OverUnderExposureCompOptions& options,
                               std::vector<double>* DGradientDExp_comp_vec);
}

namespace ae_utils = auto_exposure_utils;
