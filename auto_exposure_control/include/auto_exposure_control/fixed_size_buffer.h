#pragma once

#include <vector>
#include <algorithm>
#include <glog/logging.h>

namespace auto_exposure_utils
{
template <typename T>
class FixedSizeBuffer
{
public:
  FixedSizeBuffer(size_t fixed_size)
    : fixed_size_(fixed_size)
    , vec_(fixed_size)
  {
  }

  FixedSizeBuffer() = delete;
  FixedSizeBuffer(const FixedSizeBuffer&) = delete;
  void operator=(const FixedSizeBuffer&) = delete;

  ~FixedSizeBuffer() {}

  void add(const T v)
  {
    vec_[write_idx_] = v;
    write_idx_ = wrapInc(write_idx_);

    write_cnt_ ++;

    if (write_cnt_ == fixed_size_)
    {
      is_filled_ = true;
    }
  }

  T get(const size_t i) const
  {
    CHECK_LT(i, fixed_size_);
    size_t read_idx = write_idx_ + i;
    if (read_idx >= fixed_size_)
    {
      read_idx = read_idx - fixed_size_;
    }
    return vec_[read_idx];
  }

  bool isFilled() const { return is_filled_; }
  size_t size() const { return fixed_size_; }
  size_t cnt() const { return write_cnt_; }


  T mean() const
  {
    CHECK(is_filled_);
    T sum (0.0);
    for (size_t i = 0; i < vec_.size(); i++)
    {
      sum += vec_[i];
    }

    return sum / vec_.size();
  }

  T stdev() const
  {
    CHECK(is_filled_);
    T m = mean();
    T accum = 0.0;
    std::for_each (std::begin(vec_), std::end(vec_), [&](const T d) {
        accum += (d - m) * (d - m);
    });

    return std::sqrt(accum / (vec_.size()-1));
  }

private:
  size_t wrapInc(const size_t idx)
  {
    size_t next_idx = idx + 1;
    if (next_idx >= fixed_size_)
    {
      next_idx = next_idx - fixed_size_;
    }
    return next_idx;
  }
  const size_t fixed_size_;
  std::vector<T> vec_;

  size_t write_idx_ = 0u;
  size_t write_cnt_ = 0u;  // used for filling the buffer first time
  bool is_filled_ = false;
};
}

