#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace photometric_camera
{
typedef struct
{
  std::vector<double> exp_time_ms;
  std::vector<int> img_ids;
} ExposureDataset;

typedef struct
{
  std::vector<double> gains_db;
  std::vector<ExposureDataset> exp_datasets;
} GainExposureDatasets;

class PhotometricDataset
{
public:
  PhotometricDataset(const std::string& dataset_dir) : dataset_dir_(dataset_dir)
  {
    init();
  }

  void readImage(double gain_db, double exp_ms, cv::Mat* img) const;

  size_t readImageClosestExpMS(double gain_db, double exp_ms,
                               cv::Mat* img) const;

  size_t getClosestIndexExpMS(double exp_ms) const;

  void getGainList(std::vector<double>* gain_dbs) const;
  void getExposureList(std::vector<double>* exp_ms) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const PhotometricDataset& dataset);

private:
  // parsing images.txt
  void init();
  void readImage(int id, cv::Mat* img) const;

  std::string dataset_dir_;
  GainExposureDatasets dataset_;
};

std::ostream& operator<<(std::ostream& os, const PhotometricDataset& d)
{
  os << "dataset directory: " << d.dataset_dir_ << std::endl;

  size_t gain_num = d.dataset_.gains_db.size();

  os << "loaded " << gain_num << " different gains." << std::endl;

  for (size_t i = 0; i < gain_num; i++)
  {
    os << "Gain (db): " << d.dataset_.gains_db[i] << std::endl;
    os << "=================================\n";
    os << "Exp\t ID\n";
    const ExposureDataset& cur_exp_dataset = d.dataset_.exp_datasets[i];
    for (size_t j = 0; j < cur_exp_dataset.exp_time_ms.size(); j++)
    {
      os << cur_exp_dataset.exp_time_ms[j] << "\t" << cur_exp_dataset.img_ids[j]
         << "\n";
    }
  }

  return os;
}

}  // namespace photometric_camera
