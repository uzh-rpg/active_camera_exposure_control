#include "photometric_camera/photometric_dataset.h"

#include <limits>

#include <glog/logging.h>

namespace photometric_camera
{
void PhotometricDataset::init()
{
  std::ifstream img_fs(dataset_dir_ + "/images.txt");
  if (!img_fs.is_open())
  {
    LOG(FATAL) << "Fail to read the image list file" << dataset_dir_ + "/images"
                                                                       ".txt";
  }

  std::vector<int> ids;
  std::vector<double> gains_db;
  std::vector<double> exposures_ms;

  while (img_fs.good() && !img_fs.eof())
  {
    // skip comment
    while (img_fs.peek() == '#')
    {
      img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    int img_id;
    double gain_db;
    double exp_ms;
    img_fs >> img_id >> gain_db >> exp_ms;
    img_fs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    ids.emplace_back(img_id);
    gains_db.emplace_back(gain_db);
    exposures_ms.emplace_back(exp_ms);
  }

  // assumes that gain increase monotonically
  size_t num_img = ids.size();
  double pre_gain_db = -2;
  for (size_t i = 0; i < num_img; i++)
  {
    // if it is a new gain, create a new sub dataset
    double cur_gain_db = gains_db[i];
    if (std::fabs(cur_gain_db - pre_gain_db) > 1)
    {
      ExposureDataset exp_dataset;
      dataset_.gains_db.emplace_back(cur_gain_db);
      dataset_.exp_datasets.emplace_back(exp_dataset);
      pre_gain_db = cur_gain_db;
    }

    ExposureDataset& cur_exp_dataset = dataset_.exp_datasets.back();
    cur_exp_dataset.exp_time_ms.emplace_back(exposures_ms[i]);
    cur_exp_dataset.img_ids.emplace_back(ids[i]);
  }
}

void PhotometricDataset::readImage(double gain_db, double exp_ms,
                                   cv::Mat* img) const
{
  CHECK_NOTNULL(img);

  auto gain_iter =
      std::find_if(dataset_.gains_db.begin(), dataset_.gains_db.end(),
                   [gain_db](const double& gain)
                   {
                     return std::fabs(gain_db - gain) < 0.5;
                   });
  CHECK(gain_iter != dataset_.gains_db.end());

  const ExposureDataset& cur_exp_data =
      dataset_.exp_datasets[gain_iter - dataset_.gains_db.begin()];

  auto exp_iter =
      std::find_if(cur_exp_data.exp_time_ms.begin(),
                   cur_exp_data.exp_time_ms.end(), [exp_ms](const double& e)
                   {
                     return std::fabs(exp_ms - e) < 0.05;
                   });
  CHECK(exp_iter != cur_exp_data.exp_time_ms.end());

  int img_id =
      cur_exp_data.img_ids[exp_iter - cur_exp_data.exp_time_ms.begin()];

  readImage(img_id, img);
}

size_t PhotometricDataset::readImageClosestExpMS(double gain_db, double exp_ms,
                                                 cv::Mat* img) const
{
  size_t closest_idx = this->getClosestIndexExpMS(exp_ms);
  std::vector<double> exp_list;
  this->getExposureList(&exp_list);
  readImage(gain_db, exp_list[closest_idx], img);

  return closest_idx;
}

size_t PhotometricDataset::getClosestIndexExpMS(double exp_ms) const
{
  std::vector<double> exp_list;
  this->getExposureList(&exp_list);
  std::for_each(exp_list.begin(), exp_list.end(), [&exp_ms](double& e)
                {
                  e = std::fabs(e - exp_ms);
                });
  return std::min_element(exp_list.begin(), exp_list.end()) - exp_list.begin();
}

void PhotometricDataset::getGainList(std::vector<double>* gain_dbs) const
{
  CHECK_NOTNULL(gain_dbs);
  gain_dbs->clear();
  gain_dbs->insert(gain_dbs->begin(), dataset_.gains_db.begin(),
                   dataset_.gains_db.end());
}

void PhotometricDataset::getExposureList(std::vector<double>* exp_ms) const
{
  CHECK_NOTNULL(exp_ms);
  exp_ms->clear();
  ExposureDataset sample = dataset_.exp_datasets[0];
  exp_ms->insert(exp_ms->begin(), sample.exp_time_ms.begin(),
                 sample.exp_time_ms.end());
}

void PhotometricDataset::readImage(int id, cv::Mat* img) const
{
  CHECK_NOTNULL(img);
  std::string img_name = dataset_dir_ + "/image_" + std::to_string(id) + ".png";

  (*img) = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
}
}
