// Copyright (C) <2020> <Jiawei Mo, Junaed Sattar>

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <chrono>
#include <queue>

#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include "FullSystem/FullSystem.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "util/Undistort.h"

using namespace dso;

class VioNode {
private:
  int start_frame_;
  double td_cam_imu_;
  int incoming_id_;
  FullSystem *full_system_;
  Undistort *undistorter_;
  std::queue<Vec7> imu_queue_;
  std::queue<ImageAndExposure *> img_queue_;
  boost::mutex imu_queue_mutex_;
  boost::mutex img_queue_mutex_;

  void settingsDefault(int preset, int mode);

public:
  bool isLost;
  std::vector<double> frame_tt_;

  VioNode(int start_frame, double td_cam_imu, const std::string &calib,
          const std::string &vignette, const std::string &gamma, bool nomt,
          int preset, int mode);
  ~VioNode();

  void imuMessageCallback(const sensor_msgs::ImuConstPtr &msg);
  void imageMessageCallback(const sensor_msgs::ImageConstPtr &msg);
  void printResult(std::string file) { full_system_->printResult(file); }
};

void VioNode::settingsDefault(int preset, int mode) {
  printf("\n=============== PRESET Settings: ===============\n");
  if (preset == 1 || preset == 3) {
    printf("preset=%d is not supported", preset);
    exit(1);
  }
  if (preset == 0) {
    printf("DEFAULT settings:\n"
           "- 2000 active points\n"
           "- 5-7 active frames\n"
           "- 1-6 LM iteration each KF\n"
           "- original image resolution\n");

    setting_desiredImmatureDensity = 1500;
    setting_desiredPointDensity = 2000;
    setting_minFrames = 5;
    setting_maxFrames = 7;
    setting_maxOptIterations = 6;
    setting_minOptIterations = 1;
  }

  if (preset == 2) {
    printf("FAST settings:\n"
           "- 800 active points\n"
           "- 4-6 active frames\n"
           "- 1-4 LM iteration each KF\n"
           "- 424 x 320 image resolution\n");

    setting_desiredImmatureDensity = 600;
    setting_desiredPointDensity = 800;
    setting_minFrames = 4;
    setting_maxFrames = 6;
    setting_maxOptIterations = 4;
    setting_minOptIterations = 1;

    benchmarkSetting_width = 424;
    benchmarkSetting_height = 320;
  }

  if (mode == 0) {
    printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
  }
  if (mode == 1) {
    printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
  }
  if (mode == 2) {
    printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_minGradHistAdd = 3;
  }

  printf("==============================================\n");

  isLost = false;
}

VioNode::VioNode(int start_frame, double td_cam_imu, const std::string &calib,
                 const std::string &vignette, const std::string &gamma,
                 bool nomt, int preset, int mode)
    : start_frame_(start_frame), td_cam_imu_(td_cam_imu) {

  // DSO front end
  settingsDefault(preset, mode);

  multiThreading = !nomt;

  undistorter_ = Undistort::getUndistorterForFile(calib, gamma, vignette);

  setGlobalCalib((int)undistorter_->getSize()[0],
                 (int)undistorter_->getSize()[1],
                 undistorter_->getK().cast<float>());

  full_system_ = new FullSystem();
  if (undistorter_->photometricUndist != 0)
    full_system_->setGammaFunction(undistorter_->photometricUndist->getG());

  if (!disableAllDisplay) {
    IOWrap::PangolinDSOViewer *viewer =
        new IOWrap::PangolinDSOViewer(wG[0], hG[0], true);
    full_system_->outputWrapper.push_back(viewer);
  }

  incoming_id_ = 0;
}

VioNode::~VioNode() {
  delete undistorter_;
  for (auto &ow : full_system_->outputWrapper) {
    delete ow;
  }
  delete full_system_;
}

void VioNode::imuMessageCallback(const sensor_msgs::ImuConstPtr &msg) {
  boost::unique_lock<boost::mutex> lock(imu_queue_mutex_);
  Vec7 imu_data;
  imu_data[0] = msg->header.stamp.toSec() - td_cam_imu_;
  imu_data.segment<3>(1) << msg->linear_acceleration.x,
      msg->linear_acceleration.y, msg->linear_acceleration.z;
  imu_data.tail(3) << msg->angular_velocity.x, msg->angular_velocity.y,
      msg->angular_velocity.z;
  imu_queue_.push(imu_data);
}

void VioNode::imageMessageCallback(const sensor_msgs::ImageConstPtr &msg) {
  if (start_frame_ > 0) {
    start_frame_--;
    incoming_id_++;
    while (!imu_queue_.empty()) {
      imu_queue_.pop();
    }
    return;
  }

  boost::unique_lock<boost::mutex> img_lock(img_queue_mutex_);
  cv::Mat img;
  try {
    img = cv_bridge::toCvShare(msg, "mono8")->image;
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  MinimalImageB minImg((int)img.cols, (int)img.rows, (unsigned char *)img.data);
  ImageAndExposure *undistImg =
      undistorter_->undistort<unsigned char>(&minImg, 1, 0, 1.0f);
  undistImg->timestamp = msg->header.stamp.toSec();

  img_queue_.push(undistImg);

  boost::unique_lock<boost::mutex> imu_lock(imu_queue_mutex_);
  while (!imu_queue_.empty() && !img_queue_.empty() &&
         img_queue_.front()->timestamp < imu_queue_.back()[0]) {
    // current image pair
    ImageAndExposure *cur_img = img_queue_.front();
    img_queue_.pop();

    // get all imu data by current img timestamp
    std::vector<Vec7> cur_imu_data;
    while (imu_queue_.front()[0] < cur_img->timestamp) {
      cur_imu_data.push_back(imu_queue_.front());
      imu_queue_.pop();
    }
    assert(!imu_queue_.empty());

    if (!cur_imu_data.empty()) {
      // interpolate imu data at cur image time
      Vec7 last_imu_data =
          ((imu_queue_.front()[0] - cur_img->timestamp) * cur_imu_data.back() +
           (cur_img->timestamp - cur_imu_data.back()[0]) * imu_queue_.front()) /
          ((imu_queue_.front()[0] - cur_imu_data.back()[0]));
      last_imu_data[0] = cur_img->timestamp;
      cur_imu_data.push_back(last_imu_data);

      auto start = std::chrono::steady_clock::now();
      full_system_->addActiveFrame(cur_imu_data, cur_img, incoming_id_);
      auto end = std::chrono::steady_clock::now();
      frame_tt_.push_back(
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());

      // reinitialize if necessary
      if (full_system_->initFailed && incoming_id_ < 250) {
        std::vector<IOWrap::Output3DWrapper *> wraps =
            full_system_->outputWrapper;
        delete full_system_;

        printf("Reinitializing\n");
        full_system_ = new FullSystem();
        if (undistorter_->photometricUndist != 0)
          full_system_->setGammaFunction(
              undistorter_->photometricUndist->getG());
        full_system_->outputWrapper = wraps;
        // setting_fullResetRequested=false;
      }
    }

    delete cur_img;
    incoming_id_++;

    if (full_system_->isLost) {
      printf("LOST!!\n");
      isLost = true;
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "spline_vio");
  ros::NodeHandle nhPriv("~");

  /* *********************** required parameters ************************ */
  // stereo camera parameters
  std::vector<double> tfm_imu;
  double imu_rate, imu_acc_nd, imu_acc_rw, imu_gyro_nd, imu_gyro_rw;
  std::string imu_topic, cam_topic, calib, results_path;
  if (!nhPriv.getParam("T_imu/data", tfm_imu) ||
      !nhPriv.getParam("rate_hz", imu_rate) ||
      !nhPriv.getParam("accelerometer_noise_density", imu_acc_nd) ||
      !nhPriv.getParam("accelerometer_random_walk", imu_acc_rw) ||
      !nhPriv.getParam("gyroscope_noise_density", imu_gyro_nd) ||
      !nhPriv.getParam("gyroscope_random_walk", imu_gyro_rw) ||
      !nhPriv.getParam("imu_topic", imu_topic) ||
      !nhPriv.getParam("cam_topic", cam_topic) ||
      !nhPriv.getParam("calib", calib) ||
      !nhPriv.getParam("results", results_path)) {
    ROS_INFO("Fail to get sensor topics/params, exit.!!!!");
    return -1;
  }

  /* *********************** optional parameters ************************ */
  // DSO settings
  bool nomt;
  int preset, mode;
  std::string vignette, gamma;
  nhPriv.param("quiet", setting_debugout_runquiet, true);
  nhPriv.param("nogui", disableAllDisplay, false);
  nhPriv.param("nomt", nomt, false);
  nhPriv.param("preset", preset, 0);
  nhPriv.param("mode", mode, 1);
  nhPriv.param<std::string>("vignette", vignette, "");
  nhPriv.param<std::string>("gamma", gamma, "");

  double td_cam_imu;
  nhPriv.param("timeshift_cam_imu", td_cam_imu, 0.0);
  nhPriv.param("weight_imu_dso", setting_weight_imu_dso, 1.0);

  // read from a bag file
  std::string bag_path;
  int start_frame;
  nhPriv.param<std::string>("bag", bag_path, "");
  nhPriv.param("start_frame", start_frame, 0);

  /* ******************************************************************** */

  VioNode vio_node(start_frame, td_cam_imu, calib, vignette, gamma, nomt,
                   preset, mode);

  cv::Mat tfm_imu_cv = cv::Mat(tfm_imu);
  tfm_imu_cv = tfm_imu_cv.reshape(0, 4);
  Mat44 tfm_imu_cam;
  cv::cv2eigen(tfm_imu_cv, tfm_imu_cam);
  setting_rot_imu_cam = tfm_imu_cam.topLeftCorner<3, 3>();

  setting_weight_imu = Mat66::Identity();
  setting_weight_imu.topLeftCorner<3, 3>() /=
      (imu_acc_nd * imu_acc_nd * imu_rate);
  setting_weight_imu.bottomRightCorner<3, 3>() /=
      (imu_gyro_nd * imu_gyro_nd * imu_rate);
  setting_weight_imu *= setting_weight_imu_dso;

  setting_weight_imu_bias = Mat66::Identity();
  setting_weight_imu_bias.topLeftCorner<3, 3>() /= (imu_acc_rw * imu_acc_rw);
  setting_weight_imu_bias.bottomRightCorner<3, 3>() /=
      (imu_gyro_rw * imu_gyro_rw);
  setting_weight_imu_bias *= setting_weight_imu_dso;

  if (!bag_path.empty()) {

    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    std::vector<std::string> topics = {imu_topic, cam_topic};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    sensor_msgs::ImageConstPtr img;
    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      if (vio_node.isLost) {
        break;
      }
      if (m.getTopic() == imu_topic) {
        sensor_msgs::ImuConstPtr imu = m.instantiate<sensor_msgs::Imu>();
        vio_node.imuMessageCallback(imu);
      }
      if (m.getTopic() == cam_topic) {
        img = m.instantiate<sensor_msgs::Image>();
        vio_node.imageMessageCallback(img);
      }
    }
    bag.close();
  } else {
    ros::NodeHandle nh;

    // ROS subscribe to imu data
    ros::Subscriber imu_sub =
        nh.subscribe(imu_topic, 10000, &VioNode::imuMessageCallback, &vio_node);

    // ROS subscribe to images
    ros::Subscriber img_sub = nh.subscribe(
        cam_topic, 10000, &VioNode::imageMessageCallback, &vio_node);

    ros::spin();
  }

  vio_node.printResult(results_path);

  int total_frame_tt = 0;
  for (int tt : vio_node.frame_tt_) {
    total_frame_tt += tt;
  }
  printf("frame_tt: %.1f\n", float(total_frame_tt) / vio_node.frame_tt_.size());

  ros::spinOnce();
  return 0;
}
