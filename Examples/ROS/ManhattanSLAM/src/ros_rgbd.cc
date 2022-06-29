/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>

#include<opencv2/core/core.hpp>

#include"../../../include/System.h"
#include"../include/ImuTypes.h"

using namespace std;

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM, ImuGrabber *pImuGb): mpSLAM(pSLAM), mpImuGb(pImuGb){}

    void GrabImageRGB(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageDepth(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgRGBBuf, imgDepthBuf;
    std::mutex mBufMutexRGB,mBufMutexDepth;
   
    ORB_SLAM2::System* mpSLAM;
    ImuGrabber *mpImuGb;
};



int main(int argc, char **argv)
{
  ros::init(argc, argv, "RGBD");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  if(argc != 3)
  {
    cerr << endl << "Usage: rosrun ManhattanSLAM RGBD path_to_vocabulary path_to_settings" << endl;
    ros::shutdown();
    return 1;
  }

  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::IMU_RGBD,true);
  ORB_SLAM2::Config::SetParameterFile(argv[2]);

  ImuGrabber imugb;
  ImageGrabber igb(&SLAM,&imugb);

  // Maximum delay, 5 seconds
  ros::Subscriber sub_imu = n.subscribe("/d400/imu0", 4000, &ImuGrabber::GrabImu, &imugb);
  ros::Subscriber sub_img_rgb = n.subscribe("/d400/color/image_raw", 300, &ImageGrabber::GrabImageRGB,&igb);
  ros::Subscriber sub_img_depth = n.subscribe("/d400/aligned_depth_to_color/image_raw", 300, &ImageGrabber::GrabImageDepth,&igb);

  // ros::Subscriber sub_imu = n.subscribe("/imu", 1000, &ImuGrabber::GrabImu, &imugb);
  // ros::Subscriber sub_img_rgb = n.subscribe("/cam0/color", 300, &ImageGrabber::GrabImageRGB,&igb);
  // ros::Subscriber sub_img_depth = n.subscribe("/cam0/depth", 300, &ImageGrabber::GrabImageDepth,&igb);

  // message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/cam0/color", 1);
  // message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/cam0/depth", 1);
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

  // message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
  // sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

  std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);

  ros::spin();

  SLAM.Shutdown();

    // Save camera trajectory
  SLAM.SaveTrajectoryTUM("CameraTrajectory_opt.txt");
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

  ros::shutdown();

  return 0;
}



void ImageGrabber::GrabImageRGB(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexRGB.lock();
  if (!imgRGBBuf.empty())
    imgRGBBuf.pop();
  imgRGBBuf.push(img_msg);
  mBufMutexRGB.unlock();
}

void ImageGrabber::GrabImageDepth(const sensor_msgs::ImageConstPtr &img_msg)
{
  mBufMutexDepth.lock();
  if (!imgDepthBuf.empty())
    imgDepthBuf.pop();
  imgDepthBuf.push(img_msg);
  mBufMutexDepth.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    // if(cv_ptr->image.type()==0)
    // {
    return cv_ptr->image.clone();
    // }
    // else
    // {
    //   std::cout << "Error type" << std::endl;
    //   return cv_ptr->image.clone();
    // }
}

void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    static Eigen::Matrix4d gyro_calib,acc_calib;

    acc_calib <<       1.0129998922348022,     1.6299745067954063e-02, -1.6567818820476532e-02, -2.3803437128663063e-02,
            9.0558832744136453e-04, 1.0179165601730347,     -8.2402275875210762e-03, -9.5768600702285767e-02,
            -2.2675324231386185e-02,6.7262286320328712e-03,  1.0164324045181274e+00,  2.4007377028465271e-01,
            0.000000,               0.000000 ,                0.000000 ,              1.000000 ;

    gyro_calib <<      1.000000,            0.000000,            0.000000,         -6.7636385210789740e-05,
            0.000000,            1.000000,            0.000000,         -9.5424675237154588e-06,
            0.000000,            0.000000,            1.000000,         -1.7504280549474061e-05,
            0.000000,            0.000000,            0.000000,         1.000000000000000000000;

    cout << "Gyro instric matrix:" <<endl;
    cout << gyro_calib << endl;
    cout << "Accel instric matrix:" << endl;
    cout << acc_calib << endl;
  while(1)
  {
    cv::Mat imRGB, imDepth;
    double tImRGB = 0, tImDepth = 0;
    if (!imgRGBBuf.empty()&&!imgDepthBuf.empty()&&!mpImuGb->imuBuf.empty())
    {
      tImRGB = imgRGBBuf.front()->header.stamp.toSec();
      tImDepth = imgDepthBuf.front()->header.stamp.toSec();

      this->mBufMutexDepth.lock();
      while((tImRGB-tImDepth)>maxTimeDiff && imgDepthBuf.size()>1)
      {
        imgDepthBuf.pop();
        tImDepth = imgDepthBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexDepth.unlock();

      this->mBufMutexRGB.lock();
      while((tImDepth-tImRGB)>maxTimeDiff && imgRGBBuf.size()>1)
      {
        imgRGBBuf.pop();
        tImRGB = imgRGBBuf.front()->header.stamp.toSec();
      }
      this->mBufMutexRGB.unlock();

      if((tImRGB-tImDepth)>maxTimeDiff || (tImDepth-tImRGB)>maxTimeDiff)
      {
        // std::cout << "big time difference" << std::endl;
        continue;
      }
      if(tImRGB>mpImuGb->imuBuf.back()->header.stamp.toSec())
          continue;

      this->mBufMutexRGB.lock();
      imRGB = GetImage(imgRGBBuf.front());
      imgRGBBuf.pop();
      this->mBufMutexRGB.unlock();

      this->mBufMutexDepth.lock();
      imDepth = GetImage(imgDepthBuf.front());
      imgDepthBuf.pop();
      this->mBufMutexDepth.unlock();

      vector<ORB_SLAM2::IMU::Point> vImuMeas;
      mpImuGb->mBufMutex.lock();
      if(!mpImuGb->imuBuf.empty())
      {
        // Load imu measurements from buffer
        vImuMeas.clear();
        while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImRGB)
        {
          double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
          cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
          cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
          vImuMeas.push_back(ORB_SLAM2::IMU::Point(acc,gyr,t));

            // Eigen::Vector4d imu_gyro,imu_acc;

            // for(auto& imudata:vImuMeas){
            //     imu_gyro(0) = imudata.w.x;
            //     imu_gyro(1) = imudata.w.y;
            //     imu_gyro(2) = imudata.w.z;
            //     imu_gyro(3) = 1.0f;

            //     imu_acc(0) = imudata.a.x;
            //     imu_acc(1) = imudata.a.y;
            //     imu_acc(2) = imudata.a.z;
            //     imu_acc(3) = 1.0f;

            //     imu_gyro = gyro_calib * imu_gyro;
            //     imu_acc = acc_calib * imu_acc;

            //     imudata.a.x = imu_acc(0);
            //     imudata.a.y = imu_acc(1);
            //     imudata.a.z = imu_acc(2);

            //     imudata.w.x = imu_gyro(0);
            //     imudata.w.y = imu_gyro(1);
            //     imudata.w.z = imu_gyro(2);
            // }

          mpImuGb->imuBuf.pop();
        }
      }
      mpImuGb->mBufMutex.unlock();
      mpSLAM->TrackRGBD(imRGB,imDepth,tImRGB,vImuMeas);

      std::chrono::milliseconds tSleep(1);
      std::this_thread::sleep_for(tSleep);
    }
  }
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}


