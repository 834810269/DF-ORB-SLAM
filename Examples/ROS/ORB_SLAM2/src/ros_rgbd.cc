/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

// std
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>

// ros
#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

// ros message_filters
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// opencv
#include<opencv2/core/core.hpp>

// SLAM
#include "../../../include/System.h"
#include "../../../include/Converter.h"

using namespace std;

// Publisher
ros::Publisher CamPose_Pub;
ros::Publisher path_Pub;

// messages
nav_msgs::Path Cam_path;
geometry_msgs::PoseStamped Cam_Pose;

// global var
cv::Mat Tcw;

// tf
tf::Transform orbslam;
tf::TransformBroadcaster* orb_slam_broadcaster;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM){}

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

    ORB_SLAM2::System* mpSLAM;
};

void PubCameraPose(const cv::Mat &Tcw);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }

    //ros::NodeHandle nh;
    //ros::NodeHandle private_nh("~");
    //tf::TransformListener listener;

    // get parameters
    //bool use_rviz;
    //private_nh.param("use_rviz", use_rviz, false);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD, true, false);

    // publish necessary message of SLAM
    //ORB_SLAM2_DENSE::MessageUtils msgUtils(listener, &SLAM);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "camera/depth_registered/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    CamPose_Pub = nh.advertise<geometry_msgs::PoseStamped>("/DF_ORB_SLAM/Camera_Pose",1);
    path_Pub = nh.advertise<nav_msgs::Path>("/DF_ORB_SLAM/path",10);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Tcw = mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());

    //PubCameraPose(Tcw);

}

void PubCameraPose(const cv::Mat &Tcw)
{
    orb_slam_broadcaster = new tf::TransformBroadcaster;
    cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
    cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

    vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc); // x, y, z, w

    orbslam.setOrigin(tf::Vector3(twc.at<float>(2), -twc.at<float>(0), -twc.at<float>(1)));
    orbslam.setRotation(tf::Quaternion(q[2], -q[0], -q[1], q[3]));
    orb_slam_broadcaster->sendTransform(tf::StampedTransform(orbslam, ros::Time::now(), "/map", "/base_link"));

    Cam_Pose.header.stamp = ros::Time::now();
    Cam_Pose.header.frame_id = "/map";
    tf::pointTFToMsg(orbslam.getOrigin(), Cam_Pose.pose.position);
    tf::quaternionTFToMsg(orbslam.getRotation(), Cam_Pose.pose.orientation);
    CamPose_Pub.publish(Cam_Pose); // geometry_msgs::PoseStamped

    Cam_path.header.frame_id = "/map";
    Cam_path.header.stamp = Cam_Pose.header.stamp;
    Cam_path.poses.push_back(Cam_Pose);
    path_Pub.publish(Cam_path); // nav_msgs::Path
}
