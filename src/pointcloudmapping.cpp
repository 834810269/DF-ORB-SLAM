//
// Created by wang on 2020/12/17.
//

/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

// ORB_SLAM2
#include "pointcloudmapping.h"
#include "KeyFrame.h"
#include "Converter.h"
#include "PointCloudElement.h"
#include "System.h"
#include "TicToc.h"

// ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <pcl_ros/transforms.h>

// STL
#include <chrono>

bool firstKF = true;
int currentloopcount = 0;
ros::Publisher globalcloud_pub;
ros::Publisher curcloud_pub;
sensor_msgs::PointCloud2 global_points;
sensor_msgs::PointCloud2 cur_points;

PointCloudMapping::PointCloudMapping(const std::string &strSettingPath, bool bUseViewer) :
        mbCloudBusy(false), mbLoopBusy(false), mbStop(false), mbShutDownFlag(false),
        mpPclGlobalMap(new PointCloudMapping::PointCloud()), mbPointCloudMapUpdated(false),
        mbUseViewer(bUseViewer)
{
    // parse parameters
    cv::FileStorage fsSetting = cv::FileStorage(strSettingPath, cv::FileStorage::READ);

    // set initial Tbc: footprint<--optical
    Eigen::Vector3d tbc(0,0,0); //fsTbc["x"], fsTbc["y"], fsTbc["z"]);
    Eigen::Matrix3d Rbc;
    Rbc <<  0, 0, 1,
           -1, 0, 0,
            0,-1, 0;//Eigen::AngleAxisd(-M_PI/2,  Eigen::Vector3d::UnitX());
    updateTbc(Rbc, tbc);

    // voxel grid filter
    //resolution_ = fsPointCloudMapping["Resolution"];
    voxel.setLeafSize( resolution_, resolution_, resolution_);

    // statistical filter
    statistical_filter.setMeanK(meanK_);
    statistical_filter.setStddevMulThresh(meanK_);

    int h = fsSetting["Camera.height"];
    int w = fsSetting["Camera.width"];

    slic = new SLIC(h, w);
    // start point cloud mapping thread
    mThRunning = make_shared<thread>( bind(&PointCloudMapping::run, this ) );
}


void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        mbShutDownFlag = true;
        keyframeUpdated.notify_one();
    }
    mThRunning->join();
}


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth,int idk,vector<KeyFrame*> vpKFs, double th)
{
    cout<<"receive a keyframe, Frame id = "<< idk << " , KF No." << kf->mnId << endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    currentvpKFs = vpKFs;
    PointCloudE pointcloude;
    pointcloude.pcID = idk;
    pointcloude.T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    pointcloude.pcE = generatePointCloud(kf,color,depth, th); // 当前相机坐标系下点云
    pointcloud.push_back(pointcloude);
    keyframeUpdated.notify_one();
}


PointCloudMapping::PointCloud::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, double th)//,Eigen::Isometry3d T
{
    PointCloud::Ptr tmp( new PointCloud() );
    slic->generate_super_pixels(color, depth);

    // th
    //cv::Scalar mean;  //均值
    //cv::Scalar stddev;  //标准差

    //cv::meanStdDev( depth(cv::Rect(10,10,depth.cols-10,depth.rows-10)), mean, stddev );  //计算均值和标准差
    //double mean_pxl = mean.val[0];
    //double stddev_pxl = stddev.val[0];
    //double th = mean_pxl + stddev_pxl;

    // point cloud is null ptr
    int pt_cnt = 0;
    for ( int m=0; m<slic->superpixel_seeds.size(); m++ )
    {
        if(slic->superpixel_seeds[m].size<10) continue;
        float d = slic->superpixel_seeds[m].mean_depth;
        if (isnan(d) || d < 0.01 || d> th)
            continue;
        PointT p;
        p.z = d;
        p.x = ( slic->superpixel_seeds[m].x - kf->cx) * p.z / kf->fx;
        p.y = ( slic->superpixel_seeds[m].y - kf->cy) * p.z / kf->fy;

        p.b = slic->superpixel_seeds[m].B;
        p.g = slic->superpixel_seeds[m].G;
        p.r = slic->superpixel_seeds[m].R;

        tmp->points.push_back(p);
        pt_cnt++;
    }
    return tmp;
}


void PointCloudMapping::run()
{
    ros::NodeHandle n;
    globalcloud_pub = n.advertise<sensor_msgs::PointCloud2>("/DF_ORB_SLAM/GlobalPointCloud",1);
    curcloud_pub = n.advertise<sensor_msgs::PointCloud2>("/DF_ORB_SLAM/CurPointCloud",1);
    //pcl::visualization::CloudViewer viewer("point cloud map");
    while(true)
    {
        // shutdown request
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (mbShutDownFlag)
            {
                break;
            }
        }
        // wait for update (have generated pointcloud)
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdatedMutex );
            keyframeUpdated.wait( lck_keyframeUpdated );
        }
        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        // loop busy, or thread request stop
        if(mbLoopBusy || mbStop)
        {
            cerr << "Point Cloud Mapping thread is Looping or has terminated!" << endl;
            continue;
        }

        // no keyframe insert
        if(lastkeyframeSize == N)
            mbCloudBusy = false;
        mbCloudBusy = true;

        setPointCloudMapUpdatedFlag(false);
        cout << "*** Running PointCloudMapping thread. ***" << endl;

        // get extrinsic matrix
        Eigen::Matrix4d Tbc;
        getTbc(Tbc);
        Eigen::Matrix4d Tcb = Tbc.inverse();

        // create new PointCloud
        PointCloud::Ptr pNewCloud(new PointCloud());
        for ( size_t i=lastkeyframeSize; i<N ; i++ ) // for new keyframes
        {
            PointCloud::Ptr p (new PointCloud);
            pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());
            pcl::transformPointCloud( *p, *p, Tbc);
            *pNewCloud += *p;
        }

        pcl::toROSMsg(*pNewCloud, cur_points);
        cur_points.header.stamp = ros::Time::now();
        cur_points.header.frame_id = "/map";
        curcloud_pub.publish(cur_points);

        *pNewCloud += *mpPclGlobalMap;

        // depth filter and statistical removal
        PointCloud::Ptr pNewCloudOutliersFilter(new PointCloud());
        statistical_filter.setInputCloud(pNewCloud);
        statistical_filter.filter( *pNewCloudOutliersFilter );

        // voxel grid filter
        PointCloud::Ptr pNewCloudVoxelFilter(new PointCloud());
        voxel.setInputCloud( pNewCloudOutliersFilter );
        voxel.filter( *pNewCloudVoxelFilter );
        {
            unique_lock<mutex> lock(mMtxPointCloudUpdated);
            mpPclGlobalMap->swap(*pNewCloudVoxelFilter);
        }
        cout << "show global map, size=" << N << "   " << mpPclGlobalMap->points.size() << endl;


        // visualize, if needed
        //viewer.showCloud( mpPclGlobalMap );
        //cout<<"showing"<<endl;
        pcl::toROSMsg(*mpPclGlobalMap, global_points);
        global_points.header.stamp = ros::Time::now();
        global_points.header.frame_id = "/map";
        globalcloud_pub.publish(global_points);

        // update flag
        lastkeyframeSize = N;
        mbCloudBusy = false;
        setPointCloudMapUpdatedFlag(true);
    }
}


void PointCloudMapping::save()
{
    pcl::io::savePCDFile( "result.pcd", *mpPclGlobalMap );
    cout<<"globalMap save finished"<<endl;
}


void PointCloudMapping::updateCloseLoopCloud()
{
    while (mbCloudBusy) // run()在执行,阻塞当前
    {
        std::cout << "CloseLooping thread has activate point cloud map reconstruct, "
                     "but PointCloudMapping thread is busy currently." << std::endl;
        usleep(1000);
    }
    mbLoopBusy = true; // 阻塞run()
    std::cout << "******************* Start Loop Mapping *******************" << std::endl;

    // transform the whole point cloud according to extrinsic matrix
    Eigen::Matrix4d Tbc;
    getTbc(Tbc);
    Eigen::Matrix4d Tcb = Tbc.inverse();

    // reset new point cloud map
    PointCloud::Ptr pNewCloud(new PointCloud());
    PointCloud::Ptr pNewPlaneCloud(new PointCloud());
    cout << "Current KeyFrame size: " << currentvpKFs.size() << endl;
    cout << "Curremt PointCloude size: " << pointcloud.size() << endl;
    for (int i=0;i<currentvpKFs.size();i++)
    {
        for (int j=0;j<pointcloud.size();j++)
        {
            if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId) // 闭环后仍然存在的关键帧
            {
                cout << "Start dealing with KeyFrame [" << i << "]" << endl;
                Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );

                PointCloud::Ptr p (new PointCloud);
                pcl::transformPointCloud( *(pointcloud[j].pcE), *p, T.inverse().matrix());
                pcl::transformPointCloud( *p, *p, Tbc);
                *pNewCloud += *p;

                continue;
            }
        }
    }
    cout << "Gather all KeyFrame complete." << endl;


    // depth filter and statistical removal
    //! Prohibit it because it is too time-costly in updating close loop point cloud
    // PointCloud::Ptr pNewCloudOutliersFilter(new PointCloud());
    // statistical_filter.setInputCloud(pNewCloud);
    // statistical_filter.filter( *pNewCloudOutliersFilter );

    // voxel grid filter
    PointCloud::Ptr pNewCloudVoxelFilter(new PointCloud());
    voxel.setInputCloud(pNewCloud);
    voxel.filter( *pNewCloudVoxelFilter );
    {
        unique_lock<mutex> lock(mMtxPointCloudUpdated);
        mpPclGlobalMap->swap(*pNewCloudVoxelFilter);
    }

    // update flag
    mbLoopBusy = false;
    loopcount++;
    setPointCloudMapUpdatedFlag(true);

    std::cout << "******************* Finish Loop Mapping *******************" << std::endl;
}


void PointCloudMapping::setPointCloudMapUpdatedFlag(bool bFlag)
{
    unique_lock<mutex> lock1(mMtxPointCloudUpdated);
    mbPointCloudMapUpdated = bFlag;
}


bool PointCloudMapping::getGlobalCloud(PointCloud::Ptr &pCloud)
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    if (mpPclGlobalMap->empty())
        return false;
    pCloud = mpPclGlobalMap->makeShared();
    return true;
}


void PointCloudMapping::updateTbc(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    mTbc = Eigen::Matrix4d::Identity();
    mTbc.block(0,0,3,3) = Rbc;
    mTbc.block(0,3,3,1) = tbc;
    mfCameraHeight = tbc[2];
}


void PointCloudMapping::getTbc(Eigen::Matrix4d &Tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    Tbc = mTbc;
}
