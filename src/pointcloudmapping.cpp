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

// ORB_SLAM2
#include "pointcloudmapping.h"
#include "KeyFrame.h"
#include "Converter.h"
#include "PointCloudElement.h"
#include "System.h"
#include "TicToc.h"

// STL
#include <chrono>

bool firstKF = true;
int currentloopcount = 0;

PointCloudMapping::PointCloudMapping(const std::string &strSettingPath, bool bUseViewer) :
        mbCloudBusy(false), mbLoopBusy(false), mbStop(false), mbShutDownFlag(false),
        mpPclGlobalMap(new PointCloudMapping::PointCloud()), mpPclObstacle(new PointCloud()), mpPclGroundPlane(new PointCloud()),
        mPlaneCoeffs(0, 0, 1, 0), mLastPlaneCoeffs(0, 0, 1, 0), mbPointCloudMapUpdated(false),
        mbUseViewer(bUseViewer)
// mViewer("viewer")
{
    // parse parameters
    cv::FileStorage fsSetting = cv::FileStorage(strSettingPath, cv::FileStorage::READ);

    // set initial Tbc: footprint<--optical
    Eigen::Vector3d tbc(0,0,0); //fsTbc["x"], fsTbc["y"], fsTbc["z"]);
    Eigen::Matrix3d Rbc;
    Rbc = Eigen::AngleAxisd(-M_PI/2,  Eigen::Vector3d::UnitX());
    updateTbc(Rbc, tbc);

    // voxel grid filter
    //resolution_ = fsPointCloudMapping["Resolution"];
    voxel.setLeafSize( resolution_, resolution_, resolution_);

    // statistical filter
    //cv::FileNode fsStatisticFilter = fsPointCloudMapping["StatisticFilter"];
    //meanK_ = fsStatisticFilter["MeanK"];
    //thresh_ = fsStatisticFilter["Thres"];
    statistical_filter.setMeanK(meanK_);
    statistical_filter.setStddevMulThresh(meanK_);

    // plane segmentation
    //cv::FileNode fsPlaneSegmentation = fsPointCloudMapping["PlaneSegmentation"];
    //mbUsePlaneSegmentation = true; //int(fsPlaneSegmentation["UsePlaneSegmentation"]);
    //mbSegmentPerFrame = true; // int(fsPlaneSegmentation["SegmentPerFrame"]);
    mfPlaneDistThres = 0.15; //fsPlaneSegmentation["PlaneDistThres"];
    mfFramePlaneDistThres = 0.5; //fsPlaneSegmentation["FramePlaneDistThres"];

    cout << "---" << endl;
    cout << "Point Cloud Thread Parameters:" << endl;
    cout << "- Tbc: " << endl << mTbc << endl;
    cout << "- CameraHeight: " << mfCameraHeight << endl;
    cout << "- Resolution: " << resolution_ << endl;
    cout << "- StatisticFilter " << endl;
    cout << "\t- MeanK: " <<  meanK_ << endl;
    cout << "\t- Thres: " << thresh_ << endl;
    cout << "- PlaneSegmentation: " << endl;
    cout << "\t- UsePlaneSegmentation: " << mbUsePlaneSegmentation << endl;
    cout << "\t- SegmentPerFrame: " << mbSegmentPerFrame << endl;
    cout << "\t- PlaneDistThres: " << mfPlaneDistThres << endl;
    cout << "\t- FramePlaneDistThres: " << mfFramePlaneDistThres << endl;

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
    pcl::visualization::CloudViewer viewer("point cloud map");
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
        PointCloud::Ptr pNewPlaneCloud(new PointCloud());
        for ( size_t i=lastkeyframeSize; i<N ; i++ ) // for new keyframes
        {
            PointCloud::Ptr p (new PointCloud);
            pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());
            pcl::transformPointCloud( *p, *p, Tbc);
            *pNewCloud += *p;
        }

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
        viewer.showCloud( mpPclGlobalMap );
        //cout<<"showing"<<endl;
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
    unique_lock<mutex> lock2(mMtxPlaneSegmentation);
    mbPointCloudMapUpdated = bFlag;
}


bool PointCloudMapping::getPointCloudMapUpdatedFlag()
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    unique_lock<mutex> lock2(mMtxPlaneSegmentation);
    return mbPointCloudMapUpdated;
}


bool PointCloudMapping::getGlobalCloud(PointCloud::Ptr &pCloud)
{
    unique_lock<mutex> lock(mMtxPointCloudUpdated);
    if (mpPclGlobalMap->empty())
        return false;
    pCloud = mpPclGlobalMap->makeShared();
    return true;
}


bool PointCloudMapping::getObstacleCloud(PointCloud::Ptr &pCloud)
{
    unique_lock<mutex> lock(mMtxPlaneSegmentation);
    if (!mbUsePlaneSegmentation || mpPclObstacle->empty())
        return false;
    pCloud = mpPclObstacle->makeShared();
    return true;
}


bool PointCloudMapping::getPlaneCloud(PointCloud::Ptr &pCloud)
{
    unique_lock<mutex> lock(mMtxPlaneSegmentation);
    if (!mbUsePlaneSegmentation || mpPclGroundPlane->empty())
        return false;
    pCloud = mpPclGroundPlane->makeShared();
    return true;
}


bool PointCloudMapping::getPlaneCoeffs(Eigen::Vector4d &planeCoeffs)
{
    unique_lock<mutex> lock(mMtxPlaneSegmentation);
    if (!mbUsePlaneSegmentation)
        return false;
    planeCoeffs = mPlaneCoeffs;
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


void PointCloudMapping::updateTbc(const Eigen::Matrix4d &Tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    mTbc = Tbc;
    mfCameraHeight = Tbc(3,3);
}


void PointCloudMapping::getTbc(Eigen::Matrix4d &Tbc)
{
    unique_lock<mutex> lock(mMtxTbcUpdated);
    Tbc = mTbc;
}


void PointCloudMapping::getPlaneTransformMatrix(const Eigen::Vector4d &target_plane,
                                                const Eigen::Vector4d &source_plane, Eigen::Matrix4d &T)
{
    if ((target_plane-source_plane).norm() < 1e-6)
    {
        T = Eigen::Matrix4d::Identity();
        return;
    }
    Eigen::Vector3d norm_target = target_plane.segment(0, 3);
    Eigen::Vector3d norm_source = source_plane.segment(0, 3);
    double th_mo = acos (norm_source.dot(norm_target) / (norm_source.norm() *norm_target.norm()) + 1e-6 );  // [0, pi)
    Eigen::Matrix3d Rmo;
    if (fabs(th_mo) < 1e-3 || (norm_source - norm_target).norm() < 1e-6)
    {
        Rmo = Eigen::Matrix3d::Identity();
    }
    else
    {
        Eigen::Vector3d norm_mo = norm_source.cross(norm_target).normalized();  // get the norm vector from odom->map
        Eigen::AngleAxisd r_mo(th_mo, norm_mo);
        Rmo = r_mo.toRotationMatrix();
    }
    double tz = (source_plane[3]-target_plane[3]) / (target_plane[2] + 1e-6);
    T = Eigen::Matrix4d::Identity();
    T.block(0,0,3,3) = Rmo;
    T(2,3) = tz;
}


void PointCloudMapping::setUsePlaneSegmentationFlag(bool bFlag)
{
    unique_lock<mutex> lock1(mMtxPointCloudUpdated);
    unique_lock<mutex> lock2(mMtxPlaneSegmentation);
    mbUsePlaneSegmentation = bFlag;
}


bool PointCloudMapping::getUsePlanSegmentationFlag()
{
    unique_lock<mutex> lock1(mMtxPointCloudUpdated);
    unique_lock<mutex> lock2(mMtxPlaneSegmentation);
    return mbUsePlaneSegmentation;
}


bool PointCloudMapping::planeSACSegmentation(PointCloud::Ptr &pPclMap, PointCloud::Ptr &pPclPlane,
                                             Eigen::Vector4d &planeCoeffs, const Eigen::Vector4d &lastPlaneCoeffs,
                                             pcl::PointIndices::Ptr &pInliers)
{
    // segment plane with RANSAC
    pcl::ModelCoefficients::Ptr pcl_coeffs(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    pcl::SACSegmentation<PointCloudMapping::PointT> seg;

    // perform segment
    seg.setOptimizeCoefficients(true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (mfPlaneDistThres);  // 距离阈值
    seg.setInputCloud(pPclMap);
    seg.segment(*inliers, *pcl_coeffs);

    // segment directly fail
    if (inliers->indices.size() == 0)
    {
        cerr << "Plane segmentation fail!" << endl;
        return false;
    }

    planeCoeffs[0] = pcl_coeffs->values[0];
    planeCoeffs[1] = pcl_coeffs->values[1];
    planeCoeffs[2] = pcl_coeffs->values[2];
    planeCoeffs[3] = pcl_coeffs->values[3];

    // norm vector
    Eigen::Vector3d norm_plane = planeCoeffs.segment(0, 3);
    Eigen::Vector3d norm_last_plane = lastPlaneCoeffs.segment(0, 3);
    planeCoeffs = planeCoeffs / norm_plane.norm();  // normalize sacle

    // check angle between current estimated plane and last estimated plane
    double angle_last = acos( norm_plane.dot(norm_last_plane) / (norm_plane.norm()*norm_last_plane.norm()) );
    if (fabs(angle_last) > M_PI_2 )
    {
        // in pi=[n|d], always d>0, n determines whether oringial point is up or under the plane
        // we should keep n in the same direction as last norm (always positive)
        planeCoeffs = -planeCoeffs;
        norm_plane = planeCoeffs.segment(0, 3);
        angle_last = acos( norm_plane.dot(norm_last_plane) / (norm_plane.norm()*norm_last_plane.norm()) );
    }
    if ( fabs(angle_last) > M_PI/12 )
    {
        cerr << "Segment ground plane fail, "
                "fail segment parameters: " << planeCoeffs.transpose()
             << ", relative angle: " << angle_last << endl;
        planeCoeffs = lastPlaneCoeffs;  // roll back...
        return false;
    }

    // set indices
    if (pInliers != nullptr)
        pInliers = inliers;

    // filter plane
    PointCloudMapping::PointCloud::Ptr pcl_obstalce(new PointCloudMapping::PointCloud() );
    pcl::ExtractIndices<PointCloudMapping::PointT> extract;
    extract.setInputCloud(pPclMap);
    extract.setIndices(inliers);
    extract.filter(*pPclPlane);

    // filter map
    extract.setNegative(true);
    extract.setInputCloud(pPclMap);
    extract.setIndices(inliers);
    extract.filter(*pcl_obstalce);

    // output
    pPclMap = pcl_obstalce;
    cout << "Segment ground plane succeed, "
            "successful segment parameters: " << planeCoeffs.transpose()
         << ", relative angle: " << angle_last << endl;
    return true;
}


bool PointCloudMapping::framePlaneSegmentation(PointCloud::Ptr &pPclFrame, PointCloud::Ptr &pPclPlane)
{
    // pass through-->sac segmentation-->indices filter
    // pass through filter with y axis
    PointCloud::Ptr pFramePassFilter(new PointCloud());
    vector<int> vFramePassIdxs;
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(pPclFrame);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-mfFramePlaneDistThres, +mfFramePlaneDistThres);
    pass.filter(*pFramePassFilter);
    pass.filter(vFramePassIdxs);

    // mViewer.showCloud(pFramePassFilter);

    if (vFramePassIdxs.empty())
    {
        cerr << "No valid point in pass through filter range!" << endl;
        return false;
    }

    // perform plane segmentation
    Eigen::Vector4d planeCoeffs;
    Eigen::Vector4d lastPlaneCoeffs(0, 0, 1, 0);  // z axis norm vector
    pcl::PointIndices::Ptr pInliers(new pcl::PointIndices());
    bool bPlaneSegmentation = planeSACSegmentation(pFramePassFilter, pPclPlane, planeCoeffs, lastPlaneCoeffs, pInliers);
    if (!bPlaneSegmentation)
        return false;

    // remove plane from original point cloud
    pcl::PointIndices::Ptr pFrameInliers(new pcl::PointIndices());
    for (int i=0; i<pInliers->indices.size(); i++)
    {
        pFrameInliers->indices.push_back(vFramePassIdxs[pInliers->indices[i]]);
    }
    PointCloud::Ptr pFramePlaneFilter(new PointCloud());
    pcl::ExtractIndices<PointT> extractor;
    extractor.setIndices(pFrameInliers);
    extractor.setInputCloud(pPclFrame);
    extractor.setNegative(true);
    extractor.filter(*pFramePlaneFilter);

    // output
    pPclFrame = pFramePlaneFilter;
//    mViewer.showCloud(pPclFrame);
    return true;
}


