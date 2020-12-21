//
// Created by wang on 2020/12/17.
//

#ifndef ORB_SLAM2_POINTCLOUDMAPPING_H
#define ORB_SLAM2_POINTCLOUDMAPPING_H

// STL
#include <condition_variable>
#include <thread>

// PCL
#include "pcl/common/transforms.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/io/pcd_io.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include "pcl/visualization/cloud_viewer.h"

// ORB_SLAM2
#include "System.h"
#include "PointCloudElement.h"
#include "KeyFrame.h"

// OPENCV
#include <opencv2/core/core.hpp>

#include "SLIC.h"

using namespace std;
using namespace ORB_SLAM2;

class PointCloudMapping{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloudMapping(const std::string &strSettingPath, bool bUseViewer=true);

    void save();

    void insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, int idk, vector<KeyFrame*> vpKFs, double th);

    void shutdown();

    void run();

    void updateCloseLoopCloud();

    int loopcount = 0;
    vector<KeyFrame*> currentvpKFs;

    bool mbCloudBusy;
    bool mbLoopBusy;
    bool mbStop;

protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, double th);

    shared_ptr<thread> mThRunning;

    // Shutdown
    mutex shutDownMutex;
    bool mbShutDownFlag;

    // store keyframe
    condition_variable keyframeUpdated;
    mutex              keyFrameUpdatedMutex;
    vector<PointCloudE, Eigen::aligned_allocator<Eigen::Isometry3d>> pointcloud;

    // data to generate point cloud
    vector<KeyFrame*>   keyframes;
    vector<cv::Mat>     colorImgks;
    vector<int>         ids;
    mutex               keyframeMutex;
    uint16_t            lastkeyframeSize = 0;

    // statistical filter
    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
    double meanK_ = 50;
    double thresh_ = 2.0;

    // voxel grid filter
    pcl::VoxelGrid<PointT> voxel;
    double resolution_ = 0.05;

public:
    void setPointCloudMapUpdatedFlag(bool bFlag);
    bool getGlobalCloud(PointCloud::Ptr &pCloud);

    // ground <-- optical extrinsic matrix
    void updateTbc(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc);
    void getTbc(Eigen::Matrix4d &Tbc);

protected:

    // output point cloud
    mutex mMtxTbcUpdated;
    Eigen::Matrix4d mTbc;
    double mfCameraHeight;

    // point cloud updated complete flag
    mutex mMtxPointCloudUpdated;
    PointCloud::Ptr mpPclGlobalMap;
    bool mbPointCloudMapUpdated;

    // pcl viewer
    bool mbUseViewer;
    // pcl::visualization::CloudViewer mViewer;
    SLIC* slic;

};


#endif //ORB_SLAM2_POINTCLOUDMAPPING_H
