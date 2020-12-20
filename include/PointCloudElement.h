//
// Created by wang on 2020/12/17.
//

#ifndef ORB_SLAM2_POINTCLOUDELEMENT_H
#define ORB_SLAM2_POINTCLOUDELEMENT_H

#include "pointcloudmapping.h"
#include "pcl/common/transforms.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "condition_variable"
#include "pcl/io/pcd_io.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <opencv2/core/core.hpp>
#include <mutex>

namespace ORB_SLAM2{

    class PointCloudE{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef pcl::PointXYZRGBA PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        PointCloud::Ptr pcE;
        Eigen::Isometry3d T;
        int pcID;

    };

}   // namespace ORB_SLAM2

#endif //ORB_SLAM2_POINTCLOUDELEMENT_H
