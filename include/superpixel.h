//
// Created by wang on 2020/12/17.
//

#ifndef ORB_SLAM2_SUPERPIXEL_H
#define ORB_SLAM2_SUPERPIXEL_H

struct Superpixel_seed
{
    float x, y;
    float size;
    float mean_depth;
    //float mean_intensity;
    bool stable;
    float R,G,B;

    // for debug
    float min_eigen_value;
    float max_eigen_value;
};

#endif //ORB_SLAM2_SUPERPIXEL_H
